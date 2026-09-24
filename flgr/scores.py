"""Baseline attribution scores (data- and parameter-level) computed from saved snapshots."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .learners import masked_logits
from .ledger import ProbeSet, probe_grads
from .utils import flat, params_of, set_flat_params


def per_sample_grads(model, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor], chunk: int = 64) -> torch.Tensor:
    """[n, P] per-sample gradients in eval mode (vmap). Use ``chunk`` to bound memory."""
    model.eval()
    if getattr(model, "no_vmap", False):                       # e.g. HF language models: one backward per sample
        rows = []
        for i in range(len(x)):
            out = model(x[i:i + 1])
            m_ = mask[i:i + 1] if mask is not None else None
            l = F.cross_entropy(masked_logits(out, m_), y[i:i + 1])
            rows.append(flat(torch.autograd.grad(l, params_of(model))).detach())
        return torch.stack(rows)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n: b for n, b in model.named_buffers()}

    def loss_fn(prm, xi, yi, mi):
        out = functional_call(model, (prm, buffers), (xi.unsqueeze(0),))
        if mi is not None:
            out = out.masked_fill(~mi.unsqueeze(0), -1e9)
        return F.cross_entropy(out, yi.unsqueeze(0))

    outs = []
    for i in range(0, len(x), chunk):
        mi = mask[i:i + chunk] if mask is not None else None
        if mi is None:
            g = vmap(grad(lambda prm, a, b: loss_fn(prm, a, b, None)), in_dims=(None, 0, 0))(params, x[i:i + chunk], y[i:i + chunk])
        else:
            g = vmap(grad(loss_fn), in_dims=(None, 0, 0, 0))(params, x[i:i + chunk], y[i:i + chunk], mi)
        outs.append(torch.cat([g[n].reshape(len(x[i:i + chunk]), -1) for n in names], 1))
    return torch.cat(outs)


def _task_inputs(sc, t: int, epoch: int = 0):
    tk = sc.tasks[t]
    x = sc.inputs(sc.xtr[tk.train_idx], t)             # no augmentation for scoring
    y = sc.target(sc.ytr[tk.train_idx])
    st = torch.full((len(y),), t, device=sc.device)
    return x, y, sc.logit_mask(t, st, n=len(y))


def grad_dot_scores(model, sc, t: int, Gm: torch.Tensor, chunk: int = 64, want_cos: bool = False,
                    R: Optional[torch.Tensor] = None):
    """For every training sample i of task t: s[i,g] = grad l_i . G_g, optionally the cosine,
    the per-sample gradient norm and a random projection R^T grad l_i (TRAK features)."""
    x, y, mask = _task_inputs(sc, t)
    if getattr(model, "no_vmap", False):
        return _dots_double_backward(model, x, y, mask, Gm, chunk=int(getattr(model, "score_chunk", 16)))
    dots, coss, norms, projs = [], [], [], []
    gn = Gm.norm(dim=1).clamp_min(1e-12)
    for i in range(0, len(x), chunk):
        sl = slice(i, i + chunk)
        g = per_sample_grads(model, x[sl], y[sl], mask[sl] if mask is not None else None, chunk)
        d = g @ Gm.T
        dots.append(d)
        n = g.norm(dim=1)
        norms.append(n)
        if want_cos:
            coss.append(d / (n.clamp_min(1e-12).view(-1, 1) * gn.view(1, -1)))
        if R is not None:
            projs.append(g @ R)
    return (torch.cat(dots), torch.cat(coss) if want_cos else None, torch.cat(norms),
            torch.cat(projs) if R is not None else None)


def _dots_double_backward(model, x, y, mask, Gm: torch.Tensor, chunk: int = 16):
    """grad l_i . G_g for all i without per-sample gradients: d/du_i [G_g . grad sum_j u_j l_j] (double backward).
    Returns (dots [N,G], None, NaN norms, None) -- cosine and projections need per-sample gradients."""
    model.eval()
    out = []
    eye = torch.eye(Gm.shape[0], device=Gm.device)
    for i in range(0, len(x), chunk):
        xb, yb = x[i:i + chunk], y[i:i + chunk]
        mb = mask[i:i + chunk] if mask is not None else None
        u = torch.ones(len(yb), device=Gm.device, requires_grad=True)
        ce = F.cross_entropy(masked_logits(model(xb), mb), yb, reduction="none")
        grads = torch.autograd.grad((ce * u).sum(), params_of(model), create_graph=True)
        h = Gm @ flat(grads)
        try:
            D = torch.autograd.grad(h, u, grad_outputs=eye, is_grads_batched=True)[0]
        except RuntimeError:
            D = torch.stack([torch.autograd.grad(h[k], u, retain_graph=k < len(h) - 1)[0] for k in range(len(h))])
        out.append(D.T.detach())
    dots = torch.cat(out)
    return dots, None, torch.full((len(x),), float("nan"), device=dots.device), None


def trak_harm(Psi: torch.Tensor, phi: torch.Tensor, lam_rel: float = 1e-3) -> torch.Tensor:
    """Projected-influence harm of each training sample on each probe group (one checkpoint).

    Influence of up-weighting sample i on L_g is -grad L_g^T H^-1 grad l_i (Koh & Liang); TRAK
    replaces H by the projected empirical-Fisher kernel Psi^T Psi. Harm = L_g increase caused by
    including i = -phi_g^T (Psi^T Psi + lam I)^-1 psi_i.   Psi [N,k], phi [G,k] -> [N,G]."""
    K = Psi.T @ Psi
    lam = lam_rel * torch.trace(K) / K.shape[0]
    Kinv_phi = torch.linalg.solve(K + lam * torch.eye(K.shape[0], dtype=K.dtype, device=K.device), phi.T)   # [k,G]
    return -(Psi @ Kinv_phi)


@torch.no_grad()
def margin_scores(model, sc, t: int) -> torch.Tensor:
    """Correct-class logit minus the largest other logit, at the current parameters."""
    model.eval()
    x, y, mask = _task_inputs(sc, t)
    out = torch.cat([model(x[i:i + 1000]) for i in range(0, len(x), 1000)])
    out = masked_logits(out, mask)
    corr = out.gather(1, y.view(-1, 1)).squeeze(1)
    other = out.scatter(1, y.view(-1, 1), -1e9).max(1).values
    return corr - other


@torch.no_grad()
def loss_scores(model, sc, t: int) -> torch.Tensor:
    model.eval()
    x, y, mask = _task_inputs(sc, t)
    out = torch.cat([model(x[i:i + 1000]) for i in range(0, len(x), 1000)])
    return F.cross_entropy(masked_logits(out, mask), y, reduction="none")


@torch.no_grad()
def feature_proximity(model, sc, t: int, probe: ProbeSet) -> torch.Tensor:
    """cos(feature(x_i), centroid of old group g) at the current parameters -> [N, G]."""
    model.eval()
    x, _, _ = _task_inputs(sc, t)
    f = torch.cat([model.features(x[i:i + 1000]) for i in range(0, len(x), 1000)])
    fp = model.features(probe.x)
    cent = probe.W @ fp
    return F.normalize(f, dim=1) @ F.normalize(cent, dim=1).T


@torch.no_grad()
def feature_class_similarity(model, sc, t: int, probe: ProbeSet) -> torch.Tensor:
    """Cosine similarity between new-class feature centroids and old-group centroids -> [C_new, G]."""
    model.eval()
    x, _, _ = _task_inputs(sc, t)
    ylab = sc.ytr[sc.tasks[t].train_idx]
    f = torch.cat([model.features(x[i:i + 1000]) for i in range(0, len(x), 1000)])
    cents = torch.stack([f[ylab == c].mean(0) for c in sc.tasks[t].classes])
    old = probe.W @ model.features(probe.x)
    return F.normalize(cents, dim=1) @ F.normalize(old, dim=1).T


def fisher_diag(model, probe: ProbeSet, chunk: int = 64) -> torch.Tensor:
    g = per_sample_grads(model, probe.x, probe.y, probe.mask, chunk)
    return (g ** 2).mean(0)
