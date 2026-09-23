"""The Forgetting Ledger: exact per-step decomposition of old-knowledge loss changes.

For one SGD step with objective sum_i l_i(theta) (weighted per-sample terms) and update
Delta = -lr * P(grad), the change of every old-group probe loss L_g is decomposed as

    L_g(theta', s') - L_g(theta, s)
        = [L_g(theta, s') - L_g(theta, s)]                        running-statistics drift (exact)
        + gbar_g . Delta                                           parameter path (trapezoid / Simpson)
        + O(|Delta|^3)                                             integration error

and the parameter-path term is split exactly over parameters (elementwise product) and over
loss terms (gbar_g . Delta = -lr * sum_i grad l_i . P^T gbar_g). Per-term inner products are
obtained with a double-backward identity:  d/du_i [ (grad_theta sum_j u_j l_j) . v ] = grad l_i . v.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.func import functional_call

from .learners import Projection, Terms, masked_logits
from .models import unit_index
from .utils import flat, params_of


class ProbeSet:
    """Concatenated probe data of all old tasks + group-averaging matrix W [G, Np]."""

    def __init__(self, sc, upto_task: int, current_task: int, group_by: str, device):
        self.sc, self.device = sc, device
        xs, ys, ts = [], [], []
        for t in range(upto_task):
            x, y, _ = sc.probe_batch(t)
            xs.append(x); ys.append(y); ts.append(torch.full((len(y),), t, device=device))
        self.x, self.y, self.task = torch.cat(xs), torch.cat(ys), torch.cat(ts)
        self.groups = sc.groups(upto_task, group_by)
        G, N = len(self.groups), len(self.y)
        W = torch.zeros(G, N, device=device)
        for g, grp in enumerate(self.groups):
            W[g, grp["pos"]] = 1.0 / len(grp["pos"])
        self.W = W
        self.mask = sc.logit_mask(current_task, self.task, n=N)

    @property
    def G(self) -> int:
        return self.W.shape[0]

    def losses(self, logits: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(masked_logits(logits, self.mask), self.y, reduction="none")


def _call(model, params: Optional[Dict[str, torch.Tensor]], x):
    if params is None:
        return model(x)
    return functional_call(model, params, (x,))


def probe_grads(model, probe: ProbeSet, flat_theta: Optional[torch.Tensor] = None, chunk: int = 0):
    """Group losses [G] and group gradients [G, P] at theta (current params if flat_theta is None).
    Always evaluated in eval mode (the model as it would be tested)."""
    was = model.training
    model.eval()
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    plist = params_of(model)
    if flat_theta is None:
        leaves = plist
        pdict = None
    else:
        leaves, i = [], 0
        for p in plist:
            leaves.append(flat_theta[i:i + p.numel()].view_as(p).detach().requires_grad_(True)); i += p.numel()
        pdict = dict(zip(names, leaves))
    ce = probe.losses(_call(model, pdict, probe.x))
    Lg = probe.W @ ce.detach()
    try:
        gs = torch.autograd.grad(ce, leaves, grad_outputs=probe.W, is_grads_batched=True)
        Gm = torch.cat([g.reshape(probe.G, -1) for g in gs], 1)
    except RuntimeError:                                            # fallback: one backward per group
        rows = []
        for g in range(probe.G):
            gg = torch.autograd.grad(ce, leaves, grad_outputs=probe.W[g], retain_graph=g < probe.G - 1)
            rows.append(flat(gg))
        Gm = torch.stack(rows)
    model.train(was)
    return Lg, Gm.detach()


def probe_losses(model, probe: ProbeSet) -> torch.Tensor:
    was = model.training
    model.eval()
    with torch.no_grad():
        Lg = probe.W @ probe.losses(model(probe.x))
    model.train(was)
    return Lg


class LedgerAccumulator:
    """Holds all ledger tensors of one task."""

    def __init__(self, model, n_new: int, n_mem_keys: int, G: int, device, store_param_groups: bool = False):
        P = sum(p.numel() for p in params_of(model))
        unit_of, meta, layer_of, lnames = unit_index(model)
        self.unit_of, self.layer_of = unit_of.to(device), layer_of.to(device)
        self.unit_meta, self.layer_names = meta, lnames
        U, L = int(unit_of.max()) + 1, len(lnames)
        z = lambda *s: torch.zeros(*s, device=device, dtype=torch.float64)
        self.data = z(n_new, G)                # new-task samples
        self.mem = z(n_mem_keys, G) if n_mem_keys else None   # replay memories keyed by task*N + gidx
        self.reg = z(G)                        # regulariser terms
        self.stats = z(G)                      # running-statistics drift
        self.param = z(P)                      # per parameter, summed over groups
        self.param_g = z(G, P) if store_param_groups else None
        self.unit = z(U, G)
        self.layer = z(L, G)
        self.path_g = z(G)                     # parameter-path total per group
        self.learn_param = z(P)                # credit for decreasing the training objective
        self.learn_unit = z(U)
        self.true_dL = z(G)
        self.L0: Optional[torch.Tensor] = None
        self.steps = 0
        self.n_evals = 0                       # probe-gradient evaluations (cost)
        self.path_var = z(G)                   # sum_t |L_g(t+1) - L_g(t)|  (total variation of the path)
        self.step_err: List[float] = []
        self.early: Dict[str, torch.Tensor] = {}

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k not in ("unit_meta",)}

    def load_state_dict(self, d):
        self.__dict__.update(d)


def path_average_gradient(model, probe: ProbeSet, theta: torch.Tensor, delta: torch.Tensor,
                          G0: torch.Tensor, G1: torch.Tensor, dL_exact: torch.Tensor, rule: str,
                          tol: float = 1e-4, max_intervals: int = 8):
    """Average of the probe-group gradients along the segment theta -> theta + delta.

    ``adaptive`` uses composite Simpson quadrature and doubles the number of sub-intervals until
    the integrated change matches the *exactly known* endpoint change dL_exact (per group) within
    ``tol + tol * |dL_exact|``, or ``max_intervals`` is reached. Returns (gbar [G,P], n_evals).
    """
    if rule == "euler":
        return G0, 0
    if rule == "trapezoid":
        return 0.5 * (G0 + G1), 0
    if max_intervals & (max_intervals - 1) or max_intervals < 2:
        raise ValueError("max_intervals must be a power of two >= 2")
    nodes = {0: G0, max_intervals: G1}
    n, evals = 2, 0
    while True:
        for k in range(n + 1):
            key = k * (max_intervals // n)
            if key not in nodes:
                _, nodes[key] = probe_grads(model, probe, theta + (k / n) * delta)
                evals += 1
        w = torch.tensor([1.0] + [4.0 if k % 2 else 2.0 for k in range(1, n)] + [1.0], device=G0.device) / (3.0 * n)
        gbar = sum(w[k] * nodes[k * (max_intervals // n)] for k in range(n + 1))
        if rule == "simpson" or n >= max_intervals:
            return gbar, evals
        err = (gbar @ delta - dL_exact).abs()
        if bool((err <= tol + tol * dL_exact.abs()).all()):
            return gbar, evals
        n *= 2


def ledger_step(model, lr: float, terms: Terms, grads_graph: List[torch.Tensor], u: torch.Tensor,
                proj: Optional[Projection], G_start: torch.Tensor, probe: ProbeSet, acc: LedgerAccumulator,
                rule: str = "adaptive", frozen: Optional[torch.Tensor] = None, L_start: Optional[torch.Tensor] = None,
                tol: float = 1e-4, max_intervals: int = 8):
    """Perform the ledger bookkeeping for one step and return (Delta, L_end, G_end).

    ``grads_graph`` are the parameter gradients of sum_i u_i l_i built with create_graph=True.
    Parameters are NOT modified here (the caller applies Delta afterwards).
    """
    gflat = flat(grads_graph)
    g = gflat.detach()
    d = g
    coef = 0.0
    if proj is not None and proj.active:
        coef = float(g @ proj.g_ref) / proj.ref_norm2
        d = g - coef * proj.g_ref
    if frozen is not None:
        d = d * (~frozen)
    delta = -lr * d
    theta = flat(params_of(model)).detach()
    L_end, G_end = probe_grads(model, probe, theta + delta)
    dL_exact = (L_end - L_start) if L_start is not None else torch.zeros_like(L_end)
    gbar, n_ev = path_average_gradient(model, probe, theta, delta, G_start, G_end, dL_exact, rule, tol, max_intervals)
    acc.n_evals += n_ev + 1
    acc.path_var += dL_exact.abs().double()
    # ---- parameter / unit / layer attribution
    pg = gbar * delta.unsqueeze(0)                                   # [G, P]
    acc.param += pg.sum(0).double()
    if acc.param_g is not None:
        acc.param_g += pg.double()
    acc.unit.index_add_(0, acc.unit_of, pg.T.double())
    acc.layer.index_add_(0, acc.layer_of, pg.T.double())
    acc.path_g += pg.sum(1).double()
    lp = (-(d) * delta)                                              # objective-decrease credit
    acc.learn_param += lp.double()
    acc.learn_unit.index_add_(0, acc.unit_of, lp.double())
    # ---- per-term attribution via double backward
    V = gbar if frozen is None else gbar * (~frozen).unsqueeze(0)
    extra = proj is not None and proj.active
    if extra:
        V = torch.cat([V, proj.g_ref.unsqueeze(0)])
    h = V @ gflat                                                    # [G(+1)] scalars, differentiable in u
    eye = torch.eye(len(h), device=h.device)
    try:
        D = torch.autograd.grad(h, u, grad_outputs=eye, is_grads_batched=True)[0]   # [G(+1), m]
    except RuntimeError:
        D = torch.stack([torch.autograd.grad(h[i], u, retain_graph=i < len(h) - 1)[0] for i in range(len(h))])
    D = D.T                                                          # [m, G(+1)]
    if extra:
        gref_dot = (proj.g_ref.unsqueeze(0) * V[:-1]).sum(1) if frozen is None else (proj.g_ref * (~frozen)) @ V[:-1].T
        contrib = -lr * (D[:, :-1] - D[:, -1:] * (gref_dot.view(1, -1) / proj.ref_norm2))
    else:
        contrib = -lr * D
    contrib = contrib.double()
    for kind, ids, sl in zip(terms.kinds, terms.ids, terms.slices):
        c = contrib[sl]
        if kind == "new":
            acc.data.index_add_(0, ids, c)
        elif kind == "mem":
            acc.mem.index_add_(0, ids, c)
        else:
            acc.reg += c.sum(0)
    acc.steps += 1
    return delta, L_end, G_end
