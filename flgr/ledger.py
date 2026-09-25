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


def _call(model, params: Optional[Dict[str, torch.Tensor]], x, buffers: Optional[Dict[str, torch.Tensor]] = None):
    if params is None and buffers is None:
        return model(x)
    if params is None:
        params = {n: p for n, p in model.named_parameters()}
    if buffers is None:
        return functional_call(model, params, (x,))
    return functional_call(model, (params, buffers), (x,))


def probe_grads(model, probe: ProbeSet, flat_theta: Optional[torch.Tensor] = None,
                buffers: Optional[Dict[str, torch.Tensor]] = None):
    """Group losses [G] and group gradients [G, P] at theta (current params if flat_theta is None)
    and normalisation buffers (current if None). Always evaluated in eval mode (as at test time)."""
    was = model.training
    model.eval()
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    plist = params_of(model)
    if flat_theta is None:
        leaves = plist
        pdict = None if buffers is None else dict(zip(names, leaves))
    else:
        leaves, i = [], 0
        for p in plist:
            leaves.append(flat_theta[i:i + p.numel()].view_as(p).detach().requires_grad_(True)); i += p.numel()
        pdict = dict(zip(names, leaves))
    ce = probe.losses(_call(model, pdict, probe.x, buffers))
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
        self.step_nev: List[int] = []            # quadrature evaluations per step (== max_intervals - 1: capped)
        self.early: Dict[str, torch.Tensor] = {}
        self.data_euler = z(n_new, G)          # optional: per-sample Euler (idealised TracIn) attribution
        self.sample_loss: List[torch.Tensor] = []   # per-epoch per-sample training loss (anatomy analysis)
        self.sample_correct: List[torch.Tensor] = []

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k not in ("unit_meta",)}

    def load_state_dict(self, d):
        self.__dict__.update(d)


def path_average_gradient(model, probe: ProbeSet, theta: torch.Tensor, delta: torch.Tensor,
                          G0: torch.Tensor, G1: torch.Tensor, dL_exact: torch.Tensor, rule: str,
                          tol: float = 1e-3, max_intervals: int = 8, buffers=None):
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
                _, nodes[key] = probe_grads(model, probe, theta + (k / n) * delta, buffers)
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
                tol: float = 1e-3, max_intervals: int = 8, also_euler: bool = False, shapley: Optional[dict] = None):
    """Perform the ledger bookkeeping for one step and return (Delta, L_end, G_end).

    ``grads_graph`` are the parameter gradients of sum_i u_i l_i built with create_graph=True;
    parameters are NOT modified here (the caller applies Delta afterwards).
    ``G_start``/``L_start`` are evaluated at (theta_t, s_{t+1}).
    ``shapley`` (models with running statistics): dict(buffers_old, G_old, L_old) at (theta_t, s_t);
    the step is then decomposed symmetrically (Shapley average over the two orders of
    'statistics first' and 'parameters first'), which removes the arbitrariness of the order.
    ``also_euler``: additionally accumulate the per-sample Euler (left-point, idealised TracIn)
    attribution in ``acc.data_euler`` for comparison.
    """
    gflat = flat(grads_graph)
    g = gflat.detach()
    d = g
    if proj is not None and proj.active:
        d = g - (float(g @ proj.g_ref) / proj.ref_norm2) * proj.g_ref
    if frozen is not None:
        d = d * (~frozen)
    delta = -lr * d
    theta = flat(params_of(model)).detach()
    L_end, G_end = probe_grads(model, probe, theta + delta)
    dL_exact = (L_end - L_start) if L_start is not None else torch.zeros_like(L_end)
    gbar, n_ev = path_average_gradient(model, probe, theta, delta, G_start, G_end, dL_exact, rule, tol, max_intervals)
    acc.n_evals += n_ev + 1
    acc.step_nev.append(int(n_ev))
    if shapley is not None:
        # path at the old statistics, and the symmetric statistics term
        L_end0, G_end0 = probe_grads(model, probe, theta + delta, shapley["buffers_old"])
        dL0 = L_end0 - shapley["L_old"]
        gbar0, n_ev0 = path_average_gradient(model, probe, theta, delta, shapley["G_old"], G_end0, dL0, rule, tol,
                                             max_intervals, shapley["buffers_old"])
        acc.n_evals += n_ev0 + 1
        gbar = 0.5 * (gbar + gbar0)
        acc.stats += (0.5 * ((L_start - shapley["L_old"]) + (L_end - L_end0))).double()
    else:
        pass                                                         # statistics-first term added by the caller
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
    G = gbar.shape[0]
    blocks = [gbar]
    if also_euler:
        blocks.append(G_start)
    extra = proj is not None and proj.active
    V = torch.cat(blocks)
    if frozen is not None:
        V = V * (~frozen).unsqueeze(0)
    if extra:
        V = torch.cat([V, proj.g_ref.unsqueeze(0)])
    h = V @ gflat                                                    # scalars, differentiable in u
    eye = torch.eye(len(h), device=h.device)
    try:
        D = torch.autograd.grad(h, u, grad_outputs=eye, is_grads_batched=True)[0]
    except RuntimeError:
        D = torch.stack([torch.autograd.grad(h[i], u, retain_graph=i < len(h) - 1)[0] for i in range(len(h))])
    D = D.T                                                          # [m, rows]
    if extra:
        gref = proj.g_ref if frozen is None else proj.g_ref * (~frozen)
        gref_dot = V[:-1] @ gref                                     # [rows-1]
        contrib_all = -lr * (D[:, :-1] - D[:, -1:] * (gref_dot.view(1, -1) / proj.ref_norm2))
    else:
        contrib_all = -lr * D
    contrib = contrib_all[:, :G].double()
    for kind, ids, sl in zip(terms.kinds, terms.ids, terms.slices):
        c = contrib[sl]
        if kind == "new":
            acc.data.index_add_(0, ids, c)
            if also_euler:
                acc.data_euler.index_add_(0, ids, contrib_all[sl, G:2 * G].double())
        elif kind == "mem":
            acc.mem.index_add_(0, ids, c)
        else:
            acc.reg += c.sum(0)
    acc.steps += 1
    return delta, L_end, G_end
