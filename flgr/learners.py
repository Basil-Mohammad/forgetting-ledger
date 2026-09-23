"""Continual learners expressed as *weighted per-sample loss terms*.

Every learner returns, for one optimisation step, a vector ``terms`` whose sum is the batch
objective, together with a description of which source each entry belongs to. This is what
allows the ledger to split every SGD step exactly across new samples, replayed memories and
regularisers. A-GEM additionally returns its (linear, given the conflict test) projection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .data import STREAM_BUFFER, Scenario
from .utils import flat, hash_uniform, params_of


@dataclass
class Terms:
    values: torch.Tensor                        # [m] per-term losses (already weighted); sum = objective
    kinds: List[str]                            # per segment: "new" | "mem" | "reg"
    ids: List[torch.Tensor]                     # per segment: sample ids (new: position in task; mem: global idx)
    slices: List[slice]
    n_new: int = 0


@dataclass
class Projection:                               # A-GEM: d = g - coef * g_ref  (coef = (g.g_ref)/|g_ref|^2 or 0)
    g_ref: torch.Tensor
    active: bool
    ref_norm2: float


def masked_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return logits
    return logits.masked_fill(~mask, -1e9)


class HashBuffer:
    """Bottom-k hash reservoir: keeps the M seen samples with the smallest hash priority.

    Equivalent in distribution to reservoir sampling, but a sample's membership never depends
    on which *other* samples were seen, so removing training samples leaves the rest intact.
    """

    def __init__(self, size: int, seed: int, n_out: int, device):
        self.size, self.seed, self.device = size, seed, device
        self.gidx = torch.empty(0, dtype=torch.long, device=device)
        self.task = torch.empty(0, dtype=torch.long, device=device)
        self.prio = torch.empty(0, device=device)
        self.logits = torch.empty(0, n_out, device=device)

    def __len__(self):
        return len(self.gidx)

    def add(self, gidx: torch.Tensor, task: int, logits: Optional[torch.Tensor] = None):
        pr = hash_uniform(self.seed, STREAM_BUFFER, task, 0, gidx).squeeze(1)
        new_logits = logits.detach() if logits is not None else torch.zeros(len(gidx), self.logits.shape[1], device=self.device)
        g = torch.cat([self.gidx, gidx]); t = torch.cat([self.task, torch.full_like(gidx, task)])
        p = torch.cat([self.prio, pr]); l = torch.cat([self.logits, new_logits])
        # de-duplicate (a sample seen in several epochs keeps its first entry)
        key = g * 1000 + t
        _, first = _unique_first(key)
        g, t, p, l = g[first], t[first], p[first], l[first]
        keep = torch.argsort(p)[: self.size]
        self.gidx, self.task, self.prio, self.logits = g[keep], t[keep], p[keep], l[keep]

    def sample(self, n: int, key_a: int, key_b: int):
        if len(self) == 0:
            return None
        u = hash_uniform(self.seed, STREAM_BUFFER + 1, key_a, key_b, torch.arange(n, device=self.device)).squeeze(1)
        return (u * len(self)).long().clamp_(max=len(self) - 1)

    def state_dict(self):
        return dict(gidx=self.gidx, task=self.task, prio=self.prio, logits=self.logits)

    def load_state_dict(self, d):
        self.gidx, self.task, self.prio, self.logits = (d[k].to(self.device) for k in ("gidx", "task", "prio", "logits"))


def _unique_first(key: torch.Tensor):
    uniq, inv = torch.unique(key, return_inverse=True)
    first = torch.full((len(uniq),), len(key), dtype=torch.long, device=key.device)
    first.scatter_reduce_(0, inv, torch.arange(len(key), device=key.device), reduce="amin")
    return uniq, first


class Learner:
    """Plain SGD fine-tuning (the forgetting baseline)."""

    uses_buffer = False

    def __init__(self, cfg: dict, sc: Scenario, device):
        self.cfg, self.sc, self.device = cfg, sc, device
        lc = cfg.get("learner", {})
        self.lc = lc
        self.buffer = HashBuffer(int(lc.get("buffer_size", 500)), sc.seed, sc.n_outputs, device) if self.uses_buffer else None

    # ------------------------------------------------------------------ hooks
    def begin_task(self, model, t: int):
        pass

    def end_task(self, model, t: int):
        pass

    def after_step(self, model, t: int, gidx: torch.Tensor, logits: torch.Tensor):
        if self.buffer is not None:
            self.buffer.add(gidx, t, logits)

    # ------------------------------------------------------------------ objective
    def _new_terms(self, model, t, epoch, pos):
        tk = self.sc.tasks[t]
        gidx = tk.train_idx[pos]
        x, y = self.sc.train_batch(t, gidx, epoch)
        return x, y, gidx

    def _mem_inputs(self, sel: torch.Tensor, t: int, epoch: int):
        g, tt = self.buffer.gidx[sel], self.buffer.task[sel]
        xs, ys = [], []
        order = []
        for task in tt.unique().tolist():
            rows = (tt == task).nonzero().squeeze(1)
            xs.append(self.sc.inputs(self.sc.xtr[g[rows]], task, g[rows], epoch=10_000 + t, train=True))
            ys.append(self.sc.target(self.sc.ytr[g[rows]]))
            order.append(rows)
        order = torch.cat(order)
        inv = torch.empty_like(order); inv[order] = torch.arange(len(order), device=order.device)
        key = tt * len(self.sc.xtr) + g                       # memory key: (task, global index)
        return torch.cat(xs)[inv], torch.cat(ys)[inv], key, tt

    def mask_for(self, t: int, sample_task: torch.Tensor):
        return self.sc.logit_mask(t, sample_task, n=len(sample_task))

    def terms(self, model, t: int, epoch: int, pos: torch.Tensor, step_key: int) -> tuple:
        x, y, gidx = self._new_terms(model, t, epoch, pos)
        st = torch.full((len(y),), t, device=self.device)
        logits = model(x)
        lv = F.cross_entropy(masked_logits(logits, self.mask_for(t, st)), y, reduction="none") / len(y)
        return Terms(lv, ["new"], [pos], [slice(0, len(y))], n_new=len(y)), logits[: len(y)], gidx

    def project(self, model, t: int, grad: torch.Tensor, step_key: int) -> Optional[Projection]:
        return None

    # ------------------------------------------------------------------ state
    def state_dict(self):
        return {"buffer": self.buffer.state_dict() if self.buffer is not None else None}

    def load_state_dict(self, d):
        if self.buffer is not None and d.get("buffer") is not None:
            self.buffer.load_state_dict(d["buffer"])


class ER(Learner):
    uses_buffer = True

    def terms(self, model, t, epoch, pos, step_key):
        x, y, gidx = self._new_terms(model, t, epoch, pos)
        mb = int(self.lc.get("minibatch", len(y)))
        sel = self.buffer.sample(mb, t, step_key) if t > 0 else None
        st = torch.full((len(y),), t, device=self.device)
        if sel is None:
            logits = model(x)
            lv = F.cross_entropy(masked_logits(logits, self.mask_for(t, st)), y, reduction="none") / len(y)
            return Terms(lv, ["new"], [pos], [slice(0, len(y))], n_new=len(y)), logits, gidx
        xm, ym, gm, tm = self._mem_inputs(sel, t, epoch)
        logits = model(torch.cat([x, xm]))
        mask = self.mask_for(t, torch.cat([st, tm]))
        ce = F.cross_entropy(masked_logits(logits, mask), torch.cat([y, ym]), reduction="none")
        n = len(y)
        lv = torch.cat([ce[:n] / n, ce[n:] / len(ym)])
        return Terms(lv, ["new", "mem"], [pos, gm], [slice(0, n), slice(n, n + len(ym))], n_new=n), logits[:n], gidx


class DERpp(Learner):
    uses_buffer = True

    def terms(self, model, t, epoch, pos, step_key):
        x, y, gidx = self._new_terms(model, t, epoch, pos)
        n = len(y)
        st = torch.full((n,), t, device=self.device)
        if t == 0 or len(self.buffer) == 0:
            logits = model(x)
            lv = F.cross_entropy(masked_logits(logits, self.mask_for(t, st)), y, reduction="none") / n
            return Terms(lv, ["new"], [pos], [slice(0, n)], n_new=n), logits, gidx
        mb = int(self.lc.get("minibatch", n))
        a, b = float(self.lc.get("alpha", 0.1)), float(self.lc.get("beta", 0.5))
        s1, s2 = self.buffer.sample(mb, t, 2 * step_key), self.buffer.sample(mb, t, 2 * step_key + 1)
        x1, _, g1, _ = self._mem_inputs(s1, t, epoch)
        x2, y2, g2, t2 = self._mem_inputs(s2, t, epoch)
        logits = model(torch.cat([x, x1, x2]))
        ce_new = F.cross_entropy(masked_logits(logits[:n], self.mask_for(t, st)), y, reduction="none") / n
        mse = ((logits[n:n + mb] - self.buffer.logits[s1]) ** 2).mean(1) * a / mb
        ce2 = F.cross_entropy(masked_logits(logits[n + mb:], self.mask_for(t, t2)), y2, reduction="none") * b / mb
        lv = torch.cat([ce_new, mse, ce2])
        return (Terms(lv, ["new", "mem", "mem"], [pos, g1, g2], [slice(0, n), slice(n, n + mb), slice(n + mb, n + 2 * mb)], n_new=n),
                logits[:n], gidx)


class EWC(Learner):
    """Online EWC (Schwarz et al. 2018). The penalty is one extra term of kind "reg"."""

    def __init__(self, cfg, sc, device):
        super().__init__(cfg, sc, device)
        self.fisher: Optional[torch.Tensor] = None
        self.anchor: Optional[torch.Tensor] = None

    def terms(self, model, t, epoch, pos, step_key):
        out, logits, gidx = super().terms(model, t, epoch, pos, step_key)
        if self.fisher is None:
            return out, logits, gidx
        lam = float(self.lc.get("lambda", 100.0))
        theta = flat(params_of(model))
        pen = 0.5 * lam * (self.fisher * (theta - self.anchor) ** 2).sum()
        vals = torch.cat([out.values, pen.view(1)])
        m = len(out.values)
        return Terms(vals, out.kinds + ["reg"], out.ids + [torch.tensor([t], device=self.device)],
                     out.slices + [slice(m, m + 1)], n_new=out.n_new), logits, gidx

    def end_task(self, model, t):
        n = int(self.lc.get("fisher_samples", 200))
        tk = self.sc.tasks[t]
        gidx = tk.train_idx[:n]
        was = model.training
        model.eval()
        fis = torch.zeros_like(flat(params_of(model)).detach())
        for i in range(len(gidx)):
            x = self.sc.inputs(self.sc.xtr[gidx[i:i + 1]], t)
            y = self.sc.target(self.sc.ytr[gidx[i:i + 1]])
            st = torch.full((1,), t, device=self.device)
            loss = F.cross_entropy(masked_logits(model(x), self.mask_for(t, st)), y)
            g = torch.autograd.grad(loss, params_of(model))
            fis += flat(g) ** 2
        fis /= len(gidx)
        model.train(was)
        gamma = float(self.lc.get("gamma", 1.0))
        self.fisher = fis if self.fisher is None else gamma * self.fisher + fis
        self.anchor = flat(params_of(model)).detach().clone()

    def state_dict(self):
        d = super().state_dict(); d.update(fisher=self.fisher, anchor=self.anchor); return d

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.fisher, self.anchor = d.get("fisher"), d.get("anchor")


class AGEM(Learner):
    uses_buffer = True

    def project(self, model, t, grad, step_key):
        if t == 0 or len(self.buffer) == 0:
            return None
        sel = self.buffer.sample(int(self.lc.get("ref_batch", 64)), t, step_key)
        xm, ym, _, tm = self._mem_inputs(sel, t, 0)
        loss = F.cross_entropy(masked_logits(model(xm), self.mask_for(t, tm)), ym)
        g_ref = flat(torch.autograd.grad(loss, params_of(model))).detach()
        dot = float(grad @ g_ref)
        n2 = float(g_ref @ g_ref) + 1e-12
        return Projection(g_ref, dot < 0, n2)


LEARNERS = {"finetune": Learner, "er": ER, "derpp": DERpp, "ewc": EWC, "agem": AGEM}


def build_learner(cfg, sc, device) -> Learner:
    return LEARNERS[cfg.get("learner", {}).get("name", "finetune")](cfg, sc, device)
