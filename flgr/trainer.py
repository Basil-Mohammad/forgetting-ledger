"""Sequential training with the ledger, full checkpoint/resume, snapshots and evaluation.

Directory layout of a run (``run_dir``)::

    config.json  env.json  git.txt  metrics.jsonl
    state/latest.pt                    resumable state (atomic, every ``ckpt_every`` steps)
    snapshots/task{k}_start.pt         model + learner state at the start of task k
    snapshots/task{k}_end.pt           ... at the end of task k
    snapshots/task{k}_cp{j}.pt         intra-task parameter checkpoints (TracIn-CP)
    ledger/task{k}.pt                  ledger tensors of task k (k >= 1)
    eval/task{k}.json                  accuracy matrix row, per-group test metrics
"""
from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .data import Scenario
from .learners import Learner, masked_logits
from .ledger import LedgerAccumulator, ProbeSet, ledger_step, probe_grads, probe_losses
from .utils import (append_jsonl, atomic_json, atomic_torch_save, buffers_state, flat, flat_params,
                    has_running_stats, load_buffers, params_of, rng_state, set_flat_params, set_rng_state)


class Trainer:
    def __init__(self, cfg: dict, run_dir: str, sc: Scenario, model: torch.nn.Module, learner: Learner,
                 device, track: bool = True, log: bool = True):
        self.cfg, self.run_dir, self.sc, self.model, self.learner, self.device = cfg, run_dir, sc, model, learner, device
        self.track, self.log = track, log
        tc = cfg["train"]
        self.lr, self.bs, self.epochs = float(tc["lr"]), int(tc["batch_size"]), int(tc["epochs"])
        if float(tc.get("momentum", 0.0)) != 0.0:
            raise ValueError("exact per-sample ledger requires plain SGD (momentum = 0)")
        lc = cfg.get("ledger", {})
        self.rule = lc.get("rule", "adaptive")
        self.tol = float(lc.get("tol", 1e-3))
        self.max_intervals = int(lc.get("max_intervals", 8))
        self.group_by = lc.get("group_by", "class")
        self.ckpt_every = int(cfg.get("ckpt_every", 200))
        self.n_cp = int(lc.get("tracin_checkpoints", 10))
        self.early_fracs = [float(f) for f in lc.get("early_fractions", [0.1, 0.2, 0.5])]
        self.store_param_groups = bool(lc.get("store_param_groups", False))
        self.stats = has_running_stats(model)

    # ------------------------------------------------------------------ paths
    def p(self, *parts):
        return os.path.join(self.run_dir, *parts)

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, upto_task: int) -> dict:
        """Test accuracy per task and per group for tasks <= upto_task (eval mode, current masks)."""
        m = self.model
        was = m.training
        m.eval()
        out = {"task_acc": [], "groups": []}
        for t in range(upto_task + 1):
            x, y, ylab = self.sc.test_batch(t)
            st = torch.full((len(y),), t, device=self.device)
            logits = torch.cat([m(x[i:i + 1000]) for i in range(0, len(x), 1000)])
            ml = masked_logits(logits, self.sc.logit_mask(upto_task, st, n=len(y)))
            pred = ml.argmax(1)
            loss = F.cross_entropy(ml, y, reduction="none")
            out["task_acc"].append((pred == y).float().mean().item())
            for c in self.sc.tasks[t].classes:
                sel = ylab == c
                out["groups"].append(dict(task=t, label=c, acc=(pred[sel] == y[sel]).float().mean().item(),
                                          loss=loss[sel].mean().item()))
        m.train(was)
        return out

    # ------------------------------------------------------------------ snapshots
    def snapshot(self, name: str, extra: Optional[dict] = None):
        d = {"model": self.model.state_dict(), "learner": self.learner.state_dict()}
        if extra:
            d.update(extra)
        atomic_torch_save(d, self.p("snapshots", f"{name}.pt"))

    def load_snapshot(self, name: str):
        d = torch.load(self.p("snapshots", f"{name}.pt"), map_location=self.device, weights_only=False)
        self.model.load_state_dict(d["model"])
        self.learner.load_state_dict(d["learner"])
        return d

    # ------------------------------------------------------------------ resume state
    def save_state(self, pos: dict, acc: Optional[LedgerAccumulator], extra: dict):
        atomic_torch_save({"pos": pos, "model": self.model.state_dict(), "learner": self.learner.state_dict(),
                           "ledger": acc.state_dict() if acc is not None else None, "rng": rng_state(), **extra},
                          self.p("state", "latest.pt"))

    def load_state(self) -> Optional[dict]:
        f = self.p("state", "latest.pt")
        if not os.path.exists(f):
            return None
        d = torch.load(f, map_location=self.device, weights_only=False)
        self.model.load_state_dict(d["model"])
        self.learner.load_state_dict(d["learner"])
        set_rng_state(d["rng"])
        return d

    # ------------------------------------------------------------------ main loop
    def run(self, tasks: Optional[List[int]] = None):
        tasks = list(range(self.sc.n_tasks)) if tasks is None else tasks
        st = self.load_state()
        start_task, start_epoch, start_batch = 0, 0, 0
        resume_acc = None
        if st is not None:
            pos = st["pos"]
            start_task, start_epoch, start_batch = pos["task"], pos["epoch"], pos["batch"]
            resume_acc = st.get("ledger")
            if self.log:
                print(f"[resume] task {start_task} epoch {start_epoch} batch {start_batch}", flush=True)
        for t in tasks:
            if t < start_task:
                continue
            self.train_task(t, start_epoch if t == start_task else 0, start_batch if t == start_task else 0,
                            resume_acc if t == start_task else None)
            resume_acc = None
        if self.log:
            print("[done]", flush=True)

    def train_task(self, t: int, start_epoch: int = 0, start_batch: int = 0, resume_acc=None,
                   keep: Optional[torch.Tensor] = None, frozen: Optional[torch.Tensor] = None,
                   track: Optional[bool] = None, save: bool = True) -> Optional[LedgerAccumulator]:
        """Train task t. ``keep`` (bool over task positions) removes samples; ``frozen`` (bool over
        parameters) freezes parameters. Returns the ledger accumulator (if tracked)."""
        track = self.track if track is None else track
        track = track and t > 0
        sc, model, lr, bs = self.sc, self.model, self.lr, self.bs
        tk = sc.tasks[t]
        N = len(tk.train_idx)
        fresh = start_epoch == 0 and start_batch == 0 and resume_acc is None
        if fresh:
            self.learner.begin_task(model, t)
            if save:
                self.snapshot(f"task{t}_start")
        steps_per_epoch = [math.ceil(int((keep[sc.epoch_order(t, e)] if keep is not None else torch.ones(N, dtype=torch.bool, device=self.device)).sum()) / bs) for e in range(self.epochs)]
        total_steps = sum(steps_per_epoch)
        cp_steps = sorted(set(int(round(j * total_steps / self.n_cp)) for j in range(self.n_cp))) if save and track else []
        acc, probe = None, None
        if track:
            probe = ProbeSet(sc, t, t, self.group_by, self.device)
            n_mem = ((t + 1) * len(sc.xtr)) if self.learner.buffer is not None else 0   # buffer may hold current-task samples
            acc = LedgerAccumulator(model, N, n_mem, probe.G, self.device, self.store_param_groups)
            if resume_acc is not None:
                acc.load_state_dict(resume_acc)
            L_cur, G_cur = probe_grads(model, probe)
            if acc.L0 is None:
                acc.L0 = L_cur.double()
        model.train()
        step = sum(steps_per_epoch[:start_epoch]) + start_batch
        t0 = time.time()
        for epoch in range(start_epoch, self.epochs):
            order = sc.epoch_order(t, epoch)
            if keep is not None:
                order = order[keep[order]]
            b0 = start_batch if epoch == start_epoch else 0
            for bi in range(b0, math.ceil(len(order) / bs)):
                if step in cp_steps:
                    atomic_torch_save({"theta": flat_params(model).cpu(), "buffers": buffers_state(model), "step": step},
                                      self.p("snapshots", f"task{t}_cp{cp_steps.index(step)}.pt"))
                pos = order[bi * bs:(bi + 1) * bs]
                step_key = epoch * 100_000 + bi
                if track:
                    L_cur, G_cur = self._tracked_step(t, epoch, pos, step_key, probe, acc, L_cur, G_cur, frozen)
                else:
                    self._plain_step(t, epoch, pos, step_key, frozen)
                step += 1
                if track:
                    for f in self.early_fracs:
                        key = f"{f:g}"
                        if key not in acc.early and step >= f * total_steps:
                            acc.early[key] = acc.data.clone()
                if save and self.ckpt_every and step % self.ckpt_every == 0:
                    nb = math.ceil(len(order) / bs)
                    nxt = dict(task=t, epoch=epoch, batch=bi + 1) if bi + 1 < nb else dict(task=t, epoch=epoch + 1, batch=0)
                    self.save_state(nxt, acc, {})
            if save and self.log:
                msg = f"task {t} epoch {epoch} step {step}/{total_steps} {time.time() - t0:.0f}s"
                if track:
                    err = self._completeness(acc)
                    msg += (f" | dL={acc.true_dL.sum().item():+.4f} ledger={(acc.path_g + acc.stats).sum().item():+.4f}"
                            f" err={err:.3%} err_tv={self._completeness_tv(acc):.3%} evals/step={acc.n_evals / max(1, acc.steps):.2f}")
                print(msg, flush=True)
        self.learner.end_task(model, t)
        if save:
            ev = self.evaluate(t)
            atomic_json(ev, self.p("eval", f"task{t}.json"))
            append_jsonl({"task": t, "task_acc": ev["task_acc"], "time_s": time.time() - t0,
                          "completeness": self._completeness(acc) if acc is not None else None,
                          "completeness_tv": self._completeness_tv(acc) if acc is not None else None}, self.p("metrics.jsonl"))
            if acc is not None:
                d = acc.state_dict()
                d["unit_meta"] = acc.unit_meta
                d["groups"] = [dict(name=g["name"], task=g["task"], label=g["label"]) for g in probe.groups]
                d["cp_steps"] = cp_steps
                d["total_steps"] = total_steps
                atomic_torch_save(d, self.p("ledger", f"task{t}.pt"))
            self.snapshot(f"task{t}_end")
            self.save_state(dict(task=t + 1, epoch=0, batch=0), None, {})
        return acc

    @staticmethod
    def _completeness(acc: LedgerAccumulator) -> float:
        """Residual of the ledger relative to the realised change, sum_g |dL_g|."""
        pred = (acc.path_g + acc.stats)
        return float((pred - acc.true_dL).abs().sum() / acc.true_dL.abs().sum().clamp_min(1e-12))

    @staticmethod
    def _completeness_tv(acc: LedgerAccumulator) -> float:
        """Residual relative to the total variation of the loss path (robust when dL ~ 0)."""
        pred = (acc.path_g + acc.stats)
        return float((pred - acc.true_dL).abs().sum() / acc.path_var.sum().clamp_min(1e-12))

    # ------------------------------------------------------------------ steps
    def _plain_step(self, t, epoch, pos, step_key, frozen):
        model = self.model
        terms, logits, gidx = self.learner.terms(model, t, epoch, pos, step_key)
        grads = torch.autograd.grad(terms.values.sum(), params_of(model))
        g = flat(grads)
        proj = self.learner.project(model, t, g, step_key)
        if proj is not None and proj.active:
            g = g - (float(g @ proj.g_ref) / proj.ref_norm2) * proj.g_ref
        if frozen is not None:
            g = g * (~frozen)
        set_flat_params(model, flat_params(model) - self.lr * g)
        self.learner.after_step(model, t, gidx, logits.detach()[: len(gidx)])

    def _tracked_step(self, t, epoch, pos, step_key, probe, acc, L_cur, G_cur, frozen):
        model = self.model
        terms, logits, gidx = self.learner.terms(model, t, epoch, pos, step_key)   # train-mode forward (updates BN stats)
        if self.stats:                                                              # exact running-statistics drift
            L_s, G_s = probe_grads(model, probe)
            acc.stats += (L_s - L_cur).double()
            L_cur, G_cur = L_s, G_s
        u = torch.ones(len(terms.values), device=self.device, requires_grad=True)
        grads = torch.autograd.grad((terms.values * u).sum(), params_of(model), create_graph=True)
        g_det = flat([g.detach() for g in grads])
        proj = self.learner.project(model, t, g_det, step_key)
        delta, L_end, G_end = ledger_step(model, self.lr, terms, list(grads), u, proj, G_cur, probe, acc, self.rule, frozen,
                                          L_start=L_cur, tol=self.tol, max_intervals=self.max_intervals)
        del grads
        set_flat_params(model, flat_params(model) + delta)
        acc.true_dL = (L_end.double() - acc.L0)
        acc.step_err.append(float(acc.true_dL.sum() - (acc.path_g + acc.stats).sum()))   # cumulative residual
        self.learner.after_step(model, t, gidx, logits.detach()[: len(gidx)])
        return L_end, G_end
