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
        self.base_lr, self.base_epochs = self.lr, self.epochs
        self.first_lr = float(tc.get("first_task_lr", self.lr))           # task 0 may be trained longer / differently
        self.first_epochs = int(tc.get("first_task_epochs", self.epochs))
        self.optim = tc.get("optimizer", "sgd")                    # sgd | adam
        self.base_optim = self.optim
        self.first_optim = tc.get("first_task_optimizer", self.optim)
        if self.optim == "sgd" and float(tc.get("momentum", 0.0)) != 0.0:
            raise ValueError("SGD with momentum: use optimizer=adam-style adjoint (not implemented for SGD-momentum)")
        self.betas = tuple(tc.get("betas", (0.9, 0.999)))
        self.eps = float(tc.get("eps", 1e-8))
        self.adam: Optional[dict] = None                              # {m, v, k}
        lc = cfg.get("ledger", {})
        self.rule = lc.get("rule", "adaptive")
        self.tol = float(lc.get("tol", 1e-3))
        self.max_intervals = int(lc.get("max_intervals", 8))
        self.also_euler = bool(lc.get("also_euler", True))
        self.stats_order = lc.get("stats_order", "shapley")          # shapley | stats_first
        self.last_train_time = 0.0
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
        d = {"model": self.model.state_dict(), "learner": self.learner.state_dict(), "adam": self.adam}
        if extra:
            d.update(extra)
        atomic_torch_save(d, self.p("snapshots", f"{name}.pt"))

    def load_snapshot(self, name: str):
        d = torch.load(self.p("snapshots", f"{name}.pt"), map_location=self.device, weights_only=False)
        self.model.load_state_dict(d["model"])
        self.learner.load_state_dict(d["learner"])
        self.adam = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d["adam"].items()} if d.get("adam") else None
        return d

    # ------------------------------------------------------------------ resume state
    def save_state(self, pos: dict, acc: Optional[LedgerAccumulator], extra: dict):
        atomic_torch_save({"pos": pos, "model": self.model.state_dict(), "learner": self.learner.state_dict(),
                           "ledger": acc.state_dict() if acc is not None else None, "rng": rng_state(),
                           "adam": self.adam, **extra},
                          self.p("state", "latest.pt"))

    def load_state(self) -> Optional[dict]:
        f = self.p("state", "latest.pt")
        if not os.path.exists(f):
            return None
        d = torch.load(f, map_location=self.device, weights_only=False)
        self.model.load_state_dict(d["model"])
        self.learner.load_state_dict(d["learner"])
        set_rng_state(d["rng"])
        self.adam = d.get("adam")
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
                   track: Optional[bool] = None, save: bool = True, order_rep: int = 0) -> Optional[LedgerAccumulator]:
        """Train task t. ``keep`` (bool over task positions) removes samples; ``frozen`` (bool over
        parameters) freezes parameters. Returns the ledger accumulator (if tracked)."""
        track = self.track if track is None else track
        track = track and t > 0
        self.lr = self.first_lr if t == 0 else self.base_lr
        self.epochs = self.first_epochs if t == 0 else self.base_epochs
        self.optim = self.first_optim if t == 0 else self.base_optim
        sc, model, lr, bs = self.sc, self.model, self.lr, self.bs
        tk = sc.tasks[t]
        N = len(tk.train_idx)
        fresh = start_epoch == 0 and start_batch == 0 and resume_acc is None
        if fresh:
            self.learner.begin_task(model, t)
            if save:
                self.snapshot(f"task{t}_start")
        steps_per_epoch = [math.ceil(int((keep[sc.epoch_order(t, e + 1000 * order_rep)] if keep is not None else torch.ones(N, dtype=torch.bool, device=self.device)).sum()) / bs) for e in range(self.epochs)]
        total_steps = sum(steps_per_epoch)
        cp_steps = sorted(set(int(round(j * total_steps / self.n_cp)) for j in range(self.n_cp))) if save and track else []
        acc, probe = None, None
        if track and self.optim == "adam":
            if self.stats or self.learner.buffer is not None or self.learner.__class__.__name__ != "Learner":
                raise ValueError("the Adam ledger is implemented for fine-tuning without running statistics")
            if save and self.ckpt_every:
                self.ckpt_every = 0                                   # two-pass ledger: no mid-task resume
            self._adam_task_start = self._adam_state()["m"].clone()
            self._adam_tau = 0
            self._adam_log = []
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
            order = sc.epoch_order(t, epoch + 1000 * order_rep)     # order_rep > 0: alternative visiting order
            if keep is not None:
                order = order[keep[order]]
            b0 = start_batch if epoch == start_epoch else 0
            for bi in range(b0, math.ceil(len(order) / bs)):
                if step in cp_steps:
                    atomic_torch_save({"theta": flat_params(model).cpu(), "buffers": buffers_state(model), "step": step},
                                      self.p("snapshots", f"task{t}_cp{cp_steps.index(step)}.pt"))
                pos = order[bi * bs:(bi + 1) * bs]
                step_key = epoch * 100_000 + bi
                if track and self.optim == "adam":
                    L_cur, G_cur = self._tracked_step_adam(t, epoch, pos, step_key, probe, acc, L_cur, G_cur, frozen, step)
                elif track:
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
            if track:
                l_, c_ = self._sample_stats(t)
                acc.sample_loss.append(l_); acc.sample_correct.append(c_)
            if save and self.log:
                msg = f"task {t} epoch {epoch} step {step}/{total_steps} {time.time() - t0:.0f}s"
                if track:
                    err = self._completeness(acc)
                    msg += (f" | dL={acc.true_dL.sum().item():+.4f} ledger={(acc.path_g + acc.stats).sum().item():+.4f}"
                            f" err={err:.3%} err_tv={self._completeness_tv(acc):.3%} evals/step={acc.n_evals / max(1, acc.steps):.2f}")
                print(msg, flush=True)
        if track and self.optim == "adam":
            self._adam_reverse_pass(t, probe, acc)
        self.last_train_time = time.time() - t0
        self.last_train_steps = step
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
        if self.optim == "adam":
            D, m = self._adam_update(g, frozen)
            set_flat_params(model, flat_params(model) - D * m)
        else:
            set_flat_params(model, flat_params(model) - self.lr * g)
        self.learner.after_step(model, t, gidx, logits.detach()[: len(gidx)])

    # ------------------------------------------------------------------ Adam
    def _adam_state(self):
        if self.adam is None:
            P = flat_params(self.model).numel()
            z = torch.zeros(P, device=self.device)
            self.adam = {"m": z.clone(), "v": z.clone(), "k": 0}
        return self.adam

    def _adam_update(self, g: torch.Tensor, frozen=None):
        """Adam moment update; returns (D, m) with Delta = -D * m, D the elementwise step size."""
        st = self._adam_state()
        b1, b2 = self.betas
        st["k"] += 1
        st["m"].mul_(b1).add_(g, alpha=1 - b1)
        st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
        k = st["k"]
        D = self.lr / ((1 - b1 ** k) * ((st["v"] / (1 - b2 ** k)).sqrt() + self.eps))
        if frozen is not None:
            D = D * (~frozen)
        return D, st["m"]

    def _tracked_step_adam(self, t, epoch, pos, step_key, probe, acc, L_cur, G_cur, frozen, step):
        """Forward pass of the two-pass Adam ledger.

        With the elementwise step size D_t fixed, Delta_t = -D_t * m_t is linear in the gradients of
        all steps of the task, m_t = b1^tau m_0 + sum_s (1-b1) b1^(tau-s) g_s. The parameter-level
        ledger is computed here exactly; per-sample credit needs u_s = sum_{t>=s} (1-b1) b1^(t-s) D_t gbar_t,
        which depends on the future and is obtained in a reverse pass (``_adam_reverse_pass``)."""
        from .ledger import path_average_gradient
        model = self.model
        terms, logits, gidx = self.learner.terms(model, t, epoch, pos, step_key)
        g = flat(torch.autograd.grad(terms.values.sum(), params_of(model)))
        if frozen is not None:
            g = g * (~frozen)
        theta = flat_params(model)
        D, m = self._adam_update(g, frozen)
        delta = -D * m
        L_end, G_end = probe_grads(model, probe, theta + delta)
        dL = L_end - L_cur
        gbar, n_ev = path_average_gradient(model, probe, theta, delta, G_cur, G_end, dL, self.rule, self.tol, self.max_intervals)
        acc.n_evals += n_ev + 1
        acc.path_var += dL.abs().double()
        pg = gbar * delta.unsqueeze(0)
        acc.param += pg.sum(0).double()
        acc.unit.index_add_(0, acc.unit_of, pg.T.double())
        acc.layer.index_add_(0, acc.layer_of, pg.T.double())
        acc.path_g += pg.sum(1).double()
        lp = -g * delta
        acc.learn_param += lp.double()
        acc.learn_unit.index_add_(0, acc.unit_of, lp.double())
        self._adam_tau += 1
        w = D.unsqueeze(0) * gbar                                            # [G, P]
        acc.reg += (-(self.betas[0] ** self._adam_tau) * (w @ self._adam_task_start)).double()   # momentum carried over
        # store what the reverse pass needs on disk (gbar_t is recomputed there: same quadrature)
        f = self.p("adam_tmp", f"t{t}_s{len(self._adam_log)}.pt")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        torch.save({"theta": theta.cpu(), "D": D.cpu(), "delta": delta.cpu(), "pos": pos.cpu(),
                    "epoch": epoch, "step_key": step_key}, f)
        self._adam_log.append(f)
        set_flat_params(model, theta + delta)
        acc.true_dL = (L_end.double() - acc.L0)
        acc.step_err.append(float(acc.true_dL.sum() - (acc.path_g + acc.stats).sum()))
        self.learner.after_step(model, t, gidx, logits.detach()[: len(gidx)])
        acc.steps += 1
        return L_end, G_end

    def _adam_reverse_pass(self, t, probe, acc):
        """u_s = (1-b1) D_s gbar_s + b1 u_{s+1};  per-term credit C_i = -grad l_i(theta_s) . u_s (exact)."""
        from .ledger import path_average_gradient
        model = self.model
        b1 = self.betas[0]
        end_params = flat_params(model).clone()
        u = None
        for f in reversed(self._adam_log):
            d = torch.load(f, map_location=self.device, weights_only=False)
            L0_, G0_ = probe_grads(model, probe, d["theta"])
            L1_, G1_ = probe_grads(model, probe, d["theta"] + d["delta"])
            gbar, _ = path_average_gradient(model, probe, d["theta"], d["delta"], G0_, G1_, L1_ - L0_, self.rule,
                                            self.tol, self.max_intervals)
            w = d["D"].unsqueeze(0) * gbar
            u = (1 - b1) * w if u is None else (1 - b1) * w + b1 * u
            set_flat_params(model, d["theta"])
            model.train()
            terms, _, _ = self.learner.terms(model, t, d["epoch"], d["pos"].to(self.device), d["step_key"])
            uu = torch.ones(len(terms.values), device=self.device, requires_grad=True)
            grads = torch.autograd.grad((terms.values * uu).sum(), params_of(model), create_graph=True)
            h = u @ flat(grads)
            Dm = torch.autograd.grad(h, uu, grad_outputs=torch.eye(len(h), device=self.device), is_grads_batched=True)[0].T
            for kind, ids, sl in zip(terms.kinds, terms.ids, terms.slices):
                if kind == "new":
                    acc.data.index_add_(0, ids, (-Dm[sl]).double())
            os.remove(f)
        set_flat_params(model, end_params)
        self._adam_log = []

    def _tracked_step(self, t, epoch, pos, step_key, probe, acc, L_cur, G_cur, frozen):
        model = self.model
        buf_before = buffers_state(model) if self.stats else None
        terms, logits, gidx = self.learner.terms(model, t, epoch, pos, step_key)   # train-mode forward (updates BN stats)
        shap = None
        if self.stats:
            L_s, G_s = probe_grads(model, probe)                                    # (theta_t, s_{t+1})
            if self.stats_order == "shapley":
                shap = dict(buffers_old=buf_before, G_old=G_cur, L_old=L_cur)
            else:                                                                   # statistics-first order
                acc.stats += (L_s - L_cur).double()
            L_cur, G_cur = L_s, G_s
        u = torch.ones(len(terms.values), device=self.device, requires_grad=True)
        grads = torch.autograd.grad((terms.values * u).sum(), params_of(model), create_graph=True)
        g_det = flat([g.detach() for g in grads])
        proj = self.learner.project(model, t, g_det, step_key)
        delta, L_end, G_end = ledger_step(model, self.lr, terms, list(grads), u, proj, G_cur, probe, acc, self.rule, frozen,
                                          L_start=L_cur, tol=self.tol, max_intervals=self.max_intervals,
                                          also_euler=self.also_euler, shapley=shap)
        del grads
        set_flat_params(model, flat_params(model) + delta)
        acc.true_dL = (L_end.double() - acc.L0)
        acc.step_err.append(float(acc.true_dL.sum() - (acc.path_g + acc.stats).sum()))   # cumulative residual
        self.learner.after_step(model, t, gidx, logits.detach()[: len(gidx)])
        return L_end, G_end

    @torch.no_grad()
    def _sample_stats(self, t: int):
        """Per-sample training loss and correctness on task t (no augmentation, eval mode)."""
        m, sc = self.model, self.sc
        was = m.training
        m.eval()
        tk = sc.tasks[t]
        x = sc.inputs(sc.xtr[tk.train_idx], t)
        y = sc.target(sc.ytr[tk.train_idx])
        st = torch.full((len(y),), t, device=self.device)
        out = torch.cat([m(x[i:i + 1000]) for i in range(0, len(x), 1000)])
        ml = masked_logits(out, sc.logit_mask(t, st, n=len(y)))
        loss = F.cross_entropy(ml, y, reduction="none")
        m.train(was)
        return loss.float().cpu(), (ml.argmax(1) == y).cpu()
