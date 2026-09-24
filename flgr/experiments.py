"""Experiment commands (see EXPERIMENTS.md). Each command is idempotent and resumable."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import run_name
from .data import Scenario
from .ledger import ProbeSet, probe_grads, probe_losses
from .learners import build_learner
from .models import build_model, merge_units_by_module, unit_index
from .scores import (feature_class_similarity, feature_proximity, fisher_diag, grad_dot_scores, loss_scores,
                     margin_scores, trak_harm)
from .trainer import Trainer
from .utils import (atomic_json, atomic_torch_save, env_info, flat_params, git_commit, hash_uniform, params_of,
                    pick_device, seed_everything, set_flat_params)

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- setup

def setup(cfg: dict, run_dir: str, track: bool = True, log: bool = True):
    seed_everything(int(cfg["seed"]))
    device = pick_device(cfg.get("device", "auto"))
    if device.type == "cpu":                     # respect OMP_NUM_THREADS (parallel workers must not oversubscribe)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 0)) or max(1, os.cpu_count() or 1))
    if cfg["benchmark"] == "agnews_dbpedia":
        from .text import TextScenario
        sc = TextScenario(cfg, device)
    else:
        sc = Scenario(cfg, device)
    model = build_model(cfg, sc).to(device)
    learner = build_learner(cfg, sc, device)
    os.makedirs(run_dir, exist_ok=True)
    if not os.path.exists(os.path.join(run_dir, "config.json")):
        atomic_json(cfg, os.path.join(run_dir, "config.json"))
        atomic_json(env_info(), os.path.join(run_dir, "env.json"))
        with open(os.path.join(run_dir, "git.txt"), "w") as f:
            f.write(git_commit(os.path.join(HERE, "..")) + "\n")
    return Trainer(cfg, run_dir, sc, model, learner, device, track=track, log=log)


def _tasks(cfg, sc) -> List[int]:
    out = []
    for t in cfg["interventions"]["tasks"]:
        t = sc.n_tasks + t if t < 0 else t
        if 1 <= t < sc.n_tasks and t not in out:
            out.append(t)
    return out


def _load_ledger(tr: Trainer, t: int) -> dict:
    return torch.load(tr.p("ledger", f"task{t}.pt"), map_location=tr.device, weights_only=False)


def _results(tr: Trainer, name: str) -> dict:
    f = tr.p("interventions", f"{name}.json")
    return json.load(open(f)) if os.path.exists(f) else {}


def _save_results(tr: Trainer, name: str, d: dict):
    atomic_json(d, tr.p("interventions", f"{name}.json"))


def _require_trained(tr: Trainer):
    if not os.path.exists(tr.p("eval", f"task{tr.sc.n_tasks - 1}.json")):
        raise RuntimeError("run `train` first (the tracked run is incomplete)")


# ----------------------------------------------------------------------------- train

def cmd_train(cfg, run_dir):
    tr = setup(cfg, run_dir)
    tr.run()


# ----------------------------------------------------------------------------- scores

def cmd_scores(cfg, run_dir):
    tr = setup(cfg, run_dir, log=False)
    _require_trained(tr)
    sc, model = tr.sc, tr.model
    for t in _tasks(cfg, sc):
        out_f = tr.p("scores", f"task{t}.pt")
        if os.path.exists(out_f):
            continue
        t0 = time.time()
        led = _load_ledger(tr, t)
        probe = ProbeSet(sc, t, t, tr.group_by, tr.device)
        S: Dict[str, torch.Tensor] = {"ledger": led["data"].float()}
        for k, v in led["early"].items():
            if float(v.abs().sum()) > 0:                  # (the two-pass Adam ledger has no early snapshots)
                S[f"ledger_early{int(float(k) * 100)}"] = v.float()
        N, G = S["ledger"].shape
        # --- checkpoint-based gradient scores (TracIn-CP family) and TRAK-style projected influence
        cps = led["cp_steps"]
        total = led["total_steps"]
        bounds = cps + [total]
        k_proj = int(cfg.get("scores", {}).get("trak_dim", 512))
        trak_cps = set(np.linspace(0, len(cps) - 1, 4).round().astype(int).tolist())
        P_ = sum(p.numel() for p in params_of(model))
        gR = torch.Generator(device="cpu").manual_seed(1234 + int(cfg["seed"]))
        R = (torch.randn(P_, k_proj, generator=gR) / np.sqrt(k_proj)).to(tr.device) if k_proj > 0 else None
        per_cp, trak = [], torch.zeros(N, G, device=tr.device)
        for j in range(len(cps)):
            d = torch.load(tr.p("snapshots", f"task{t}_cp{j}.pt"), map_location=tr.device, weights_only=False)
            set_flat_params(model, d["theta"].to(tr.device))
            for k_, v in d["buffers"].items():
                dict(model.named_buffers())[k_].copy_(v)
            _, Gm = probe_grads(model, probe)
            use_R = R if j in trak_cps else None
            dots, cos, norms, Psi = grad_dot_scores(model, sc, t, Gm, want_cos=(j == 0), R=use_R)
            per_cp.append(-tr.lr * dots * (bounds[j + 1] - bounds[j]) / N)
            if Psi is not None:
                trak += trak_harm(Psi, Gm @ R)
            if j == 0:
                S["static_tracin"] = -dots
                if cos is not None:
                    S["grad_cos"] = -cos
                S["loss"] = loss_scores(model, sc, t).view(-1, 1).expand(-1, G).contiguous()
                S["feature_prox"] = feature_proximity(model, sc, t, probe)
                sim = feature_class_similarity(model, sc, t, probe)
                feats = dict(loss_start=S["loss"][:, 0].clone(), margin_start=margin_scores(model, sc, t),
                             gradnorm_start=norms, prox_start=S["feature_prox"].max(1).values)
        if R is not None:
            S["trak"] = trak / len(trak_cps)
        del R
        if led.get("data_euler") is not None and float(led["data_euler"].abs().sum()) > 0:
            S["ledger_euler"] = led["data_euler"].float()
        pc = torch.stack(per_cp)                                                # [m, N, G]
        S["tracin_cp10"] = pc.sum(0)
        pick3 = sorted(set(np.linspace(0, len(cps) - 1, 3).round().astype(int).tolist()))
        S["tracin_cp3"] = pc[pick3].sum(0) * (len(cps) / len(pick3))
        S["random"] = hash_uniform(int(cfg["seed"]), 77, t, 0, torch.arange(N, device=tr.device)).expand(-1, G).contiguous()
        # --- single-point first-order explanations of the realised change (RQ1)
        tr.load_snapshot(f"task{t}_start"); th0 = flat_params(model); _, G0 = probe_grads(model, probe)
        tr.load_snapshot(f"task{t}_end");   th1 = flat_params(model); _, G1 = probe_grads(model, probe)
        dth = th1 - th0
        single = dict(start=(G0 @ dth).tolist(), end=(G1 @ dth).tolist(), true=led["true_dL"].tolist(),
                      ledger=(led["path_g"] + led["stats"]).tolist(), stats=led["stats"].tolist())
        # --- interference map: new class x old group
        ylab = sc.ytr[sc.tasks[t].train_idx]
        imap = torch.stack([S["ledger"][ylab == c].sum(0) for c in sc.tasks[t].classes])
        # per-sample learning dynamics from the tracked run (anatomy of harmful samples)
        if led.get("sample_correct"):
            C = torch.stack(led["sample_correct"]).float()                     # [E, N]
            first = torch.where(C.any(0), C.argmax(0).float(), torch.full((N,), float(len(C)), device=C.device))
            feats["first_correct_epoch"] = first
            feats["loss_end"] = led["sample_loss"][-1]
            feats["label"] = sc.ytr[sc.tasks[t].train_idx].cpu()
        atomic_torch_save(dict(scores={k: v.cpu() for k, v in S.items()}, single=single, imap=imap.cpu(),
                               feat_sim=sim.cpu(), new_classes=sc.tasks[t].classes, groups=led["groups"],
                               features={k: v.cpu() for k, v in feats.items()},
                               train_idx=sc.tasks[t].train_idx.cpu()), out_f)
        print(f"[scores] task {t}: {len(S)} score types in {time.time() - t0:.0f}s", flush=True)


def _load_scores(tr, t):
    return torch.load(tr.p("scores", f"task{t}.pt"), map_location=tr.device, weights_only=False)


# ----------------------------------------------------------------------------- counterfactual retraining

def retrain(tr: Trainer, t: int, keep: Optional[torch.Tensor] = None, frozen: Optional[torch.Tensor] = None,
            rep: int = 0, zero: Optional[torch.Tensor] = None) -> dict:
    """Retrain task t from its saved start state (untracked) and evaluate. ``rep`` > 0 uses an
    alternative visiting order (variance reduction by averaging over orders)."""
    tr.load_snapshot(f"task{t}_start")
    tr.train_task(t, keep=keep, frozen=frozen, track=False, save=False, order_rep=rep, zero=zero)
    out = measure(tr, t)
    out["time_s"] = tr.last_train_time
    out["steps"] = tr.last_train_steps
    return out


def measure(tr: Trainer, t: int) -> dict:
    ev = tr.evaluate(t)
    probe = ProbeSet(tr.sc, t, t, tr.group_by, tr.device)
    L = probe_losses(tr.model, probe)
    old = [g for g in ev["groups"] if g["task"] < t]
    return dict(task_acc=ev["task_acc"], new_acc=ev["task_acc"][t], old_acc=float(np.mean(ev["task_acc"][:t])),
                group_acc=[g["acc"] for g in old], probe_L=L.tolist(), group_test_loss=[g["loss"] for g in old],
                old_test_loss=float(np.mean([g["loss"] for g in old])))


AVG_KEYS = ("new_acc", "old_acc", "old_test_loss")


def retrain_avg(tr: Trainer, t: int, prev: Optional[dict], n_rep: int, keep=None, frozen=None, zero=None) -> dict:
    """Average of ``n_rep`` retrainings with different visiting orders (rep 0 = the original order).
    ``prev`` (an earlier single result or average) is reused and extended."""
    reps = list(prev.get("reps", [prev])) if prev else []
    for r in range(len(reps), n_rep):
        reps.append(retrain(tr, t, keep=keep, frozen=frozen, rep=r, zero=zero))
    out = dict(reps=reps)
    for k in ("task_acc", "group_acc", "probe_L", "group_test_loss"):
        if all(k in x for x in reps):
            out[k] = np.mean([x[k] for x in reps], 0).tolist()
    for k in AVG_KEYS + ("time_s", "steps"):
        if all(k in x for x in reps):
            out[k] = float(np.mean([x[k] for x in reps]))
    return out


def _needs(R: dict, key: str, n_rep: int) -> bool:
    return key not in R or len(R[key].get("reps", [R[key]])) < n_rep


def _ref_acc(tr: Trainer, t: int) -> List[float]:
    """Accuracy of each old task right after it was learned (for forgetting)."""
    return [json.load(open(tr.p("eval", f"task{j}.json")))["task_acc"][j] for j in range(t)]


def _topk_keep(score: torch.Tensor, frac: float) -> torch.Tensor:
    k = int(round(frac * len(score)))
    keep = torch.ones(len(score), dtype=torch.bool, device=score.device)
    if k > 0:
        keep[torch.argsort(score, descending=True)[:k]] = False
    return keep


REMOVAL_METHODS = ["ledger", "ledger_euler", "trak", "tracin_cp3", "tracin_cp10", "static_tracin", "grad_cos", "loss",
                   "feature_prox", "ledger_early10", "ledger_early20", "ledger_early50", "random"]


def cmd_removal(cfg, run_dir):
    tr = setup(cfg, run_dir, log=False)
    _require_trained(tr)
    for t in _tasks(cfg, tr.sc):
        S = _load_scores(tr, t)["scores"]
        R = _results(tr, f"removal_task{t}")
        R.setdefault("ref_acc", _ref_acc(tr, t))
        n_rep = int(cfg["interventions"].get("reps", 3))
        if _needs(R, "none", n_rep):
            R["none"] = retrain_avg(tr, t, R.get("none"), n_rep); _save_results(tr, f"removal_task{t}", R)
        wanted = cfg["interventions"].get("removal_methods") or REMOVAL_METHODS
        methods = [m for m in wanted if m in S] + ["ledger_helpful"]
        for m in methods:
            s = -S["ledger"].sum(1) if m == "ledger_helpful" else S[m].sum(1)
            for q in cfg["interventions"]["removal_fracs"]:
                key = f"{m}@{q}"
                if not _needs(R, key, n_rep):
                    continue
                t0 = time.time()
                R[key] = retrain_avg(tr, t, R.get(key), n_rep, keep=_topk_keep(s.to(tr.device), q))
                _save_results(tr, f"removal_task{t}", R)
                print(f"[removal] task {t} {key}: old {R[key]['old_acc']:.4f} new {R[key]['new_acc']:.4f} ({time.time() - t0:.0f}s)", flush=True)


def cmd_surgery(cfg, run_dir):
    tr = setup(cfg, run_dir, log=False)
    _require_trained(tr)
    ic = cfg["interventions"]
    for t in _tasks(cfg, tr.sc):
        sc_ = _load_scores(tr, t)
        S = sc_["scores"]
        led = _load_ledger(tr, t)
        R = _results(tr, f"surgery_task{t}")
        n_rep = int(ic.get("reps", 3))
        if _needs(R, "none", n_rep):
            R["none"] = retrain_avg(tr, t, R.get("none"), n_rep); _save_results(tr, f"surgery_task{t}", R)
        base_acc = np.array(R["none"]["group_acc"])
        # targets: most-forgotten groups that are not already at zero accuracy
        order = torch.argsort(led["true_dL"], descending=True).tolist()
        targets = [g for g in order if base_acc[g] > 0.02][: int(ic["surgery_targets"])]
        R["targets"] = targets
        q = float(ic["surgery_frac"])
        for g in targets:
            sel = {"ledger_target": S["ledger"][:, g], "tracin_cp3_target": S["tracin_cp3"][:, g],
                   "static_target": S["static_tracin"][:, g], "ledger_total": S["ledger"].sum(1), "random": S["random"][:, 0]}
            for m, s in sel.items():
                key = f"g{g}:{m}"
                if not _needs(R, key, n_rep):
                    continue
                R[key] = retrain_avg(tr, t, R.get(key), n_rep, keep=_topk_keep(s.to(tr.device), q))
                _save_results(tr, f"surgery_task{t}", R)
                gain = np.array(R[key]["group_acc"]) - base_acc
                print(f"[surgery] task {t} g{g} {m}: target {gain[g]:+.3f} others {np.delete(gain, g).mean():+.3f}", flush=True)


def cmd_lds(cfg, run_dir):
    tr = setup(cfg, run_dir, log=False)
    _require_trained(tr)
    ic = cfg["interventions"]
    t = _tasks(cfg, tr.sc)[-1]                      # LDS on the last selected task (cost)
    R = _results(tr, f"lds_task{t}")
    N = len(tr.sc.tasks[t].train_idx)
    for j in range(int(ic["lds_subsets"])):
        key = f"subset{j}"
        if key in R:
            continue
        u = hash_uniform(int(cfg["seed"]), 88, t, j, torch.arange(N, device=tr.device)).squeeze(1)
        keep = u < float(ic["lds_frac"])
        R[key] = retrain(tr, t, keep=keep)
        _save_results(tr, f"lds_task{t}", R)
        print(f"[lds] task {t} subset {j}: old {R[key]['old_acc']:.4f}", flush=True)


def cf_subsets(cfg, sc, t: int):
    """Counterfactual-validity design (RQ13): random subsets at several sizes and structured subsets
    (every new class removed in full). Returns [(key, removed_mask)] -- a pure function of the seed."""
    ic = cfg["interventions"]
    N = len(sc.tasks[t].train_idx)
    dev = sc.device
    out = []
    for si, f in enumerate(ic.get("cf_fracs", [0.01, 0.02, 0.05, 0.1, 0.2, 0.5])):
        for j in range(int(ic.get("cf_per_size", 6))):
            u = hash_uniform(int(cfg["seed"]), 91, t, 100 * si + j, torch.arange(N, device=dev)).squeeze(1)
            out.append((f"rand@{f}#{j}", u < float(f)))
    if ic.get("cf_classes", True):
        ylab = sc.ytr[sc.tasks[t].train_idx]
        for c in sc.tasks[t].classes:
            out.append((f"class@{c}", ylab == c))
    return out


def cmd_cf(cfg, run_dir):
    """Remove each subset S, retrain (same visiting orders as the full retrain), and record the realised
    change of every old-group probe loss -- the quantity the ledger predicts as -sum_{i in S} C_i."""
    tr = setup(cfg, run_dir, log=False)
    _require_trained(tr)
    ic = cfg["interventions"]
    n_rep = int(ic.get("reps", 3))
    for t in _tasks(cfg, tr.sc):
        R = _results(tr, f"cf_task{t}")
        if _needs(R, "none", n_rep):
            R["none"] = retrain_avg(tr, t, R.get("none"), n_rep)
            _save_results(tr, f"cf_task{t}", R)
        for key, removed in cf_subsets(cfg, tr.sc, t):
            if not _needs(R, key, n_rep):
                continue
            R[key] = retrain_avg(tr, t, R.get(key), n_rep, keep=~removed)
            R[key]["n_removed"] = int(removed.sum())
            _save_results(tr, f"cf_task{t}", R)
            dL = np.array(R[key]["probe_L"]) - np.array(R["none"]["probe_L"])
            print(f"[cf] task {t} {key}: |S|={int(removed.sum())} dL_probe={dL.sum():+.4f}", flush=True)
        # the ledger's own counterfactual: same batches and number of steps, the loss terms of S set to zero
        for key, removed in cf_subsets(cfg, tr.sc, t):
            zkey = "zero:" + key
            if not _needs(R, zkey, n_rep):
                continue
            R[zkey] = retrain_avg(tr, t, R.get(zkey), n_rep, zero=removed)
            R[zkey]["n_removed"] = int(removed.sum())
            _save_results(tr, f"cf_task{t}", R)
            dL = np.array(R[zkey]["probe_L"]) - np.array(R["none"]["probe_L"])
            print(f"[cf] task {t} {zkey}: |S|={int(removed.sum())} dL_probe={dL.sum():+.4f}", flush=True)


# ----------------------------------------------------------------------------- parameter-level (RQ4)

def _unit_scores(param_score: torch.Tensor, unit_of: torch.Tensor, U: int) -> torch.Tensor:
    return torch.zeros(U, dtype=torch.float64, device=param_score.device).index_add_(0, unit_of, param_score.double())


def cmd_params(cfg, run_dir):
    tr = setup(cfg, run_dir, log=False)
    _require_trained(tr)
    model, sc = tr.model, tr.sc
    ic = cfg["interventions"]
    for t in _tasks(cfg, sc):
        R = _results(tr, f"params_task{t}")
        led = _load_ledger(tr, t)
        probe = ProbeSet(sc, t, t, tr.group_by, tr.device)
        tr.load_snapshot(f"task{t}_start"); th0 = flat_params(model).clone(); F0 = fisher_diag(model, probe)
        tr.load_snapshot(f"task{t}_end");   th1 = flat_params(model).clone(); _, G1 = probe_grads(model, probe)
        end_state = {k: v.clone() for k, v in model.state_dict().items()}
        dth = th1 - th0
        cp = led["param"].to(tr.device)
        lam = cp.sum() / led["learn_param"].to(tr.device).sum().clamp_min(1e-12)
        P = {"ledger": cp, "ledger_net": cp - lam * led["learn_param"].to(tr.device),
             "abs_delta": dth.abs().double(), "fisher_delta2": (F0 * dth ** 2).double(),
             "taylor_end": (G1.sum(0) * dth).double(),
             "random": hash_uniform(int(cfg["seed"]), 99, t, 0, torch.arange(len(dth), device=tr.device)).squeeze(1).double()}
        keep_scores = ic.get("param_scores")
        if keep_scores:
            P = {k: v for k, v in P.items() if k in keep_scores}
        unit_of, meta, layer_of, lnames = unit_index(model)
        unit_of = unit_of.to(tr.device)
        U = int(unit_of.max()) + 1
        R.setdefault("entanglement", _spearman(led["unit"].sum(1).cpu().numpy(), led["learn_unit"].cpu().numpy()))
        R.setdefault("layer_share", {n: float(led["layer"][i].sum()) for i, n in enumerate(lnames)})
        R.setdefault("ref_acc", _ref_acc(tr, t))
        if "end" not in R:
            R["end"] = measure(tr, t)
        # layer rollback (causal) -------------------------------------------------
        if "layer_rollback" not in R:
            lr_ = {}
            for i, n in enumerate(lnames):
                th = th1.clone(); sel = layer_of.to(tr.device) == i; th[sel] = th0[sel]
                model.load_state_dict(end_state); set_flat_params(model, th)
                lr_[n] = measure(tr, t)["old_acc"]
            R["layer_rollback"] = lr_
        # parameter- and unit-level rollback (off-trajectory negative control) -----
        for name, s in (P.items() if ic.get("rollback", True) else []):
            us = _unit_scores(s, unit_of, U)
            for f in ic["param_fracs"]:
                for level in ("param", "unit"):
                    key = f"rollback:{level}:{name}@{f}"
                    if key in R:
                        continue
                    if level == "param":
                        sel = torch.argsort(s, descending=True)[: int(f * len(s))]
                        mask = torch.zeros(len(s), dtype=torch.bool, device=tr.device); mask[sel] = True
                    else:
                        top = torch.argsort(us, descending=True)[: max(1, int(f * U))]
                        mask = torch.isin(unit_of, top)
                    th = th1.clone(); th[mask] = th0[mask]
                    model.load_state_dict(end_state); set_flat_params(model, th)
                    R[key] = measure(tr, t)
            _save_results(tr, f"params_task{t}", R)
        # freeze-and-retrain (unit level) -----------------------------------------
        for name, s in P.items():
            us = _unit_scores(s, unit_of, U)
            for f in ic["param_fracs"]:
                key = f"freeze:unit:{name}@{f}"
                n_rep = int(ic.get("reps", 3))
                if not _needs(R, key, n_rep):
                    continue
                top = torch.argsort(us, descending=True)[: max(1, int(f * U))]
                frozen = torch.isin(unit_of, top)
                R[key] = retrain_avg(tr, t, R.get(key), n_rep, frozen=frozen)
                _save_results(tr, f"params_task{t}", R)
                print(f"[params] task {t} {key}: old {R[key]['old_acc']:.4f} new {R[key]['new_acc']:.4f}", flush=True)
        if _needs(R, "freeze:none", int(ic.get("reps", 3))):
            R["freeze:none"] = retrain_avg(tr, t, R.get("freeze:none"), int(ic.get("reps", 3)))
            _save_results(tr, f"params_task{t}", R)


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)
