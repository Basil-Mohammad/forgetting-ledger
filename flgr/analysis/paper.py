"""Generates every figure (PDF) and table (LaTeX) of the paper from the saved runs.

    python -m flgr.analysis.paper --runs ./runs --fig ./paper/figures --tab ./paper/generated

All statistics come from ``flgr.analysis.stats``. Every number printed in the paper is written to
``generated/macros.tex`` or a table file, so text and data can never drift apart.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from scipy import stats as sst

from . import stats as S
from ..utils import hash_uniform

# ----------------------------------------------------------------------------- style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
       "green": "#008300", "violet": "#4a3aa7", "red": "#e34948", "grey": "#8a8984", "dark": "#2b2b2a"}
METHOD_STYLE = {   # name: (label, colour, linestyle, marker, z)
    "ledger": ("Ledger (ours)", PAL["blue"], "-", "o", 10),
    "ledger_euler": ("Euler ledger (= idealised TracIn)", PAL["aqua"], "-", "s", 8),
    "trak": ("TRAK (projected influence)", PAL["violet"], "-", "D", 7),
    "tracin_cp10": ("TracIn-CP (10 ckpt)", PAL["orange"], "--", "^", 6),
    "tracin_cp3": ("TracIn-CP (3 ckpt)", PAL["yellow"], "--", "v", 6),
    "static_tracin": ("Gradient conflict @ start", PAL["red"], "-", "x", 6),
    "grad_cos": ("Gradient cosine @ start", PAL["magenta"], ":", "+", 5),
    "loss": ("Loss @ start", PAL["green"], ":", "*", 5),
    "feature_prox": ("Feature proximity", PAL["grey"], "-.", "p", 5),
    "random": ("Random", PAL["dark"], ":", ".", 4),
    "ledger_helpful": ("Ledger, most protective", PAL["blue"], ":", "o", 4),
    "ledger_early10": ("Ledger after 10% of task", PAL["blue"], (0, (1, 1)), "o", 3),
    "ledger_early20": ("Ledger after 20% of task", PAL["blue"], (0, (3, 1)), "o", 3),
    "ledger_early50": ("Ledger after 50% of task", PAL["blue"], (0, (5, 1)), "o", 3),
}
BENCH = {  # config-name prefix -> short label
    "pmnist-domain-mlp-finetune": "PM (MLP)",
    "sfmnist-task-mlp-finetune": "Split-FMNIST (MLP)",
    "scifar10-task-cnn-none-finetune": "Split-CIFAR-10 (CNN)",
    "scifar10-task-resnet18r-bn-finetune": "ResNet-18 BN (C10)",
    "agnews_dbpedia-task-llm-finetune": "Pythia-160M LoRA (AG to DBpedia)",
}


def texlab(m):
    """Method label for LaTeX tables (escapes %)."""
    return METHOD_STYLE[m][0].replace("%", "\\%")


def setup_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"], "mathtext.fontset": "dejavuserif",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.2, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5,
        "lines.linewidth": 1.6, "lines.markersize": 4, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "legend.frameon": False, "figure.dpi": 150,
    })


def savefig(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


# ----------------------------------------------------------------------------- loading

class Run:
    def __init__(self, d: str):
        self.dir = d
        self.name = os.path.basename(d.rstrip("/"))
        self.cfg = json.load(open(os.path.join(d, "config.json")))
        self.seed = int(self.cfg["seed"])
        self.config = re.sub(r"-s\d+$", "", self.name)
        self._cache = {}

    def has(self, *p):
        return os.path.exists(os.path.join(self.dir, *p))

    def json(self, *p):
        f = os.path.join(self.dir, *p)
        return json.load(open(f)) if os.path.exists(f) else None

    def pt(self, *p):
        k = "/".join(p)
        if k not in self._cache:
            f = os.path.join(self.dir, *p)
            self._cache[k] = torch.load(f, map_location="cpu", weights_only=False) if os.path.exists(f) else None
        return self._cache[k]

    def ledger(self, t=1):
        return self.pt("ledger", f"task{t}.pt")

    def scores(self, t=1):
        return self.pt("scores", f"task{t}.pt")

    def metrics(self):
        f = os.path.join(self.dir, "metrics.jsonl")
        return [json.loads(l) for l in open(f)] if os.path.exists(f) else []

    def complete(self):
        return self.has("eval", f"task{int(self.cfg['n_tasks']) - 1}.json")


def load_runs(root: str) -> Dict[str, List[Run]]:
    out = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.exists(os.path.join(d, "config.json")):
            r = Run(d)
            if r.complete():
                out[r.config].append(r)
    for k in out:
        out[k].sort(key=lambda r: r.seed)
    return out


# ----------------------------------------------------------------------------- per-run quantities

def forgetting_pp(R: dict, key: str) -> float:
    ref = np.array(R["ref_acc"])
    return float(np.mean(ref - np.array(R[key]["task_acc"][:len(ref)])))


def removal_rho(run: Run, t=1):
    """Per-seed outcome of removal-and-retrain for every (method, q):
    dF   = forgetting reduction in accuracy points (old-task accuracy gained vs. the full retrain),
    dTL  = reduction of the old-task test loss, rho = dF / F0, cost = new-task accuracy lost."""
    R = run.json("interventions", f"removal_task{t}.json")
    if not R or "none" not in R:
        return None
    F0 = forgetting_pp(R, "none")
    TL0 = R["none"].get("old_test_loss", np.nan)
    out = {}
    for k in R:
        if "@" not in k:
            continue
        m, q = k.split("@")
        Fk = forgetting_pp(R, k)
        out[(m, float(q))] = dict(rho=(F0 - Fk) / F0 if abs(F0) > 1e-9 else np.nan, dF=F0 - Fk,
                                  dTL=TL0 - R[k].get("old_test_loss", np.nan),
                                  cost=R["none"]["new_acc"] - R[k]["new_acc"], time=R[k].get("time_s", np.nan),
                                  n_rep=len(R[k].get("reps", [1])))
    return dict(F0=F0, TL0=TL0, rows=out, t_plain=R["none"].get("time_s", np.nan), steps=R["none"].get("steps", np.nan),
                n_rep=len(R["none"].get("reps", [1])))


def harm_vector(run: Run, t=1, which="ledger"):
    sc = run.scores(t)
    if sc is None or which not in sc["scores"]:
        if which == "ledger" and run.ledger(t) is not None:
            return run.ledger(t)["data"].sum(1).numpy()
        return None
    return sc["scores"][which].sum(1).numpy()


# ----------------------------------------------------------------------------- analyses

def completeness_rows(runs: List[Run]):
    rows = []
    for r in runs:
        led = r.ledger(1)
        if led is None:
            continue
        true, pred = led["true_dL"].numpy(), (led["path_g"] + led["stats"]).numpy()
        eps = np.abs(pred - true).sum() / max(np.abs(true).sum(), 1e-12)
        eps_tv = np.abs(pred - true).sum() / max(float(led["path_var"].sum()), 1e-12)
        eul = led.get("data_euler")
        e_eul = float(np.abs(eul.sum(0).numpy() - (led["path_g"]).numpy()).sum() / max(np.abs(led["path_g"].numpy()).sum(), 1e-12)) \
            if eul is not None and float(eul.abs().sum()) > 0 else np.nan
        e_eul_tot = float(eul.sum().item() / max(abs(float(true.sum())), 1e-12)) if eul is not None and float(eul.abs().sum()) > 0 else np.nan
        sc = r.scores(1)
        st = sc["single"] if sc is not None else None
        rows.append(dict(seed=r.seed, eps=eps, eps_tv=eps_tv, evals=float(led["n_evals"]) / max(1, led["steps"]),
                         dL=float(true.sum()), euler_err=e_eul, euler_ratio=e_eul_tot,
                         start=float(np.sum(st["start"]) / np.sum(st["true"])) if st else np.nan,
                         end=float(np.sum(st["end"]) / np.sum(st["true"])) if st else np.nan,
                         start_g=np.array(st["start"]) if st else None, end_g=np.array(st["end"]) if st else None,
                         true_g=np.array(st["true"]) if st else None, ledger_g=np.array(st["ledger"]) if st else None,
                         steps=int(led["steps"]),
                         stats_ratio=float(led["stats"].sum() / led["true_dL"].sum()) if abs(float(led["true_dL"].sum())) > 1e-12 else np.nan,
                         stats_abs=float(led["stats"].abs().sum() / max(float(led["true_dL"].abs().sum()), 1e-12)),
                         path_ratio=float(led["path_g"].sum() / led["true_dL"].sum()) if abs(float(led["true_dL"].sum())) > 1e-12 else np.nan))
    return rows


def concentration(runs: List[Run]):
    """Share of positive harm and of net forgetting carried by the top-x% samples."""
    out = []
    for r in runs:
        h = harm_vector(r)
        if h is None:
            continue
        frac, cum, gini = S.lorenz(h)
        pos = np.clip(h, 0, None)
        net = h.sum()
        srt = np.sort(h)[::-1]
        k1, k10 = max(1, int(0.01 * len(h))), max(1, int(0.1 * len(h)))
        out.append(dict(seed=r.seed, frac=frac, cum=cum, gini=gini, top1=srt[:k1].sum() / pos.sum(),
                        top10=srt[:k10].sum() / pos.sum(), top10_net=srt[:k10].sum() / net if net > 0 else np.nan,
                        frac_harmful=float((h > 0).mean()), n=len(h)))
    return out


TAILS = [0.01, 0.02, 0.05, 0.1, 0.2]


def stability(runs: List[Run], which="ledger"):
    """Cross-seed reproducibility of per-sample harm: pairwise Spearman (all samples) and the overlap of
    the top-q most harmful sets for several q (enrichment = overlap / q; 1 = chance)."""
    V = [harm_vector(r, which=which) for r in runs]
    V = [v for v in V if v is not None]
    if len(V) < 3:
        return None
    st = S.pairwise_rank_stability(V)
    st["n_seeds"] = len(V)
    st["overlap"] = {}
    for q in TAILS:
        k = max(1, int(q * len(V[0])))
        tops = [set(np.argsort(-v)[:k]) for v in V]
        ov = [len(a & b) / k for i, a in enumerate(tops) for b in tops[i + 1:]]
        st["overlap"][q] = S.mean_ci(ov)
    st["top10_overlap"] = st["overlap"][0.1]
    # magnitude vs. direction: leverage |harm| and "double-edged" samples (top harmful in one seed,
    # top protective in another)
    st["abs"] = S.pairwise_rank_stability([np.abs(v) for v in V])
    st["flip"] = {}
    for q in TAILS:
        k = max(1, int(q * len(V[0])))
        tops = [set(np.argsort(-v)[:k]) for v in V]
        bots = [set(np.argsort(v)[:k]) for v in V]
        ov = [len(tops[i] & bots[j]) / k for i in range(len(V)) for j in range(len(V)) if i != j]
        st["flip"][q] = S.mean_ci(ov)
    return st


def class_map_stability(runs: List[Run]):
    M = [r.scores(1)["imap"].numpy().ravel() for r in runs if r.scores(1) is not None]
    if len(M) < 3:
        return None
    return S.pairwise_rank_stability(M)


def removal_table(runs: List[Run], methods, qs):
    per = defaultdict(list)          # (m,q) -> list of dicts aligned by seed
    seeds = []
    F0s = []
    for r in runs:
        rr = removal_rho(r)
        if rr is None:
            continue
        seeds.append(r.seed)
        F0s.append(rr["F0"])
        for m in methods:
            for q in qs:
                per[(m, q)].append(rr["rows"].get((m, q), dict(rho=np.nan, cost=np.nan, dF=np.nan, dTL=np.nan)))
    return per, seeds, F0s


def lds(runs: List[Run]):
    res = defaultdict(list)
    for r in runs:
        R = r.json("interventions", "lds_task1.json")
        sc = r.scores(1)
        if not R or sc is None:
            continue
        subs = sorted([k for k in R if k.startswith("subset")], key=lambda k: int(k[6:]))[:16]   # same 16 subsets for every seed
        if len(subs) < 8:
            continue
        N = sc["scores"]["ledger"].shape[0]
        frac = r.cfg["interventions"]["lds_frac"]
        masks = torch.stack([(hash_uniform(int(r.cfg["seed"]), 88, 1, int(k[6:]), torch.arange(N)).squeeze(1) < frac) for k in subs]).float()
        actual = np.array([R[k]["probe_L"] for k in subs])
        for m, s in sc["scores"].items():
            pred = (masks @ s.float()).numpy()
            rhos = [sst.spearmanr(pred[:, g], actual[:, g]).correlation for g in range(actual.shape[1])]
            res[m].append((r.seed, float(np.nanmean(rhos))))
    return res


def surgery(runs: List[Run]):
    res = defaultdict(lambda: defaultdict(list))
    for r in runs:
        R = r.json("interventions", "surgery_task1.json")
        if not R or "none" not in R:
            continue
        base = np.array(R["none"]["group_acc"])
        for g in R.get("targets", []):
            for k, v in R.items():
                if not k.startswith(f"g{g}:"):
                    continue
                m = k.split(":", 1)[1]
                gain = np.array(v["group_acc"]) - base
                oth = np.delete(gain, g)
                res[m]["target"].append(gain[g]); res[m]["others"].append(oth.mean())
                res[m]["seed"].append(r.seed)
    return res


def anatomy(runs: List[Run], top: float = 0.05):
    """How do the most harmful samples differ from the rest? Standardised mean difference
    (Cohen's d, pooled SD) of each property between the top-`top` harmful samples and all others, per seed."""
    feats = ["loss_start", "margin_start", "gradnorm_start", "prox_start", "first_correct_epoch", "loss_end"]
    res = defaultdict(list)
    for r in runs:
        sc = r.scores(1)
        if sc is None or "features" not in sc:
            continue
        h = sc["scores"]["ledger"].sum(1).numpy()
        k = max(1, int(top * len(h)))
        mask = np.zeros(len(h), bool); mask[np.argsort(-h)[:k]] = True
        for f in feats:
            if f in sc["features"]:
                v = sc["features"][f].numpy().astype(float)
                a_, b_ = v[mask], v[~mask]
                sd = np.sqrt(((len(a_) - 1) * a_.var(ddof=1) + (len(b_) - 1) * b_.var(ddof=1)) / (len(v) - 2))
                res[f].append((a_.mean() - b_.mean()) / sd if sd > 0 else np.nan)
    return res


def params(runs: List[Run]):
    res = defaultdict(lambda: defaultdict(list))
    ent = []
    for r in runs:
        R = r.json("interventions", "params_task1.json")
        if not R or "end" not in R:
            continue
        ent.append(R.get("entanglement", np.nan))
        ref = float(np.mean(R["ref_acc"]))
        fn = R.get("freeze:none")
        for k, v in R.items():
            if k.startswith("freeze:unit:") and fn is not None:
                name, frac = k[len("freeze:unit:"):].split("@")
                den = ref - fn["old_acc"]
                res[("freeze", name, float(frac))]["rec"].append((v["old_acc"] - fn["old_acc"]) / den if abs(den) > 1e-9 else np.nan)
                res[("freeze", name, float(frac))]["cost"].append(fn["new_acc"] - v["new_acc"])
            elif k.startswith("rollback:unit:"):
                name, frac = k[len("rollback:unit:"):].split("@")
                den = ref - R["end"]["old_acc"]
                res[("rollback", name, float(frac))]["rec"].append((v["old_acc"] - R["end"]["old_acc"]) / den if abs(den) > 1e-9 else np.nan)
                res[("rollback", name, float(frac))]["cost"].append(R["end"]["new_acc"] - v["new_acc"])
    return res, ent


# ----------------------------------------------------------------------------- figures

def fig_concentration(groups, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    cols = [PAL["blue"], PAL["orange"], PAL["aqua"], PAL["violet"]]
    summ = {}
    for (cname, lab), c in zip(groups, cols):
        con = concentration(groups[(cname, lab)])
        if not con:
            continue
        grid = np.linspace(0, 1, 201)
        curves = np.array([np.interp(grid, d["frac"], d["cum"]) for d in con])
        m = curves.mean(0)
        lo, hi = np.quantile(curves, [0.025, 0.975], axis=0)
        ax.fill_between(grid * 100, lo * 100, hi * 100, color=c, alpha=0.15, lw=0)
        ax.plot(grid * 100, m * 100, color=c, label=lab)
        summ[lab] = con
    ax.plot([0, 100], [0, 100], color=PAL["grey"], lw=0.8, ls=":")
    ax.set_xlim(0, 60); ax.set_ylim(0, 105)
    ax.set_xlabel("most harmful new-task samples (%)")
    ax.set_ylabel("share of harmful ledger mass (%)")
    ax.axvline(10, color=PAL["grey"], lw=0.6, ls="--")
    ax.legend(loc="lower right")
    ax.set_title("(a) Harm is concentrated in few samples", loc="left")
    ax = axes[1]
    labs = list(summ)
    x = np.arange(len(labs))
    for j, (key, c, name) in enumerate([("top1", PAL["violet"], "top 1%"), ("top10", PAL["blue"], "top 10%")]):
        vals = [S.mean_ci([d[key] for d in summ[l]]) for l in labs]
        ax.bar(x + (j - 0.5) * 0.36, [100 * v["mean"] for v in vals], 0.34, color=c, label=name,
               yerr=[[100 * (v["mean"] - v["lo"]) for v in vals], [100 * (v["hi"] - v["mean"]) for v in vals]],
               error_kw=dict(lw=0.8, capsize=2, ecolor="#52514e"))
    short = {"PM (MLP)": "PM\n(MLP)", "Split-FMNIST (MLP)": "SF\n(MLP)", "Split-CIFAR-10 (CNN)": "C10\n(CNN)",
             "ResNet-18 BN (C10)": "C10\n(ResNet-BN)", "Pythia-160M LoRA (AG to DBpedia)": "AG$\\to$DBp\n(Pythia)"}
    ax.set_xticks(x); ax.set_xticklabels([short.get(l, l) for l in labs])
    ax.set_ylabel("share of harmful mass (%)")
    ax.legend(loc="upper right")
    ax.set_title("(b) Share carried by the top samples", loc="left")
    fig.tight_layout()
    savefig(fig, os.path.join(out, "fig_concentration.pdf"))
    return summ


MAIN_REMOVAL = ["ledger", "trak", "tracin_cp10", "static_tracin", "loss", "random"]


def fig_removal(groups, out, methods, qs_by):
    """Forgetting prevented (top) and new-task cost (bottom) vs removal budget. The figure shows the
    main comparators only; every score is in the tables."""
    n = len(groups)
    fig, axes = plt.subplots(2, n, figsize=(2.45 * n + 0.5, 3.9), sharex="col", squeeze=False)
    tabs = {}
    for j, ((cname, lab), runs) in enumerate(groups.items()):
        per, seeds, F0s = removal_table(runs, methods + ["ledger_helpful"], qs_by[cname])
        qs = [q for q in qs_by[cname] if any(np.isfinite(d["dF"]) for d in per[("ledger", q)])]
        tabs[lab] = (per, seeds, F0s, qs)
        for m in MAIN_REMOVAL:
            if not any(np.isfinite(d["dF"]) for q in qs for d in per[(m, q)]):
                continue
            lab_m, c, ls, mk, z = METHOD_STYLE[m]
            x = [100 * q for q in qs]
            for row, key in [(0, "dF"), (1, "cost")]:
                ms = [S.mean_ci([d[key] for d in per[(m, q)]]) for q in qs]
                ax = axes[row, j]
                ax.plot(x, [100 * v["mean"] for v in ms], color=c, ls=ls, marker=mk, zorder=z,
                        label=lab_m if (j == 0 and row == 0) else None,
                        lw=2.2 if m == "ledger" else 1.2, ms=4.5 if m == "ledger" else 3.4)
                if m in ("ledger", "static_tracin", "random"):
                    ax.fill_between(x, [100 * v["lo"] for v in ms], [100 * v["hi"] for v in ms], color=c, alpha=0.13, lw=0, zorder=z - 1)
        for row in (0, 1):
            ax = axes[row, j]
            ax.axhline(0, color=PAL["grey"], lw=0.7)
            ax.set_xscale("log")
            ax.set_xticks([100 * q for q in qs]); ax.set_xticklabels([f"{100 * q:g}" for q in qs])
            ax.minorticks_off()
        axes[0, j].set_title(f"{lab}\nforgetting {np.mean(F0s) * 100:.1f} pp, {len(seeds)} seeds", loc="left", fontsize=8.3)
        axes[1, j].set_xlabel("removed new-task samples (%)")
    axes[0, 0].set_ylabel("forgetting prevented\n(old-task acc. points)")
    axes[1, 0].set_ylabel("new-task accuracy\ncost (points)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.6)
    savefig(fig, os.path.join(out, "fig_removal.pdf"))
    return tabs


FAM_CONF = ["random", "static_tracin", "grad_cos", "loss", "feature_prox"]
FAM_TRAJ = ["ledger_euler", "trak", "tracin_cp10", "tracin_cp3", "ledger_early20"]


def paired_removal(tabs, q0=0.1):
    """Paired differences (ledger - comparator) of forgetting prevented at budget q0, per benchmark and
    pooled over all benchmark x seed units. Holm correction within the two pre-specified families."""
    res = defaultdict(dict)
    for lab, (per, seeds, F0s, qs) in tabs.items():
        if q0 not in qs:
            continue
        base = np.array([d["dF"] for d in per[("ledger", q0)]])
        for m in FAM_CONF + FAM_TRAJ:
            v = np.array([d["dF"] for d in per[(m, q0)]])
            if not np.isfinite(v).any():
                continue
            res[m][lab] = dict(d=base - v, **S.paired(base, v))
    for m in list(res):
        d = np.concatenate([r["d"] for r in res[m].values()])
        res[m]["pooled"] = dict(d=d, n=int(np.isfinite(d).sum()), diff=S.mean_ci(d), p_perm=S.signflip_test(d),
                                dz=S.cohen_dz(d), wins=int((d > 0).sum()))
    for scope in list(tabs) + ["pooled"]:
        for fam in (FAM_CONF, FAM_TRAJ):
            ph = S.holm({m: res[m][scope]["p_perm"] for m in fam if m in res and scope in res[m]})
            for m, p in ph.items():
                if m in res and scope in res[m]:
                    res[m][scope]["p_holm"] = p
    return res


def fig_forest(pr, labs, out):
    """Forest plot: paired difference in forgetting prevented, ledger minus comparator."""
    order = [m for m in FAM_TRAJ + FAM_CONF if m in pr]
    scopes = [l for l in labs if any(l in pr[m] for m in order)] + ["pooled"]
    cols = [PAL["blue"], PAL["orange"], PAL["aqua"], PAL["violet"], PAL["dark"]]
    mks = ["o", "s", "^", "v", "D"]
    fig, ax = plt.subplots(figsize=(5.6, 0.36 * len(order) + 1.2))
    off = np.linspace(-0.27, 0.27, len(scopes))
    for i, m in enumerate(order):
        for k, sc in enumerate(scopes):
            r = pr[m].get(sc)
            if r is None:
                continue
            ci = r["diff"]
            y = i + off[k]
            pooled = sc == "pooled"
            ax.errorbar(100 * ci["mean"], y, xerr=[[100 * (ci["mean"] - ci["lo"])], [100 * (ci["hi"] - ci["mean"])]],
                        fmt=mks[k if not pooled else -1], color=cols[k if not pooled else -1], ms=5 if pooled else 3.6,
                        mfc=cols[-1] if pooled else "white", capsize=0, lw=1.6 if pooled else 0.9,
                        label=(sc if not pooled else "pooled (all benchmarks)") if i == 0 else None)
            if pooled:
                p = r.get("p_holm", np.nan)
                ps = S.fmt_p(p)
                ps = ps.replace("\\ensuremath{<10^{-4}}", "$<10^{-4}$")
                ax.text(1.01, 1 - (y + 0.5) / len(order), f"{r['wins']}/{r['n']}   " + (ps if ps.startswith("<") else ps), transform=ax.transAxes,
                        fontsize=6.4, va="center", color="#2b2b2a")
    ax.axvline(0, color=PAL["dark"], lw=0.8)
    ax.axhline(len([m for m in FAM_TRAJ if m in pr]) - 0.5, color=PAL["grey"], lw=0.6, ls="--")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([METHOD_STYLE[m][0] for m in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("difference in forgetting prevented, ledger $-$ comparator\n(old-task accuracy points, $q$ = 10%)")
    ax.text(1.01, 1.0, "wins   $p_{\\mathrm{Holm}}$", transform=ax.transAxes, fontsize=6.4, va="bottom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=6.4, ncol=4, frameon=False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    savefig(fig, os.path.join(out, "fig_forest.pdf"))


def fig_stability(groups, transfer, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.45), gridspec_kw=dict(width_ratios=[1.0, 1.2, 1.1]))
    first_key = list(groups)[0]
    first = groups[first_key]
    ax = axes[0]
    a, b = (harm_vector(first[0]), harm_vector(first[1])) if len(first) > 1 else (None, None)
    if a is not None and b is not None:
        ra, rb = sst.rankdata(-a) / len(a) * 100, sst.rankdata(-b) / len(b) * 100
        both = (ra <= 10) & (rb <= 10)
        ax.scatter(ra[~both], rb[~both], s=1.0, color=PAL["grey"], alpha=0.25, lw=0, rasterized=True)
        ax.scatter(ra[both], rb[both], s=2.5, color=PAL["red"], alpha=0.8, lw=0, rasterized=True, label="top 10% in both")
        ax.axvline(10, color=PAL["red"], lw=0.6, ls="--"); ax.axhline(10, color=PAL["red"], lw=0.6, ls="--")
        ax.set_title(f"(a) {first_key[1].split(' (')[0]}: two seeds", loc="left")
        ax.legend(loc="upper right", fontsize=6.2, markerscale=3)
    ax.set_xlabel("harm rank, seed 0 (%)"); ax.set_ylabel("harm rank, seed 1 (%)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax = axes[1]
    stab = {}
    styles = [("ledger", "-"), ("tracin_cp10", "--"), ("static_tracin", "-"), ("loss", ":")]
    cols = [PAL["blue"], PAL["orange"], PAL["aqua"], PAL["violet"]]
    for (key, runs), c in zip(groups.items(), cols):
        for m, ls in styles:
            st = stability(runs, which=m)
            stab[(key[1], m)] = st
            if st is None or m not in ("ledger",):
                continue
            q = TAILS
            en = [st["overlap"][x]["mean"] / x for x in q]
            lo = [st["overlap"][x]["lo"] / x for x in q]
            hi = [st["overlap"][x]["hi"] / x for x in q]
            ax.plot([100 * x for x in q], en, color=c, ls=ls, marker="o", label=key[1].split(" (")[0])
            ax.fill_between([100 * x for x in q], lo, hi, color=c, alpha=0.15, lw=0)
            fl = [st["flip"][x]["mean"] / x for x in q]
            ax.plot([100 * x for x in q], fl, color=c, ls="--", marker="o", mfc="white", ms=3, lw=1.0)
    ax.axhline(1, color=PAL["dark"], lw=0.8, ls=":")
    ax.text(1.05, 1.08, "chance", fontsize=6.5, color=PAL["dark"])
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(["1", "2", "5", "10", "20"])
    ax.set_xlabel("size of the most-harmful set (%)")
    ax.set_ylabel("cross-seed overlap / chance")
    from matplotlib.lines import Line2D
    h_, l_ = ax.get_legend_handles_labels()
    h_ += [Line2D([], [], color=PAL["grey"], marker="o", ls="-"), Line2D([], [], color=PAL["grey"], marker="o", mfc="white", ls="--")]
    l_ += ["harmful in both seeds", "harmful in one, protective in other"]
    ax.legend(h_, l_, fontsize=5.6, loc="upper right")
    ax.set_title("(b) Reproducible tail", loc="left")
    ax = axes[2]
    items = [(k, v) for k, v in transfer.items() if v]
    if items:
        x = np.arange(len(items))
        for j, (key, col, lb) in enumerate([("signed", PAL["violet"], "harm (signed)"), ("abs", PAL["aqua"], "leverage |harm|")]):
            vv = [v[key] for _, v in items]
            ax.bar(x + (j - 0.5) * 0.36, [v["mean"] for v in vv], 0.34, color=col, label=lb,
                   yerr=[[v["mean"] - v["lo"] for v in vv], [v["hi"] - v["mean"] for v in vv]], error_kw=dict(lw=0.7, capsize=1.5))
        for i, (_, v) in enumerate(items):
            ax.text(i, 0.93, f"top 1%:\n{v['top1']['mean'] / 0.01:.0f}$\\times$", ha="center", va="top", fontsize=5.8, color="#2b2b2a")
        ax.set_xticks(x); ax.set_xticklabels([k.replace(" MLP -> ", "\n$\\to$").replace("wide MLP", "wide") for k, _ in items], fontsize=6.3)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=5.8, loc="upper left", bbox_to_anchor=(0.0, 0.8))
    ax.axhline(0, color=PAL["grey"], lw=0.7)
    ax.set_ylabel("Spearman with MLP harm")
    ax.set_title("(c) Transfer across architectures", loc="left")
    fig.tight_layout()
    savefig(fig, os.path.join(out, "fig_stability.pdf"))
    return stab


def fig_imap(run_groups, out, names_by):
    items = [(lab, runs) for (c, lab), runs in run_groups.items() if runs and runs[0].scores(1) is not None]
    if not items:
        return {}
    fig, axes = plt.subplots(1, len(items), figsize=(3.2 * len(items), 2.7))
    axes = np.atleast_1d(axes)
    out_maps = {}
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", [PAL["blue"], "#f4f3ef", PAL["red"]])
    for ax, (lab, runs) in zip(axes, items):
        M = np.mean([r.scores(1)["imap"].numpy() for r in runs if r.scores(1) is not None], 0)
        sc = runs[0].scores(1)
        new_c = sc["new_classes"]
        g_names = [g["name"] for g in sc["groups"]]
        names = names_by(runs[0])
        rows = [names[c] for c in new_c]
        vmax = np.abs(M).max()
        im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
        ax.set_xticks(range(len(g_names))); ax.set_xticklabels(g_names, rotation=40, ha="right")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
        ax.set_xlabel("old class (forgotten)"); ax.set_ylabel("new class (cause)")
        ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if abs(M[i, j]) > 0.45 * vmax:
                    ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=6, color="white")
        ax.set_title(f"{lab}", loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("ledger mass (nats)", fontsize=7); cb.outline.set_linewidth(0.4)
        out_maps[lab] = (M, rows, g_names)
    fig.tight_layout()
    savefig(fig, os.path.join(out, "fig_interference.pdf"))
    return out_maps


def fig_numerics(core, rules, lrs, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.45))
    # (a) single-point ratios per group (scatter vs truth), pooled core benchmarks
    ax = axes[0]
    for (key, lab), c in zip(core, [PAL["blue"], PAL["orange"], PAL["aqua"], PAL["violet"]]):
        rows = completeness_rows(core[(key, lab)])
        T = np.concatenate([r["true_g"] for r in rows if r["true_g"] is not None]) if rows else np.array([])
        if len(T) == 0:
            continue
        St = np.concatenate([r["start_g"] for r in rows if r["start_g"] is not None])
        En = np.concatenate([r["end_g"] for r in rows if r["end_g"] is not None])
        Le = np.concatenate([r["ledger_g"] for r in rows if r["ledger_g"] is not None])
        ax.scatter(T, St, s=5, color=PAL["red"], marker="x", lw=0.6, alpha=0.7, label="gradient @ start" if c == PAL["blue"] else None)
        ax.scatter(T, En, s=5, color=PAL["orange"], marker="^", lw=0, alpha=0.7, label="gradient @ end" if c == PAL["blue"] else None)
        ax.scatter(T, Le, s=6, color=PAL["blue"], marker="o", lw=0, alpha=0.8, label="ledger" if c == PAL["blue"] else None)
    lim = ax.get_xlim()
    ax.plot(lim, lim, color=PAL["grey"], lw=0.8, ls=":")
    ax.set_xlabel("realised $\\Delta\\mathcal{L}_g$ (nats)"); ax.set_ylabel("explained $\\Delta\\mathcal{L}_g$ (nats)")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_title("(a) Explanation vs. truth", loc="left")
    # (b) integration rule
    ax = axes[1]
    names = ["euler", "trapezoid", "simpson", "adaptive"]
    vals, ev = [], []
    for n in names:
        rows = completeness_rows(rules.get(n, []))
        vals.append(S.mean_ci([r["eps"] for r in rows]))
        ev.append(np.mean([r["evals"] for r in rows]) if rows else np.nan)
    x = np.arange(len(names))
    ax.bar(x, [100 * v["mean"] for v in vals], 0.6, color=[PAL["red"], PAL["yellow"], PAL["aqua"], PAL["blue"]],
           yerr=[[100 * (v["mean"] - v["lo"]) for v in vals], [100 * (v["hi"] - v["mean"]) for v in vals]], error_kw=dict(lw=0.8, capsize=2))
    if any(np.isfinite(v["mean"]) and v["mean"] > 0 for v in vals):
        ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(["Euler\n(TracIn)", "trapez.", "Simpson", "adaptive\n(ours)"], fontsize=7)
    for i, e in enumerate(ev):
        if np.isfinite(e) and np.isfinite(vals[i]["mean"]):
            ax.text(i, 100 * vals[i]["hi"] * 1.25, f"{e + 1:.1f}", ha="center", fontsize=6.3, color="#52514e")
    ax.set_ylabel("completeness error $\\varepsilon$ (%)")
    ax.set_title("(b) Integration rule (nodes/step)", loc="left")
    # (c) step size
    ax = axes[2]
    lr_keys = sorted(lrs)
    for key, c, lab in [("eps", PAL["blue"], "adaptive (ours)"), ("euler_err", PAL["red"], "Euler (per-sample)")]:
        vv = [S.mean_ci([r[key] for r in completeness_rows(lrs[k])]) for k in lr_keys]
        if not any(np.isfinite(v["mean"]) for v in vv):
            continue
        ax.errorbar(lr_keys, [100 * v["mean"] for v in vv], yerr=[[100 * (v["mean"] - v["lo"]) for v in vv], [100 * (v["hi"] - v["mean"]) for v in vv]],
                    color=c, marker="o", label=lab, capsize=2, lw=1.4)
    if lr_keys:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(lr_keys); ax.set_xticklabels([f"{k:g}" for k in lr_keys]); ax.minorticks_off()
        ax.set_yticks([0.5, 1, 2, 5]); ax.set_yticklabels(["0.5", "1", "2", "5"])
    ax.set_xlabel("learning rate $\\eta$"); ax.set_ylabel("completeness error (%)")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_title("(c) Step size", loc="left")
    fig.tight_layout()
    savefig(fig, os.path.join(out, "fig_numerics.pdf"))


def fig_dose(dose, out):
    levels = sorted(dose)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.2))
    axes = axes.ravel()
    rec = defaultdict(list)
    for pf in levels:
        for r in dose[pf]:
            ev = r.json("eval", "task1.json"); e0 = r.json("eval", "task0.json")
            if ev is None or e0 is None:
                continue
            forg = e0["task_acc"][0] - ev["task_acc"][0]
            con = concentration([r])
            led = r.ledger(1)
            lay = led["layer"].sum(1).numpy()
            names = led["layer_names"]
            absl = np.abs(lay)
            head = sum(absl[i] for i, n in enumerate(names) if n.startswith("head")) / max(absl.sum(), 1e-12)
            first = sum(absl[i] for i, n in enumerate(names) if n.startswith("layers.0")) / max(absl.sum(), 1e-12)
            ent = sst.spearmanr(led["unit"].sum(1).numpy(), led["learn_unit"].numpy()).correlation
            rr = removal_rho(r)
            rho_l = 100 * rr["rows"].get(("ledger", 0.1), {}).get("dF", np.nan) if rr else np.nan
            rho_s = 100 * rr["rows"].get(("static_tracin", 0.1), {}).get("dF", np.nan) if rr else np.nan
            rho_r = 100 * rr["rows"].get(("random", 0.1), {}).get("dF", np.nan) if rr else np.nan
            dLtot = float(led["true_dL"].sum())
            rec["pf"].append(pf); rec["seed"].append(r.seed); rec["forg"].append(100 * forg)
            rec["top10"].append(con[0]["top10"] if con else np.nan); rec["head"].append(head); rec["first"].append(first)
            rec["ent"].append(ent); rec["rho_l"].append(rho_l); rec["rho_s"].append(rho_s); rec["rho_r"].append(rho_r)
            rec["dL"].append(dLtot)
    rec = {k: np.array(v) for k, v in rec.items()}
    if not len(rec.get("pf", [])):
        plt.close(fig)
        return None
    def panel(ax, ys, cols, labs, ylabel, title, scale=1.0):
        for y, c, l in zip(ys, cols, labs):
            ms = [S.mean_ci(rec[y][rec["pf"] == pf]) for pf in levels]
            ax.errorbar(levels, [scale * m["mean"] for m in ms], yerr=[[scale * (m["mean"] - m["lo"]) for m in ms], [scale * (m["hi"] - m["mean"]) for m in ms]],
                        color=c, marker="o", capsize=2, lw=1.4, label=l)
        ax.set_xlabel("permuted fraction $f$ (1 = orthogonal)" if ax in axes[2:] else "")
        ax.set_xticks(levels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
    panel(axes[0], ["forg"], [PAL["dark"]], [None], "forgetting (pp)", "(a) Forgetting")
    panel(axes[1], ["top10"], [PAL["blue"]], [None], "top-10% share of harm (%)", "(b) Concentration", 100)
    panel(axes[2], ["head", "first"], [PAL["orange"], PAL["aqua"]], ["output layer", "first layer"], "share of |ledger| mass (%)", "(c) Where it lives", 100)
    axes[2].legend(fontsize=6.5, loc="center right")
    panel(axes[3], ["rho_l", "rho_s", "rho_r"], [PAL["blue"], PAL["red"], PAL["dark"]], ["ledger", "gradient conflict", "random"],
          "forgetting prevented by\nremoving 10% (acc. points)", "(d) Causal removal", 1)
    axes[3].axhline(0, color=PAL["grey"], lw=0.7)
    axes[3].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    savefig(fig, os.path.join(out, "fig_dose.pdf"))
    trends = {k: S.mixed_trend(rec["pf"], rec[k], rec["seed"]) for k in ["forg", "top10", "head", "first", "ent", "rho_l", "rho_s", "rho_r"]}
    return dict(rec=rec, trends=trends, levels=levels)


def fig_sources(learners, out):
    order = ["finetune", "ewc", "agem", "er", "derpp"]
    labs = {"finetune": "Fine-tune", "ewc": "Online EWC", "agem": "A-GEM", "er": "ER", "derpp": "DER++"}
    data = {}
    for l in order:
        runs = learners.get(l, [])
        rows = []
        for r in runs:
            tot = defaultdict(float)
            for t in (1, 2):
                led = r.ledger(t)
                if led is None:
                    continue
                d = led["data"]
                tot["new_pos"] += float(d.clamp(min=0).sum()); tot["new_neg"] += float(d.clamp(max=0).sum())
                if led["mem"] is not None:
                    m = led["mem"]
                    tot["mem_pos"] += float(m.clamp(min=0).sum()); tot["mem_neg"] += float(m.clamp(max=0).sum())
                tot["reg"] += float(led["reg"].sum()); tot["stats"] += float(led["stats"].sum())
                tot["net"] += float(led["true_dL"].sum())
            ev = r.json("eval", "task2.json")
            tot["acc"] = float(np.mean(ev["task_acc"])) if ev else np.nan
            rows.append(tot)
        if rows:
            data[l] = rows
    if not data:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.75), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    ls = [l for l in order if l in data]
    x = np.arange(len(ls))
    comps = [("new_pos", "new data: harmful", PAL["red"]), ("new_neg", "new data: protective", PAL["magenta"]),
             ("mem_pos", "replay: harmful", PAL["yellow"]), ("mem_neg", "replay: protective", PAL["aqua"]),
             ("reg", "regulariser", PAL["violet"])]
    bottom_p = np.zeros(len(ls)); bottom_n = np.zeros(len(ls))
    for key, lab, c in comps:
        v = np.array([np.mean([d.get(key, 0.0) for d in data[l]]) for l in ls])
        pos = np.clip(v, 0, None); neg = np.clip(v, None, 0)
        ax.bar(x, pos, 0.6, bottom=bottom_p, color=c, label=lab, edgecolor="white", lw=0.6)
        ax.bar(x, neg, 0.6, bottom=bottom_n, color=c, edgecolor="white", lw=0.6)
        bottom_p += pos; bottom_n += neg
    net = [S.mean_ci([d["net"] for d in data[l]]) for l in ls]
    ax.errorbar(x, [n["mean"] for n in net], yerr=[[n["mean"] - n["lo"] for n in net], [n["hi"] - n["mean"] for n in net]],
                fmt="D", color="black", ms=4, capsize=2, label="net $\\Delta\\mathcal{L}$ (= sum)")
    ax.axhline(0, color=PAL["grey"], lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([labs[l] for l in ls], rotation=25, ha="right")
    ax.set_ylabel("ledger mass (nats; old classes,\nsummed over tasks 2 and 3)")
    ax.legend(fontsize=6.0, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("(a) Sources of the realised forgetting", loc="left")
    ax = axes[1]
    # where does the gross harm go? fraction cancelled by protective new samples, by replay, by the
    # regulariser; the remainder is the realised net forgetting
    parts = [("new_neg", "cancelled by protective\nnew samples", PAL["magenta"]), ("mem", "cancelled by replay", PAL["aqua"]),
             ("reg", "cancelled by regulariser", PAL["violet"]), ("net", "realised (net)", PAL["red"])]
    bottom = np.zeros(len(ls))
    for key, lab, c in parts:
        v = []
        for l in ls:
            f = []
            for d in data[l]:
                gross = d.get("new_pos", 0) + d.get("mem_pos", 0)
                val = dict(new_neg=-d.get("new_neg", 0), mem=-d.get("mem_neg", 0), reg=-d.get("reg", 0), net=d["net"])[key]
                f.append(val / max(gross, 1e-9))
            v.append(100 * np.mean(f))
        v = np.array(v)
        ax.bar(x, v, 0.6, bottom=bottom, color=c, label=lab, edgecolor="white", lw=0.6)
        bottom += v
    ax.set_xticks(x); ax.set_xticklabels([labs[l] for l in ls], rotation=25, ha="right")
    ax.set_ylabel("share of the gross harm (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=5.8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("(b) Fate of the gross harm", loc="left")
    fig.tight_layout()
    savefig(fig, os.path.join(out, "fig_sources.pdf"))
    return data


def params_abs(runs: List[Run]):
    """Freeze-and-retrain and rollback outcomes in absolute old-task accuracy points (vs. the plain
    retrain / the end point), which stays well defined when forgetting is small."""
    res = defaultdict(lambda: defaultdict(list))
    ent = []
    for r in runs:
        R = r.json("interventions", "params_task1.json")
        if not R or "end" not in R:
            continue
        ent.append(R.get("entanglement", np.nan))
        fn = R.get("freeze:none")
        for k, v in R.items():
            if k.startswith("freeze:unit:") and fn is not None:
                name, frac = k[len("freeze:unit:"):].split("@")
                res[("freeze", name, float(frac))]["gain"].append(v["old_acc"] - fn["old_acc"])
                res[("freeze", name, float(frac))]["cost"].append(fn["new_acc"] - v["new_acc"])
            elif k.startswith("rollback:unit:"):
                name, frac = k[len("rollback:unit:"):].split("@")
                res[("rollback", name, float(frac))]["gain"].append(v["old_acc"] - R["end"]["old_acc"])
                res[("rollback", name, float(frac))]["cost"].append(R["end"]["new_acc"] - v["new_acc"])
    return res, ent


PARAM_STYLE = [("ledger", "Ledger (units)", PAL["blue"]), ("fisher_delta2", "Fisher-weighted change", PAL["orange"]),
               ("abs_delta", "|parameter change|", PAL["yellow"]), ("random", "Random units", PAL["dark"])]


def fig_params(groups, out):
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(2.55 * n + 0.4, 2.35), squeeze=False)
    axes = axes[0]
    res_all = {}
    for ax, ((cname, lab), runs) in zip(axes, groups.items()):
        res, ent = params_abs(runs)
        res_all[lab] = (res, ent)
        fr = sorted({f for (k, nm, f) in res if k == "freeze"})
        if not fr:
            ax.set_visible(False)
            continue
        names = [s_ for s_ in PARAM_STYLE if ("freeze", s_[0], fr[0]) in res]
        w = 0.8 / len(PARAM_STYLE)
        for i, (nm, lb, c) in enumerate(names):
            ms = [S.mean_ci(res[("freeze", nm, f)]["gain"]) for f in fr]
            x = np.arange(len(fr)) + (i - (len(names) - 1) / 2) * w
            ax.bar(x, [100 * m["mean"] for m in ms], w * 0.9, color=c, label=lb if ax is axes[0] else None,
                   yerr=[[100 * (m["mean"] - m["lo"]) for m in ms], [100 * (m["hi"] - m["mean"]) for m in ms]],
                   error_kw=dict(lw=0.7, capsize=1.5, ecolor="#52514e"))
        ax.axhline(0, color=PAL["grey"], lw=0.7)
        ax.set_xticks(range(len(fr))); ax.set_xticklabels([f"{100 * f:g}% units" for f in fr])
        ax.set_title(f"{lab}\nentanglement $\\iota$ = {np.nanmean(ent):.2f}", loc="left", fontsize=8.3)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("forgetting prevented by\nfreezing units (acc. points)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=6.8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    savefig(fig, os.path.join(out, "fig_params.pdf"))
    return res_all


def curation(runs: List[Run]):
    """Proxy curation on held-out target runs: forgetting prevented (points) by removing the samples the
    proxies rank most harmful vs. an equal random budget. Returns {q: {'proxy': [...], 'random': [...]}}."""
    out = defaultdict(lambda: defaultdict(list))
    for r in runs:
        R = r.json("interventions", "curation_task1.json")
        if not R or "none" not in R:
            continue
        base = R["none"]["old_acc"]
        for k, v in R.items():
            if "@" in k and isinstance(v, dict) and "old_acc" in v:
                nm, q = k.split("@")
                out[float(q)][nm].append(v["old_acc"] - base)
                out[float(q)][nm + "_cost"].append(R["none"]["new_acc"] - v["new_acc"])
                out[float(q)][nm + "_seed"].append(r.seed)
    return out


def fig_curation(cur, out, q0=0.1):
    """cur: {label: curation(...)}; forgetting prevented in unseen target runs at budget q0 by removing the
    samples ranked most harmful by proxy runs vs. the same number of random samples (paired over targets)."""
    items = [(lab, c) for lab, c in cur.items() if c and q0 in c]
    if not items:
        return
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    x = np.arange(len(items))
    for i, (nm, lb, col) in enumerate([("proxy", "proxy-ranked removal", PAL["blue"]), ("random", "random removal", PAL["dark"])]):
        ms = [S.mean_ci(c[q0].get(nm, [])) for _, c in items]
        ax.bar(x + (i - 0.5) * 0.36, [100 * m["mean"] for m in ms], 0.34, color=col, label=lb,
               yerr=[[100 * (m["mean"] - m["lo"]) for m in ms], [100 * (m["hi"] - m["mean"]) for m in ms]],
               error_kw=dict(lw=0.7, capsize=1.5, ecolor="#52514e"))
        for j, (_, c) in enumerate(items):
            ax.scatter(np.full(len(c[q0][nm]), j + (i - 0.5) * 0.36) + np.linspace(-0.08, 0.08, len(c[q0][nm])),
                       100 * np.array(c[q0][nm]), s=4, color="white", edgecolor=col, lw=0.5, zorder=5)
    top = ax.get_ylim()[1]
    for j, (_, c) in enumerate(items):
        pr = S.paired(c[q0]["proxy"], c[q0]["random"])
        ax.text(j, top, f"{pr['wins']}/{pr['n']}", ha="center", va="bottom", fontsize=6.0, color="#2b2b2a")
    ax.axhline(0, color=PAL["grey"], lw=0.7)
    def _short(l):
        b, rest = l.split(": ")
        rest = {"held-out seeds": "held-out\nseeds", "MLP$\\to$CNN": "MLP$\\to$\nCNN", "MLP$\\to$wide MLP": "MLP$\\to$\nwide MLP"}.get(rest, rest)
        return f"{b}\n{rest}"
    ax.set_xticks(x); ax.set_xticklabels([_short(l) for l, _ in items], fontsize=6.3)
    ax.set_ylabel("forgetting prevented in\nunseen target runs (points)")
    ax.legend(fontsize=6.3, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    savefig(fig, os.path.join(out, "fig_curation.pdf"))


def fig_anatomy(groups, out):
    feats = [("first_correct_epoch", "epoch first learned"), ("loss_end", "loss after task"), ("loss_start", "loss before task"),
             ("margin_start", "margin before task"), ("gradnorm_start", "gradient norm"), ("prox_start", "similarity to old classes")]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    width = 0.8 / max(1, len(groups))
    cols = [PAL["blue"], PAL["orange"], PAL["aqua"], PAL["violet"]]
    res_all = {}
    for i, ((cname, lab), runs) in enumerate(groups.items()):
        res = anatomy(runs)
        res_all[lab] = res
        ms = [S.mean_ci(res.get(f, [])) for f, _ in feats]
        y = np.arange(len(feats)) + (i - (len(groups) - 1) / 2) * width
        ax.barh(y, [m["mean"] for m in ms], width * 0.9, color=cols[i], label=lab,
                xerr=[[m["mean"] - m["lo"] for m in ms], [m["hi"] - m["mean"] for m in ms]], error_kw=dict(lw=0.7, capsize=1.5))
    ax.set_yticks(np.arange(len(feats))); ax.set_yticklabels([n for _, n in feats])
    ax.axvline(0, color=PAL["grey"], lw=0.7)
    ax.set_xlabel("most harmful 5% vs. rest\n(Cohen's $d$)")
    ax.legend(fontsize=6.2, loc="best")
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, os.path.join(out, "fig_anatomy.pdf"))
    return res_all


# ----------------------------------------------------------------------------- LaTeX writers

class Tex:
    def __init__(self, d):
        self.d = d
        os.makedirs(d, exist_ok=True)
        self.macros = {}

    def macro(self, name, value):
        assert re.fullmatch(r"[A-Za-z]+", name), f"invalid macro name {name!r}"
        self.macros[name] = value

    def write(self, name, body):
        with open(os.path.join(self.d, name), "w") as f:
            f.write(body)
        print("  wrote", os.path.join(self.d, name))

    def flush(self):
        with open(os.path.join(self.d, "macros.tex"), "w") as f:
            f.write("% auto-generated by flgr.analysis.paper -- do not edit\n")
            for k, v in sorted(self.macros.items()):
                f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def ci_tex(r, pct=False, digits=1):
    if r is None or not np.isfinite(r.get("mean", np.nan)):
        return "--"
    f = 100.0 if pct else 1.0
    return f"{f * r['mean']:.{digits}f}\\,{{\\scriptsize[{f * r['lo']:.{digits}f}, {f * r['hi']:.{digits}f}]}}"


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="./runs")
    ap.add_argument("--fig", default="./paper/figures")
    ap.add_argument("--tab", default="./paper/generated")
    a = ap.parse_args()
    os.makedirs(a.fig, exist_ok=True)
    setup_style()
    tex = Tex(a.tab)
    R = load_runs(a.runs)
    print({k: len(v) for k, v in R.items()})
    core = {(k, lab): R[k] for k, lab in BENCH.items() if k in R}
    json_out = {}

    # ------------------------------------------------------------------ RQ-A concentration
    con = fig_concentration(core, a.fig)
    for lab, rows in con.items():
        key = re.sub(r"[^A-Za-z]", "", lab.split(" (")[0])
        tex.macro(f"topTen{key}", f"{100 * np.mean([d['top10'] for d in rows]):.1f}")
        tex.macro(f"topOne{key}", f"{100 * np.mean([d['top1'] for d in rows]):.1f}")
        tex.macro(f"gini{key}", f"{np.mean([d['gini'] for d in rows]):.2f}")
        tex.macro(f"fracHarmful{key}", f"{100 * np.mean([d['frac_harmful'] for d in rows]):.0f}")

    # ------------------------------------------------------------------ RQ-B removal
    methods = ["ledger", "ledger_euler", "trak", "tracin_cp10", "tracin_cp3", "static_tracin", "grad_cos", "loss",
               "feature_prox", "ledger_early20", "random"]
    qs_by = {k: sorted({float(q) for q in r.cfg["interventions"]["removal_fracs"]}) for (k, _), rr in core.items() for r in rr[:1]}
    tabs = fig_removal(core, a.fig, methods, qs_by)
    # main table: forgetting reduction (accuracy points), its share of the mean forgetting, test-loss
    # reduction, and the Holm-corrected exact sign-flip p-value against the ledger (paired over seeds)
    body = ["\\begin{tabular}{l" + "rr" * len(tabs) + "}", "\\toprule"]
    body.append("& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for lab in tabs) + " \\\\")
    body.append(" ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(tabs))))
    body.append("Score & " + " & ".join("$\\Delta F$ (pp) [95\\% CI] & $p_{\\text{Holm}}$" for _ in tabs) + " \\\\ \\midrule")
    cells = defaultdict(dict)
    removal_json = {}
    for lab, (per, seeds, F0s, qs) in tabs.items():
        q0 = 0.1 if 0.1 in qs else qs[len(qs) // 2]
        base = [d["dF"] for d in per[("ledger", q0)]]
        pv, res = {}, {}
        for m in methods + ["ledger_helpful"]:
            vals = [d["dF"] for d in per[(m, q0)]]
            if not np.isfinite(vals).any():
                continue
            res[m] = dict(ci=S.mean_ci(vals), cost=S.mean_ci([d["cost"] for d in per[(m, q0)]]),
                          dTL=S.mean_ci([d.get("dTL", np.nan) for d in per[(m, q0)]]),
                          rel=float(np.nanmean(vals) / np.nanmean(F0s)) if np.nanmean(F0s) else np.nan)
            if m != "ledger":
                pr = S.paired(base, vals)
                res[m]["paired"] = pr
                pv[m] = pr["p_perm"]
        # pre-specified families (EXPERIMENTS.md, H2): confirmatory = trajectory-free baselines;
        # trajectory-based estimators are a separate family and are additionally tested for equivalence
        fam_conf = ["random", "static_tracin", "grad_cos", "loss", "feature_prox"]
        fam_traj = ["ledger_euler", "trak", "tracin_cp10", "tracin_cp3", "ledger_early20"]
        ph = {}
        ph.update(S.holm({m: pv[m] for m in fam_conf if m in pv}))
        ph.update(S.holm({m: pv[m] for m in fam_traj if m in pv}))
        if "ledger_helpful" in res and "random" in res:           # sanity: protective removal vs random
            pr = S.paired([d["dF"] for d in per[("random", q0)]], [d["dF"] for d in per[("ledger_helpful", q0)]])
            ph["ledger_helpful"] = pr["p_perm"]
            res["ledger_helpful"]["paired_vs_random"] = pr
        for m in res:
            cells[m][lab] = (ci_tex(res[m]["ci"], pct=True, digits=1), "--" if m == "ledger" else S.fmt_p(ph.get(m, np.nan)))
        removal_json[lab] = {m: dict(dF=res[m]["ci"], rel=res[m]["rel"], cost=res[m]["cost"], dTL=res[m]["dTL"],
                                     p_holm=ph.get(m, np.nan), paired=res[m].get("paired")) for m in res}
        key = re.sub(r"[^A-Za-z]", "", lab.split(" (")[0])
        tex.macro(f"dFLedger{key}", f"{100 * res['ledger']['ci']['mean']:.1f}")
        tex.macro(f"relLedger{key}", f"{100 * res['ledger']['rel']:.0f}")
        tex.macro(f"costLedger{key}", f"{100 * res['ledger']['cost']['mean']:.1f}")
        for m, nm in [("static_tracin", "Static"), ("random", "Random"), ("ledger_helpful", "Helpful"), ("tracin_cp10", "CPten"),
                      ("trak", "Trak"), ("ledger_euler", "Euler"), ("loss", "Loss"), ("ledger_early20", "Early"),
                      ("tracin_cp3", "CPthree"), ("grad_cos", "Cos"), ("feature_prox", "Prox")]:
            if m in res:
                tex.macro(f"dF{nm}{key}", f"{100 * res[m]['ci']['mean']:.1f}")
                tex.macro(f"cost{nm}{key}", f"{100 * res[m]['cost']['mean']:.1f}")
                tex.macro(f"rel{nm}{key}", f"{100 * res[m]['rel']:.0f}")
                tex.macro(f"p{nm}{key}", S.fmt_p(ph.get(m, np.nan)))
        tex.macro(f"forget{key}", f"{100 * np.mean(F0s):.1f}")
        tex.macro(f"nseeds{key}", f"{len(seeds)}")
        if ("tracin_cp10", q0) in per:
            removal_json[lab]["tost_cp10"] = S.tost(base, [d["dF"] for d in per[("tracin_cp10", q0)]], 0.01)
    for m in methods + ["ledger_helpful"]:
        if m not in cells:
            continue
        row = [texlab(m)]
        for lab in tabs:
            c = cells[m].get(lab, ("--", "--"))
            row += [c[0], c[1]]
        pre = "\\rowcolor{blue!6}" if m == "ledger" else ""
        body.append(pre + " & ".join(row) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    tex.write("tab_removal.tex", "\n".join(body) + "\n")
    # appendix: full grid over q with new-task cost and test-loss reduction
    body = ["\\begin{tabular}{ll" + "r" * 3 + "}", "\\toprule", "Benchmark & score / $q$ & $\\Delta F$ (pp) & new-task cost (pp) & $\\Delta$ test loss \\\\ \\midrule"]
    for lab, (per, seeds, F0s, qs) in tabs.items():
        for m in methods + ["ledger_helpful"]:
            for q in qs:
                v = per.get((m, q))
                if not v or not np.isfinite([d["dF"] for d in v]).any():
                    continue
                body.append(f"{lab if (m == methods[0] and q == qs[0]) else ''} & {texlab(m)}, {100 * q:g}\\% & "
                            f"{ci_tex(S.mean_ci([d['dF'] for d in v]), pct=True)} & {ci_tex(S.mean_ci([d['cost'] for d in v]), pct=True)} & "
                            f"{ci_tex(S.mean_ci([d.get('dTL', np.nan) for d in v]), digits=3)} \\\\")
        body.append("\\midrule")
    body[-1] = "\\bottomrule"
    body.append("\\end{tabular}")
    tex.write("tab_removal_full.tex", "\n".join(body) + "\n")
    rows = []
    for lab, (per, seeds, F0s, qs) in tabs.items():
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{lab}}}}} \\\\")
        for m in methods + ["ledger_helpful"]:
            for q in qs:
                v = per.get((m, q))
                if not v or not np.isfinite([d["dF"] for d in v]).any():
                    continue
                rows.append(f"{texlab(m)}, {100 * q:g}\\% & {ci_tex(S.mean_ci([d['dF'] for d in v]), pct=True)} & "
                            f"{ci_tex(S.mean_ci([d['cost'] for d in v]), pct=True)} & {ci_tex(S.mean_ci([d.get('dTL', np.nan) for d in v]), digits=3)} \\\\")
        rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    hdr = "Score, $q$ & $\\Delta F$ (pp) & new-task cost (pp) & $\\Delta$ old-task test loss \\\\"
    lt = ["{\\footnotesize\\setlength{\\tabcolsep}{5pt}", "\\begin{longtable}{lrrr}",
          "\\caption{\\textbf{Removal and retraining, all budgets} (mean [95\\,\\% BCa CI] over ten seeds; each retraining "
          "averaged over three visiting orders).}\\label{tab:removal-full}\\\\",
          "\\toprule", hdr, "\\midrule", "\\endfirsthead",
          "\\toprule", hdr, "\\midrule", "\\endhead"] + rows + ["\\end{longtable}}"]
    tex.write("tab_removal_full_body.tex", "\n".join(lt) + "\n")
    json_out["removal"] = removal_json

    # paired comparisons per benchmark and pooled over benchmark x seed units (forest plot + appendix table)
    pr = paired_removal(tabs, 0.1)
    fig_forest(pr, list(tabs), a.fig)
    body = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
            "Comparator & scope & $\\bar d$ (pp) [95\\% CI] & $d_z$ & wins & $p_{\\text{perm}}$ & $p_{\\text{Holm}}$ & $p_t$ / $p_W$ \\\\ \\midrule"]
    pr_json = {}
    for fam_name, fam in [("trajectory-based", FAM_TRAJ), ("trajectory-free (confirmatory)", FAM_CONF)]:
        body.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{fam_name} comparators}}}} \\\\")
        for m in fam:
            if m not in pr:
                continue
            for k_, (sc, r) in enumerate(pr[m].items()):
                pt = f"{S.fmt_p(r.get('p_t', np.nan))} / {S.fmt_p(r.get('p_w', np.nan))}" if sc != "pooled" else "--"
                scl = sc if sc != "pooled" else "\\textbf{pooled}"
                body.append(f"{texlab(m) if k_ == 0 else ''} & {scl} & {ci_tex(r['diff'], pct=True)} & "
                            f"{r['dz']['dz']:.2f} & {r['wins']}/{r['n']} & {S.fmt_p(r['p_perm'])} & {S.fmt_p(r.get('p_holm', np.nan))} & {pt} \\\\")
                pr_json.setdefault(m, {})[sc] = {k: v for k, v in r.items() if k != "d"}
        body.append("\\midrule")
    body[-1] = "\\bottomrule"
    body.append("\\end{tabular}")
    tex.write("tab_paired.tex", "\n".join(body) + "\n")
    json_out["paired"] = pr_json
    for m, nm in [("static_tracin", "Static"), ("random", "Random"), ("loss", "Loss"), ("grad_cos", "Cos"), ("feature_prox", "Prox"),
                  ("tracin_cp10", "CPten"), ("trak", "Trak"), ("ledger_euler", "Euler"), ("tracin_cp3", "CPthree")]:
        if m in pr and "pooled" in pr[m]:
            r = pr[m]["pooled"]
            tex.macro(f"poolDiff{nm}", f"{100 * r['diff']['mean']:.1f}")
            tex.macro(f"poolLo{nm}", f"{100 * r['diff']['lo']:.1f}")
            tex.macro(f"poolHi{nm}", f"{100 * r['diff']['hi']:.1f}")
            tex.macro(f"poolP{nm}", S.fmt_p(r.get("p_holm", np.nan)))
            tex.macro(f"poolWins{nm}", f"{r['wins']}/{r['n']}")
            tex.macro(f"poolDz{nm}", f"{r['dz']['dz']:.2f}")
    for lab, v in removal_json.items():
        if "tost_cp10" in v:
            key = re.sub(r"[^A-Za-z]", "", lab.split(" (")[0])
            tex.macro(f"tostP{key}", S.fmt_p(v["tost_cp10"]["p"]))

    # ------------------------------------------------------------------ LDS
    L = {lab: lds(runs) for (k, lab), runs in core.items()}
    L = {lab: v for lab, v in L.items() if v}
    if L:
        mlist = [m for m in methods if any(m in v for v in L.values())]
        body = ["\\begin{tabular}{l" + "c" * len(L) + "}", "\\toprule", "Score & " + " & ".join(L) + " \\\\ \\midrule"]
        lds_json = {}
        for m in mlist:
            row = [texlab(m)]
            for lab, v in L.items():
                vals = [x for _, x in v.get(m, [])]
                ci = S.mean_ci(vals)
                lds_json.setdefault(lab, {})[m] = ci
                row.append(ci_tex(ci, digits=2))
            pre = "\\rowcolor{blue!6}" if m == "ledger" else ""
            body.append(pre + " & ".join(row) + " \\\\")
        body += ["\\bottomrule", "\\end{tabular}"]
        tex.write("tab_lds.tex", "\n".join(body) + "\n")
        json_out["lds"] = lds_json
        for lab, v in L.items():
            key = re.sub(r"[^A-Za-z]", "", lab.split(" (")[0])
            tex.macro(f"ldsLedger{key}", f"{lds_json[lab]['ledger']['mean']:.2f}")
            if "static_tracin" in lds_json[lab]:
                tex.macro(f"ldsStatic{key}", f"{lds_json[lab]['static_tracin']['mean']:.2f}")
            if "trak" in lds_json[lab]:
                tex.macro(f"ldsTrak{key}", f"{lds_json[lab]['trak']['mean']:.2f}")

    # ------------------------------------------------------------------ stability / transfer
    transfer = {}
    for k, lab in [("pmnist-domain-mlp-finetune", "PM"), ("sfmnist-task-mlp-finetune", "SF")]:
        if k not in R:
            continue
        base = [harm_vector(r) for r in R[k]]
        base = [b for b in base if b is not None]
        if not base:
            continue
        mean_h = np.mean(base, 0)
        prefix = "-".join(k.split("-")[:2])
        for tag, arch in [("wide", "wide MLP"), ("cnn", "CNN")]:
            cand = [kk for kk in R if kk.startswith(prefix) and kk.endswith("-" + tag)]
            if not cand:
                continue
            kk = cand[0]
            hs = [harm_vector(r) for r in R[kk]]
            hs = [h for h in hs if h is not None]
            if not hs:
                continue
            k1 = max(1, int(0.01 * len(mean_h)))
            top_m = set(np.argsort(-mean_h)[:k1])
            transfer[f"{lab} MLP -> {arch}"] = dict(
                signed=S.mean_ci([sst.spearmanr(mean_h, h).correlation for h in hs]),
                abs=S.mean_ci([sst.spearmanr(np.abs(mean_h), np.abs(h)).correlation for h in hs]),
                top1=S.mean_ci([len(top_m & set(np.argsort(-h)[:k1])) / k1 for h in hs]), n=len(hs))
    stab = fig_stability(core, transfer, a.fig) if core else {}
    body = ["\\begin{tabular}{llccccc}", "\\toprule",
            "Benchmark & score & Spearman (all) & top 1\\% & top 5\\% & top 10\\% & top 20\\% \\\\",
            "& & & \\multicolumn{4}{c}{overlap of the most harmful set across seeds (chance = set size)} \\\\ \\midrule"]
    stab_json = {}
    for (k, lab) in core:
        for m in ["ledger", "tracin_cp10", "static_tracin", "loss"]:
            st = stab.get((lab, m))
            if not st:
                continue
            row = [lab if m == "ledger" else "", texlab(m), ci_tex(st, digits=2)]
            row += [ci_tex(st["overlap"][q], pct=True, digits=0) for q in [0.01, 0.05, 0.1, 0.2]]
            pre = "\\rowcolor{blue!6}" if m == "ledger" else ""
            body.append(pre + " & ".join(row) + " \\\\")
            stab_json.setdefault(lab, {})[m] = dict(spearman=dict(mean=st["mean"], lo=st["lo"], hi=st["hi"]),
                                                    overlap={str(q): st["overlap"][q] for q in st["overlap"]},
                                                    abs=dict(mean=st["abs"]["mean"], lo=st["abs"]["lo"], hi=st["abs"]["hi"]),
                                                    flip={str(q): st["flip"][q] for q in st["flip"]})
            if m == "ledger":
                row = ["", "\\quad magnitude $|$harm$|$ / harmful$\\to$protective", ci_tex(st["abs"], digits=2)]
                row += [ci_tex(st["flip"][q], pct=True, digits=0) for q in [0.01, 0.05, 0.1, 0.2]]
                body.append(" & ".join(row) + " \\\\")
        cm = class_map_stability(core[(k, lab)])
        if cm:
            body.append(f"& class-level map (new $\\times$ old) & {ci_tex(cm, digits=2)} & & & & \\\\")
            stab_json.setdefault(lab, {})["class_map"] = dict(mean=cm["mean"], lo=cm["lo"], hi=cm["hi"])
            tex.macro(f"classMap{re.sub(r'[^A-Za-z]', '', lab.split(' (')[0])}", f"{cm['mean']:.2f}")
        body.append("\\midrule")
    if transfer:
        body += ["\\multicolumn{7}{l}{\\emph{Transfer: Spearman between mean MLP harm and the harm measured in another architecture}}\\\\"]
        for kk, v in transfer.items():
            arrow = kk.replace("->", "$\\to$")
            body.append(f"\\multicolumn{{2}}{{l}}{{{arrow} ({v['n']} runs; $|$harm$|$: {ci_tex(v['abs'], digits=2)})}} & {ci_tex(v['signed'], digits=2)} & {ci_tex(v['top1'], pct=True, digits=0)} & & & \\\\")
            key = re.sub(r"[^A-Za-z]", "", kk.replace("->", "to"))
            tex.macro(f"tr{key}", f"{v['signed']['mean']:.2f}")
            tex.macro(f"trAbs{key}", f"{v['abs']['mean']:.2f}")
            tex.macro(f"trTop{key}", f"{100 * v['top1']['mean']:.0f}")
    else:
        body = body[:-1]
    body += ["\\bottomrule", "\\end{tabular}"]
    tex.write("tab_stability.tex", "\n".join(body) + "\n")
    for (k, lab) in core:
        st = stab.get((lab, "ledger"))
        if st:
            key = re.sub(r"[^A-Za-z]", "", lab.split(" (")[0])
            tex.macro(f"stab{key}", f"{st['mean']:.2f}")
            tex.macro(f"overlapTen{key}", f"{100 * st['overlap'][0.1]['mean']:.0f}")
            tex.macro(f"overlapOne{key}", f"{100 * st['overlap'][0.01]['mean']:.0f}")
            tex.macro(f"enrichOne{key}", f"{st['overlap'][0.01]['mean'] / 0.01:.0f}")
            tex.macro(f"enrichTen{key}", f"{st['overlap'][0.1]['mean'] / 0.1:.1f}")
            tex.macro(f"stabAbs{key}", f"{st['abs']['mean']:.2f}")
            tex.macro(f"flipOne{key}", f"{100 * st['flip'][0.01]['mean']:.0f}")
            tex.macro(f"flipEnrichOne{key}", f"{st['flip'][0.01]['mean'] / 0.01:.0f}")
    json_out["stability"] = stab_json
    json_out["transfer"] = transfer

    # ------------------------------------------------------------------ anatomy
    anat = fig_anatomy(core, a.fig)
    json_out["anatomy"] = {lab: {f: S.mean_ci(v) for f, v in res.items()} for lab, res in anat.items()}
    for lab, res in anat.items():
        key = re.sub(r"[^A-Za-z]", "", lab.split(" (")[0])
        for f, nm in [("first_correct_epoch", "Epoch"), ("loss_end", "LossEnd"), ("loss_start", "LossStart"),
                      ("margin_start", "Margin"), ("gradnorm_start", "Grad"), ("prox_start", "Prox")]:
            if res.get(f):
                tex.macro(f"anat{nm}{key}", f"{np.nanmean(res[f]):+.2f}")

    # ------------------------------------------------------------------ surgery
    body = ["\\begin{tabular}{lccc}", "\\toprule", "Selection & target class (pp) & other classes (pp) & specificity \\\\ \\midrule"]
    sg_json = {}
    for (k, lab), runs in core.items():
        sg = surgery(runs)
        if not sg:
            continue
        body.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{lab}}}}} \\\\")
        for m, lbl in [("ledger_target", "Ledger, target column"), ("tracin_cp3_target", "TracIn-CP (3), target column"),
                       ("static_target", "Gradient conflict, target column"), ("ledger_total", "Ledger, total harm"), ("random", "Random")]:
            if m not in sg:
                continue
            t_ = S.mean_ci(sg[m]["target"]); o_ = S.mean_ci(sg[m]["others"])
            spec = np.array(sg[m]["target"]) / np.maximum(np.abs(np.array(sg[m]["others"])), 1e-3)
            body.append(f"{lbl} & {ci_tex(t_, pct=True)} & {ci_tex(o_, pct=True)} & {np.median(spec):.1f} \\\\")
            sg_json.setdefault(lab, {})[m] = dict(target=t_, others=o_)
        if "ledger_target" in sg:
            key = re.sub(r"[^A-Za-z]", "", lab.split(" (")[0])
            tex.macro(f"surgTarget{key}", f"{100 * np.mean(sg['ledger_target']['target']):.1f}")
            tex.macro(f"surgOthers{key}", f"{100 * np.mean(sg['ledger_target']['others']):.1f}")
    body += ["\\bottomrule", "\\end{tabular}"]
    tex.write("tab_surgery.tex", "\n".join(body) + "\n")
    json_out["surgery"] = sg_json

    # ------------------------------------------------------------------ interference maps
    from ..data import CIFAR10_NAMES, FASHION_NAMES
    def names_by(run):
        if run.cfg["benchmark"] == "agnews_dbpedia":
            from ..text import AG_LABELS, DBP_KEEP
            return AG_LABELS + list(DBP_KEEP.values())
        return FASHION_NAMES if run.cfg["benchmark"] == "sfmnist" else CIFAR10_NAMES if run.cfg["benchmark"] == "scifar10" else [str(i) for i in range(10)]
    imaps = fig_imap({k: v for k, v in core.items() if k[0] != "pmnist-domain-mlp-finetune"}, a.fig, names_by)
    # H6: does the class-level harm map follow feature similarity (new-class centroid vs. old class, at the
    # start of task B)? Per-seed Spearman over the C_new x G cells; exact sign-flip test over seeds.
    sim_json = {}
    for (k, lab), runs in core.items():
        rs = [sst.spearmanr(r.scores(1)["imap"].numpy().ravel(), r.scores(1)["feat_sim"].numpy().ravel()).correlation
              for r in runs if r.scores(1) is not None and "feat_sim" in r.scores(1)]
        if len(rs) >= 3:
            ci = S.mean_ci(rs); pp = S.signflip_test(rs)
            sim_json[lab] = dict(ci=ci, p=pp)
            key = re.sub(r"[^A-Za-z]", "", lab.split(" (")[0])
            tex.macro(f"simMap{key}", f"{ci['mean']:+.2f}")
            tex.macro(f"simMapLo{key}", f"{ci['lo']:+.2f}")
            tex.macro(f"simMapHi{key}", f"{ci['hi']:+.2f}")
            tex.macro(f"simMapP{key}", S.fmt_p(pp))
    json_out["map_vs_similarity"] = sim_json

    # ------------------------------------------------------------------ numerics
    rules = {r: R.get(f"pmnist-domain-mlp-finetune-rule-{r}", []) for r in ["euler", "trapezoid", "simpson", "adaptive"]}
    lrs = {float(re.findall(r"lr([0-9.]+)$", k)[0]): v for k, v in R.items() if re.match(r"pmnist-domain-mlp-finetune-lr[0-9.]+$", k)}
    if "pmnist-domain-mlp-finetune-rule-adaptive" in R:
        lrs.setdefault(0.1, R["pmnist-domain-mlp-finetune-rule-adaptive"])
    fig_numerics(core, rules, lrs, a.fig)
    # completeness of the re-tracked Split CIFAR-10 CNN with 16 quadrature sub-intervals (GPU)
    n16 = R.get("scifar10-task-cnn-none-finetune-n16", [])
    if n16:
        rows16 = completeness_rows(n16)
        tex.macro("epsCNNsixteen", f"{100 * np.mean([r['eps'] for r in rows16]):.2f}")
        tex.macro("epsCNNsixteenMax", f"{100 * np.max([r['eps'] for r in rows16]):.2f}")
        tex.macro("epsTVCNNsixteen", f"{100 * np.mean([r['eps_tv'] for r in rows16]):.3f}")
        tex.macro("evalsCNNsixteen", f"{np.mean([r['evals'] for r in rows16]) + 1:.1f}")
        tex.macro("nCNNsixteen", f"{len(rows16)}")
        json_out["c10_cnn_n16"] = dict(eps=S.mean_ci([r["eps"] for r in rows16]), eps_tv=S.mean_ci([r["eps_tv"] for r in rows16]))
    body = ["\\begin{tabular}{lccccc}", "\\toprule",
            "Benchmark & $\\varepsilon$ (\\%) & $\\varepsilon_{\\mathrm{TV}}$ (\\%) & Euler $\\varepsilon$ (\\%) & start / realised & end / realised \\\\ \\midrule"]
    num_json = {}
    for (k, lab), runs in core.items():
        rows = completeness_rows(runs)
        e = S.mean_ci([r["eps"] for r in rows]); etv = S.mean_ci([r["eps_tv"] for r in rows])
        eu = S.mean_ci([r["euler_err"] for r in rows]); st = S.mean_ci([r["start"] for r in rows]); en = S.mean_ci([r["end"] for r in rows])
        body.append(f"{lab} & {ci_tex(e, pct=True, digits=2)} & {ci_tex(etv, pct=True, digits=3)} & {ci_tex(eu, pct=True, digits=0)} & {ci_tex(st, digits=2)} & {ci_tex(en, digits=2)} \\\\")
        num_json[lab] = dict(eps=e, eps_tv=etv, euler=eu, start=st, end=en)
        key = re.sub(r"[^A-Za-z]", "", lab.split(" (")[0])
        tex.macro(f"eps{key}", f"{100 * e['mean']:.2f}")
        tex.macro(f"eulerErr{key}", f"{100 * eu['mean']:.0f}")
        tex.macro(f"ratioStart{key}", f"{st['mean']:.2f}")
        tex.macro(f"ratioEnd{key}", f"{en['mean']:.2f}")
        sr = [r["stats_ratio"] for r in rows]
        if any(abs(r["stats_abs"]) > 1e-9 for r in rows):
            tex.macro(f"statsRatio{key}", f"{np.nanmean(sr):+.1f}")
            tex.macro(f"statsRatioLo{key}", f"{np.nanmin(sr):+.1f}")
            tex.macro(f"statsRatioHi{key}", f"{np.nanmax(sr):+.1f}")
            tex.macro(f"pathRatio{key}", f"{np.nanmean([r['path_ratio'] for r in rows]):+.1f}")
            num_json[lab]["stats_ratio"] = S.mean_ci(sr)
    for rname, runs in rules.items():
        rows = completeness_rows(runs)
        if rows:
            tex.macro(f"epsRule{rname.capitalize()}", f"{100 * np.mean([r['eps'] for r in rows]):.2f}")
            tex.macro(f"evalsRule{rname.capitalize()}", f"{np.mean([r['evals'] for r in rows]) + 1:.1f}")
    body += ["\\bottomrule", "\\end{tabular}"]
    tex.write("tab_numerics.tex", "\n".join(body) + "\n")
    json_out["numerics"] = num_json

    # ------------------------------------------------------------------ dose response
    dose = {float(re.findall(r"pf([0-9.]+)$", k)[0]): v for k, v in R.items() if re.search(r"-pf[0-9.]+$", k)}
    if dose:
        d = fig_dose(dose, a.fig)
        if d:
            body = ["\\begin{tabular}{lccc}", "\\toprule", "Quantity & slope per unit of permuted fraction & 95\\% CI & $p$ \\\\ \\midrule"]
            names = dict(forg="forgetting (pp)", top10="top-10\\% share of harm", head="output-layer share of ledger mass",
                         first="first-layer share of ledger mass", ent="entanglement $\\iota$",
                         rho_l="forgetting prevented, ledger removal (pp)", rho_s="forgetting prevented, gradient-conflict removal (pp)",
                         rho_r="forgetting prevented, random removal (pp)")
            for kk, tr_ in d["trends"].items():
                body.append(f"{names[kk]} & {tr_['slope']:+.3f} & [{tr_['lo']:+.3f}, {tr_['hi']:+.3f}] & {S.fmt_p(tr_['p'])} \\\\")
            body += ["\\bottomrule", "\\end{tabular}"]
            tex.write("tab_dose.tex", "\n".join(body) + "\n")
            json_out["dose"] = d["trends"]
            for kk, tr_ in d["trends"].items():
                nm = dict(forg="Forg", top10="Topten", head="Head", first="First", ent="Ent", rho_l="L", rho_s="S", rho_r="R")[kk]
                tex.macro(f"doseSlope{nm}", f"{tr_['slope']:+.2f}")
                tex.macro(f"doseP{nm}", S.fmt_p(tr_["p"]))
            rec = d["rec"]
            for pf, nm in [(min(d["levels"]), "Lo"), (max(d["levels"]), "Hi")]:
                sel = rec["pf"] == pf
                tex.macro(f"doseForg{nm}", f"{np.nanmean(rec['forg'][sel]):.1f}")
                tex.macro(f"doseTop{nm}", f"{100 * np.nanmean(rec['top10'][sel]):.0f}")
                tex.macro(f"doseEnt{nm}", f"{np.nanmean(rec['ent'][sel]):.2f}")
                tex.macro(f"doseL{nm}", f"{np.nanmean(rec['rho_l'][sel]):.1f}")
                tex.macro(f"doseS{nm}", f"{np.nanmean(rec['rho_s'][sel]):.1f}")
                tex.macro(f"doseR{nm}", f"{np.nanmean(rec['rho_r'][sel]):.1f}")
            # excess of ledger removal over random / over gradient conflict, pooled over all levels
            ex_r = rec["rho_l"] - rec["rho_r"]; ex_s = rec["rho_l"] - rec["rho_s"]
            for arr, nm in [(ex_r, "Random"), (ex_s, "Static")]:
                ci = S.mean_ci(arr)
                tex.macro(f"doseEx{nm}", f"{ci['mean']:.1f}")
                tex.macro(f"doseExLo{nm}", f"{ci['lo']:.1f}")
                tex.macro(f"doseExHi{nm}", f"{ci['hi']:.1f}")
                tex.macro(f"doseExP{nm}", S.fmt_p(S.signflip_test(arr)))
                tex.macro(f"doseExN{nm}", f"{int(np.isfinite(arr).sum())}")

    # ------------------------------------------------------------------ learners
    learners = {}
    for l in ["finetune", "er", "derpp", "ewc", "agem"]:
        k = f"smnist-class-mlp-{l}"
        if k in R:
            learners[l] = R[k]
    src = fig_sources(learners, a.fig)
    if src:
        body = ["\\begin{tabular}{lcccccc}", "\\toprule",
                "Learner & new data (harm) & new data (help) & replay & regulariser & net $\\Delta\\mathcal{L}$ & final acc.\\ (\\%) \\\\ \\midrule"]
        for l, rows in src.items():
            f = lambda key: ci_tex(S.mean_ci([d.get(key, 0.0) for d in rows]), digits=1)
            rep = S.mean_ci([d.get("mem_pos", 0) + d.get("mem_neg", 0) for d in rows])
            body.append(f"{dict(finetune='Fine-tune', er='ER', derpp='DER++', ewc='Online EWC', agem='A-GEM')[l]} & {f('new_pos')} & {f('new_neg')} & "
                        f"{ci_tex(rep, digits=1) if l in ('er', 'derpp') else '--'} & {f('reg') if l == 'ewc' else '--'} & {f('net')} & {ci_tex(S.mean_ci([d['acc'] for d in rows]), pct=True)} \\\\")
        body += ["\\bottomrule", "\\end{tabular}"]
        tex.write("tab_sources.tex", "\n".join(body) + "\n")
        json_out["sources"] = {l: {k: S.mean_ci([d.get(k, 0.0) for d in rows]) for k in ["new_pos", "new_neg", "mem_pos", "mem_neg", "reg", "net", "acc"]} for l, rows in src.items()}
        for l, rows in src.items():
            nm = dict(finetune="FT", er="ER", derpp="DER", ewc="EWC", agem="AGEM")[l]
            tex.macro(f"srcHarm{nm}", f"{np.mean([d.get('new_pos', 0) for d in rows]):.0f}")
            tex.macro(f"srcHelp{nm}", f"{-np.mean([d.get('new_neg', 0) for d in rows]):.0f}")
            tex.macro(f"srcRep{nm}", f"{-np.mean([d.get('mem_pos', 0) + d.get('mem_neg', 0) for d in rows]):.0f}")
            tex.macro(f"srcReg{nm}", f"{-np.mean([d.get('reg', 0) for d in rows]):.0f}")
            tex.macro(f"srcNet{nm}", f"{np.mean([d['net'] for d in rows]):.1f}")
            tex.macro(f"srcAcc{nm}", f"{100 * np.mean([d['acc'] for d in rows]):.1f}")
            tex.macro(f"srcSelf{nm}", f"{100 * np.mean([-d.get('new_neg', 0) / max(d.get('new_pos', 0), 1e-9) for d in rows]):.0f}")

    # ------------------------------------------------------------------ parameters
    par = fig_params({k: v for k, v in core.items()}, a.fig)
    json_out["params"] = {lab: {"ent": S.mean_ci(ent), **{f"{k[0]}:{k[1]}@{k[2]}": dict(gain=S.mean_ci(v["gain"]), cost=S.mean_ci(v["cost"])) for k, v in res.items()}}
                          for lab, (res, ent) in par.items()}
    body = ["\\begin{tabular}{llrrrr}", "\\toprule",
            "Benchmark & unit score & \\multicolumn{2}{c}{freeze and retrain (primary)} & \\multicolumn{2}{c}{roll back (negative control)} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6} & & old-task gain (pp) & new-task cost (pp) & old-task gain (pp) & new-task cost (pp) \\\\ \\midrule"]
    for lab, (res, ent) in par.items():
        key = re.sub(r"[^A-Za-z]", "", lab.split(" (")[0])
        tex.macro(f"ent{key}", f"{np.nanmean(ent):.2f}")
        fr = sorted({f for (k, nm, f) in res if k == "freeze"})
        first = True
        for f in fr:
            for nm, lb, _ in PARAM_STYLE:
                if ("freeze", nm, f) not in res:
                    continue
                fz, rb = res[("freeze", nm, f)], res.get(("rollback", nm, f))
                rbs = (ci_tex(S.mean_ci(rb["gain"]), pct=True), ci_tex(S.mean_ci(rb["cost"]), pct=True)) if rb else ("--", "--")
                head = (lab + " ($\\iota$=" + f"{np.nanmean(ent):.2f})") if first else ""
                body.append(f"{head} & {lb}, {100 * f:g}\\% & {ci_tex(S.mean_ci(fz['gain']), pct=True)} & "
                            f"{ci_tex(S.mean_ci(fz['cost']), pct=True)} & {rbs[0]} & {rbs[1]} \\\\")
                first = False
        if fr:
            g_led = S.mean_ci(res[("freeze", "ledger", fr[-1])]["gain"]) if ("freeze", "ledger", fr[-1]) in res else None
            if g_led:
                tex.macro(f"freezeLedger{key}", f"{100 * g_led['mean']:.1f}")
            if ("freeze", "random", fr[-1]) in res:
                tex.macro(f"freezeRandom{key}", f"{100 * np.mean(res[('freeze', 'random', fr[-1])]['gain']):.1f}")
            if ("freeze", "fisher_delta2", fr[-1]) in res:
                tex.macro(f"freezeFisher{key}", f"{100 * np.mean(res[('freeze', 'fisher_delta2', fr[-1])]['gain']):.1f}")
        body.append("\\midrule")
    body[-1] = "\\bottomrule"
    body.append("\\end{tabular}")
    tex.write("tab_params.tex", "\n".join(body) + "\n")

    # ------------------------------------------------------------------ proxy curation
    cur = {}
    for k, lab in [("pmnist-domain-mlp-finetune", "PM: held-out seeds"), ("sfmnist-task-mlp-finetune", "SF: held-out seeds"),
                   ("scifar10-task-cnn-none-finetune", "C10: held-out seeds")]:
        if k in R:
            cur[lab] = curation(R[k])
    for k in sorted(R):
        for pre, bl in [("pmnist-domain", "PM"), ("sfmnist-task", "SF")]:
            for tag, arch in [("wide", "wide MLP"), ("cnn", "CNN")]:
                if k.startswith(pre) and k.endswith("-" + tag):
                    cur[f"{bl}: MLP$\\to${arch}"] = curation(R[k])
    cur = {l: c for l, c in cur.items() if c}
    if cur:
        fig_curation(cur, a.fig)
        body = ["\\begin{tabular}{llrrrr}", "\\toprule",
                "Targets & $q$ & proxy-ranked (pp) & random (pp) & proxy $-$ random (pp) & $p_{\\text{perm}}$ \\\\ \\midrule"]
        cj = {}
        for lab, c in cur.items():
            for i, q in enumerate(sorted(c)):
                pr_ = S.paired(c[q].get("proxy", []), c[q].get("random", []))
                body.append(f"{lab if i == 0 else ''} ({len(c[q].get('proxy', []))} runs) & {100 * q:g}\\% & {ci_tex(S.mean_ci(c[q].get('proxy', [])), pct=True)} & "
                            f"{ci_tex(S.mean_ci(c[q].get('random', [])), pct=True)} & {ci_tex(pr_['diff'], pct=True)} & {S.fmt_p(pr_['p_perm'])} \\\\")
                cj.setdefault(lab, {})[str(q)] = dict(proxy=S.mean_ci(c[q].get("proxy", [])), random=S.mean_ci(c[q].get("random", [])),
                                                      diff=pr_["diff"], p=pr_["p_perm"])
            body.append("\\midrule")
            key = re.sub(r"[^A-Za-z]", "", lab)
            if 0.1 in c:
                tex.macro(f"cur{key}", f"{100 * np.mean(c[0.1]['proxy']):.1f}")
                tex.macro(f"curRand{key}", f"{100 * np.mean(c[0.1]['random']):.1f}")
                tex.macro(f"curP{key}", S.fmt_p(S.paired(c[0.1]['proxy'], c[0.1]['random'])['p_perm']))
        dd = np.concatenate([np.array(c[0.1]["proxy"]) - np.array(c[0.1]["random"]) for c in cur.values() if 0.1 in c])
        if len(dd):
            pci = S.mean_ci(dd)
            body.append(f"\\textbf{{pooled}} ({len(dd)} runs) & 10\\% & & & {ci_tex(pci, pct=True)} & {S.fmt_p(S.signflip_test(dd))} \\\\")
            body.append("\\midrule")
            tex.macro("curPoolDiff", f"{100 * pci['mean']:.1f}")
            tex.macro("curPoolLo", f"{100 * pci['lo']:.1f}")
            tex.macro("curPoolHi", f"{100 * pci['hi']:.1f}")
            tex.macro("curPoolP", S.fmt_p(S.signflip_test(dd)))
            tex.macro("curPoolN", f"{len(dd)}")
            tex.macro("curPoolWins", f"{int((dd > 0).sum())}")
            json_out["curation_pooled"] = dict(diff=pci, p=S.signflip_test(dd), n=len(dd))
        body[-1] = "\\bottomrule"
        body.append("\\end{tabular}")
        tex.write("tab_curation.tex", "\n".join(body) + "\n")
        json_out["curation"] = cj

    # ------------------------------------------------------------------ Adam
    ad = R.get("pmnist-domain-mlp-finetune-adam", [])
    if ad:
        rows = completeness_rows(ad)
        ms = ["ledger", "tracin_cp3", "static_tracin", "random", "ledger_helpful"]
        per, seeds, F0s = removal_table(ad, ms, [0.1])
        base = [d["dF"] for d in per[("ledger", 0.1)]]
        pv = {m: S.paired(base, [d["dF"] for d in per[(m, 0.1)]])["p_perm"] for m in ["tracin_cp3", "static_tracin", "random"]}
        ph = S.holm(pv)
        body = ["\\begin{tabular}{lrrr}", "\\toprule", "Score & $\\Delta F$ (pp) [95\\% CI] & wins of ledger & $p_{\\text{Holm}}$ \\\\ \\midrule"]
        adam_json = dict(eps=S.mean_ci([r["eps"] for r in rows]), eps_tv=S.mean_ci([r["eps_tv"] for r in rows]), F0=S.mean_ci(F0s))
        for m in ms:
            v = [d["dF"] for d in per[(m, 0.1)]]
            ci = S.mean_ci(v)
            wins = int((np.array(base) > np.array(v)).sum()) if m != "ledger" else None
            pre = "\\rowcolor{blue!6}" if m == "ledger" else ""
            body.append(pre + f"{texlab(m)} & {ci_tex(ci, pct=True)} & {'--' if wins is None else f'{wins}/{len(v)}'} & {S.fmt_p(ph.get(m, np.nan))} \\\\")
            nm = dict(ledger="Ledger", tracin_cp3="CPthree", static_tracin="Static", random="Random", ledger_helpful="Helpful")[m]
            tex.macro(f"dFAdam{nm}", f"{100 * ci['mean']:.1f}")
            if m in ph:
                tex.macro(f"pAdam{nm}", S.fmt_p(ph[m]))
            adam_json[m] = dict(dF=ci, p_holm=ph.get(m, np.nan))
        body += ["\\bottomrule", "\\end{tabular}"]
        tex.write("tab_adam.tex", "\n".join(body) + "\n")
        tex.macro("epsAdam", f"{100 * adam_json['eps']['mean']:.2f}")
        tex.macro("epsTVAdam", f"{100 * adam_json['eps_tv']['mean']:.3f}")
        tex.macro("forgetAdam", f"{100 * np.mean(F0s):.1f}")
        tex.macro("nAdam", f"{len(seeds)}")
        json_out["adam"] = adam_json

    # ------------------------------------------------------------------ cost
    body = ["\\begin{tabular}{lcccc}", "\\toprule", "Benchmark & plain step (ms) & tracked step (ms) & overhead & probe evals / step \\\\ \\midrule"]
    cost_json = {}
    for (k, lab), runs in core.items():
        tp, tt, ev = [], [], []
        for r in runs:
            m = r.metrics()
            rr = removal_rho(r)
            led = r.ledger(1)
            if len(m) > 1 and rr and led is not None and np.isfinite(rr["t_plain"]):
                tt.append(1000 * m[1]["time_s"] / led["steps"])
                tp.append(1000 * rr["t_plain"] / rr["steps"])
                ev.append(float(led["n_evals"]) / led["steps"])
        if tp:
            body.append(f"{lab} & {np.mean(tp):.1f} & {np.mean(tt):.1f} & {np.mean(tt) / np.mean(tp):.1f}$\\times$ & {np.mean(ev):.2f} \\\\")
            cost_json[lab] = dict(plain_ms=np.mean(tp), tracked_ms=np.mean(tt), evals=np.mean(ev))
            key = re.sub(r"[^A-Za-z]", "", lab.split(" (")[0])
            tex.macro(f"overhead{key}", f"{np.mean(tt) / np.mean(tp):.0f}")
    body += ["\\bottomrule", "\\end{tabular}"]
    tex.write("tab_cost.tex", "\n".join(body) + "\n")
    json_out["cost"] = cost_json

    tex.flush()
    with open(os.path.join(a.tab, "results.json"), "w") as f:
        json.dump(json_out, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print("done")


if __name__ == "__main__":
    main()
