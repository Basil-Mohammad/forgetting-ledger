"""Aggregate all runs into the paper's tables (Markdown + LaTeX + JSON) and figures.

    python -m flgr.analysis.aggregate --runs ./runs --out ./results

A "configuration" is a run name without its seed suffix; statistics are over seeds (mean, 95% t-CI),
comparisons are paired over seeds with Holm-Bonferroni correction.
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
from scipy import stats

from ..utils import hash_uniform

# ----------------------------------------------------------------------------- helpers


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def fmt(x, pct=False, digits=3):
    m, h = ci95(x)
    if not np.isfinite(m):
        return "–"
    if pct:
        return f"{100 * m:.1f} ± {100 * h:.1f}" if np.isfinite(h) else f"{100 * m:.1f}"
    return f"{m:.{digits}f} ± {h:.{digits}f}" if np.isfinite(h) else f"{m:.{digits}f}"


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return dict(diff=float("nan"), p_t=float("nan"), p_w=float("nan"), dz=float("nan"))
    d = a - b
    pt = stats.ttest_rel(a, b).pvalue
    try:
        pw = stats.wilcoxon(a, b).pvalue
    except ValueError:
        pw = float("nan")
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("inf")
    return dict(diff=float(d.mean()), p_t=float(pt), p_w=float(pw), dz=float(dz))


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def config_of(run_dir: str) -> str:
    return re.sub(r"-s\d+$", "", os.path.basename(run_dir.rstrip("/")))


def forgetting(res: dict, ref: List[float], t: int) -> float:
    return float(np.mean(np.array(ref[:t]) - np.array(res["task_acc"][:t])))


# ----------------------------------------------------------------------------- per-run extraction


def extract_run(run: str) -> dict:
    out = {"run": run}
    cfg = load_json(os.path.join(run, "config.json"))
    out["cfg"] = cfg
    metrics = [json.loads(l) for l in open(os.path.join(run, "metrics.jsonl"))] if os.path.exists(os.path.join(run, "metrics.jsonl")) else []
    out["metrics"] = metrics
    evals = sorted(glob.glob(os.path.join(run, "eval", "task*.json")), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
    rows = [load_json(p)["task_acc"] for p in evals]
    if rows:
        K = len(rows)
        A = np.full((K, K), np.nan)
        for i, r in enumerate(rows):
            A[i, :len(r)] = r
        out["acc_matrix"] = A
        out["avg_acc"] = float(np.nanmean(A[-1]))
        out["avg_forgetting"] = float(np.mean([np.nanmax(A[:-1, j]) - A[-1, j] for j in range(K - 1)])) if K > 1 else 0.0
    # ledger summaries
    out["ledger"] = {}
    for f in glob.glob(os.path.join(run, "ledger", "task*.pt")):
        t = int(re.findall(r"task(\d+)", f)[0])
        d = torch.load(f, map_location="cpu", weights_only=False)
        true, pred = d["true_dL"].numpy(), (d["path_g"] + d["stats"]).numpy()
        src = dict(data=float(d["data"].sum()), mem=float(d["mem"].sum()) if d["mem"] is not None else 0.0,
                   reg=float(d["reg"].sum()), stats=float(d["stats"].sum()))
        data = d["data"].sum(1).numpy()
        pos = np.clip(data, 0, None)
        srt = np.sort(pos)[::-1]
        top10 = srt[: max(1, int(0.1 * len(srt)))].sum() / max(srt.sum(), 1e-12)
        out["ledger"][t] = dict(true=true, pred=pred, err=float(np.abs(pred - true).sum() / max(np.abs(true).sum(), 1e-12)),
                                err_tv=float(np.abs(pred - true).sum() / max(float(d["path_var"].sum()), 1e-12)) if "path_var" in d else float("nan"),
                                evals=float(d.get("n_evals", 0)) / max(1, d["steps"]), sources=src, top10_share=float(top10),
                                layer=d["layer"].sum(1).numpy(), layer_names=d["layer_names"],
                                ent=float(stats.spearmanr(d["unit"].sum(1).numpy(), d["learn_unit"].numpy()).correlation),
                                groups=d.get("groups"))
    return out


# ----------------------------------------------------------------------------- experiment tables


def removal_table(runs: List[str]):
    """RQ2 / RQ7: forgetting reduction rho_q and new-task cost, per method and q."""
    per = defaultdict(lambda: defaultdict(list))                     # (method,q) -> metric -> list over (run,task)
    keys = []
    for run in runs:
        for f in glob.glob(os.path.join(run, "interventions", "removal_task*.json")):
            t = int(re.findall(r"task(\d+)", f)[0])
            R = load_json(f)
            if "none" not in R:
                continue
            F0 = forgetting(R["none"], R["ref_acc"], t)
            for k, v in R.items():
                if "@" not in k:
                    continue
                m, q = k.split("@")
                Fk = forgetting(v, R["ref_acc"], t)
                per[(m, float(q))]["rho"].append((F0 - Fk) / F0 if abs(F0) > 1e-9 else np.nan)
                per[(m, float(q))]["cost"].append(R["none"]["new_acc"] - v["new_acc"])
                per[(m, float(q))]["unit"].append(f"{run}:{t}")
                if (m, float(q)) not in keys:
                    keys.append((m, float(q)))
    return per, keys


def lds_table(runs: List[str]):
    """RQ2: linear datamodeling score per attribution method."""
    res = defaultdict(list)
    for run in runs:
        cfg = load_json(os.path.join(run, "config.json"))
        for f in glob.glob(os.path.join(run, "interventions", "lds_task*.json")):
            t = int(re.findall(r"task(\d+)", f)[0])
            R = load_json(f)
            S = torch.load(os.path.join(run, "scores", f"task{t}.pt"), map_location="cpu", weights_only=False)["scores"]
            N = S["ledger"].shape[0]
            subs = sorted([k for k in R if k.startswith("subset")], key=lambda k: int(k[6:]))
            if len(subs) < 4:
                continue
            masks = torch.stack([(hash_uniform(int(cfg["seed"]), 88, t, int(k[6:]), torch.arange(N)).squeeze(1) < cfg["interventions"]["lds_frac"]) for k in subs]).float()
            actual = np.array([R[k]["probe_L"] for k in subs])                 # [M, G]
            for m, s in S.items():
                pred = (masks @ s.float()).numpy()                                 # [M, G]
                rhos = [stats.spearmanr(pred[:, g], actual[:, g]).correlation for g in range(actual.shape[1])]
                res[m].append(float(np.nanmean(rhos)))
    return res


def surgery_table(runs: List[str]):
    res = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for f in glob.glob(os.path.join(run, "interventions", "surgery_task*.json")):
            R = load_json(f)
            base = np.array(R["none"]["group_acc"])
            for g in R.get("targets", []):
                for k, v in R.items():
                    if not k.startswith(f"g{g}:"):
                        continue
                    m = k.split(":", 1)[1]
                    gain = np.array(v["group_acc"]) - base
                    others = np.delete(gain, g)
                    res[m]["target"].append(gain[g])
                    res[m]["others"].append(others.mean())
                    res[m]["spec"].append(gain[g] / max(np.abs(others).mean(), 1e-3))
    return res


def params_table(runs: List[str]):
    res = defaultdict(lambda: defaultdict(list))
    ent, layer_rank = [], []
    for run in runs:
        for f in glob.glob(os.path.join(run, "interventions", "params_task*.json")):
            t = int(re.findall(r"task(\d+)", f)[0])
            R = load_json(f)
            if "end" not in R:
                continue
            ref_old = float(np.mean(R["ref_acc"]))
            end_old, end_new = R["end"]["old_acc"], R["end"]["new_acc"]
            ent.append(R["entanglement"])
            if "layer_rollback" in R:
                names = list(R["layer_share"].keys())
                share = np.array([R["layer_share"][n] for n in names])
                gain = np.array([R["layer_rollback"][n] - end_old for n in names])
                layer_rank.append(stats.spearmanr(share, gain).correlation)
            fn = R.get("freeze:none")
            for k, v in R.items():
                if not (k.startswith("rollback:") or k.startswith("freeze:unit:")):
                    continue
                kind, rest = k.rsplit(":", 1)[0], k.rsplit(":", 1)[1]
                name, frac = rest.split("@")
                if kind.startswith("freeze"):
                    if fn is None:
                        continue
                    base_old, base_new = fn["old_acc"], fn["new_acc"]
                else:
                    base_old, base_new = end_old, end_new
                denom = ref_old - base_old
                res[(kind, name, float(frac))]["recovered"].append((v["old_acc"] - base_old) / denom if abs(denom) > 1e-9 else np.nan)
                res[(kind, name, float(frac))]["cost"].append(base_new - v["new_acc"])
    return res, ent, layer_rank


# ----------------------------------------------------------------------------- report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="./runs")
    ap.add_argument("--out", default="./results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = sorted(d for d in glob.glob(os.path.join(a.runs, "*")) if os.path.exists(os.path.join(d, "config.json")))
    by_cfg = defaultdict(list)
    for r in runs:
        by_cfg[config_of(r)].append(r)
    md = ["# Results (auto-generated)\n"]
    summary = {}
    for cname, rr in sorted(by_cfg.items()):
        ex = [extract_run(r) for r in rr]
        md.append(f"\n## {cname}  (n = {len(rr)} seeds)\n")
        md.append(f"Average accuracy {fmt([e.get('avg_acc', np.nan) for e in ex], pct=True)} %, "
                  f"average forgetting {fmt([e.get('avg_forgetting', np.nan) for e in ex], pct=True)} %\n")
        # ---- RQ1
        md.append("\n### RQ1 completeness\n\n| task | ε (Σ|ΔL|) | ε (TV) | probe evals/step | top-10% share of harm | entanglement ι |\n|---|---|---|---|---|---|")
        tasks = sorted({t for e in ex for t in e["ledger"]})
        for t in tasks:
            L = [e["ledger"][t] for e in ex if t in e["ledger"]]
            md.append(f"| {t} | {fmt([l['err'] for l in L], pct=True)} % | {fmt([l['err_tv'] for l in L], pct=True)} % | "
                      f"{fmt([l['evals'] for l in L], digits=2)} | {fmt([l['top10_share'] for l in L], pct=True)} % | {fmt([l['ent'] for l in L], digits=2)} |")
        # single-point
        sp = defaultdict(list)
        for r in rr:
            for f in glob.glob(os.path.join(r, "scores", "task*.pt")):
                s = torch.load(f, map_location="cpu", weights_only=False)["single"]
                tot = np.sum(s["true"])
                if abs(tot) > 1e-9:
                    sp["start"].append(np.sum(s["start"]) / tot)
                    sp["end"].append(np.sum(s["end"]) / tot)
                    sp["ledger"].append(np.sum(s["ledger"]) / tot)
        if sp:
            md.append(f"\nSingle-point first-order explanation / realised ΔL: at θ_start {fmt(sp['start'], digits=2)}, "
                      f"at θ_end {fmt(sp['end'], digits=2)}, ledger {fmt(sp['ledger'], digits=3)}\n")
        # ---- sources (RQ5)
        md.append("\n### RQ5 sources of the realised change (sum over tasks)\n")
        src = defaultdict(list)
        for e in ex:
            tot = defaultdict(float)
            for t, l in e["ledger"].items():
                for k, v in l["sources"].items():
                    tot[k] += v
            for k, v in tot.items():
                src[k].append(v)
        md.append("| " + " | ".join(src) + " |\n|" + "---|" * len(src) + "\n| " + " | ".join(fmt(v) for v in src.values()) + " |")
        # ---- RQ2 removal
        per, keys = removal_table(rr)
        if keys:
            qs = sorted({q for _, q in keys})
            methods = [m for m in dict.fromkeys(k[0] for k in keys)]
            md.append("\n### RQ2 removal-and-retrain: forgetting reduction ρ (%) / new-task cost (pp)\n")
            md.append("| method | " + " | ".join(f"q={100 * q:g}%" for q in qs) + " |\n|---|" + "---|" * len(qs))
            for m in methods:
                cells = []
                for q in qs:
                    d = per.get((m, q))
                    cells.append(f"{fmt(d['rho'], pct=True)} / {fmt(d['cost'], pct=True)}" if d else "–")
                md.append(f"| {m} | " + " | ".join(cells) + " |")
            q0 = 0.1 if (("ledger", 0.1) in per) else qs[-1]
            pv, rows = {}, []
            for m in methods:
                if m == "ledger" or (m, q0) not in per:
                    continue
                pr = paired(per[("ledger", q0)]["rho"], per[(m, q0)]["rho"])
                pv[m] = pr["p_t"]
                rows.append((m, pr))
            ph = holm({k: v for k, v in pv.items() if np.isfinite(v)})
            md.append(f"\nPaired comparison at q = {100 * q0:g}% (ledger − method, over seeds × tasks; Holm-corrected t-test):\n")
            md.append("| vs | Δρ | d_z | p (Holm) | p (Wilcoxon) |\n|---|---|---|---|---|")
            for m, pr in rows:
                md.append(f"| {m} | {100 * pr['diff']:+.1f} | {pr['dz']:.2f} | {ph.get(m, float('nan')):.4f} | {pr['p_w']:.4f} |")
            summary[cname + ":removal"] = {f"{m}@{q}": ci95(v["rho"]) for (m, q), v in per.items()}
        # ---- LDS
        lds = lds_table(rr)
        if lds:
            md.append("\n### RQ2 linear datamodeling score (Spearman)\n\n| method | LDS |\n|---|---|")
            for m, v in sorted(lds.items(), key=lambda kv: -np.nanmean(kv[1])):
                md.append(f"| {m} | {fmt(v, digits=3)} |")
        # ---- surgery
        sg = surgery_table(rr)
        if sg:
            md.append("\n### RQ3 class-targeted removal (10%)\n\n| selection | gain target (pp) | gain others (pp) | specificity |\n|---|---|---|---|")
            for m, v in sg.items():
                md.append(f"| {m} | {fmt(v['target'], pct=True)} | {fmt(v['others'], pct=True)} | {fmt(v['spec'], digits=1)} |")
        # ---- params
        pt, ent, lrk = params_table(rr)
        if pt:
            md.append(f"\n### RQ4 parameter level (entanglement ι = {fmt(ent, digits=2)}; layer share vs layer-rollback gain ρ = {fmt(lrk, digits=2)})\n")
            md.append("| intervention | score | frac | recovered (%) | new-task cost (pp) |\n|---|---|---|---|---|")
            for (kind, name, frac), v in sorted(pt.items()):
                md.append(f"| {kind} | {name} | {100 * frac:g}% | {fmt(v['recovered'], pct=True)} | {fmt(v['cost'], pct=True)} |")
    with open(os.path.join(a.out, "results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print("\n".join(md))


if __name__ == "__main__":
    main()
