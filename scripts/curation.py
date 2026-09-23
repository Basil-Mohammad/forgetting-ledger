#!/usr/bin/env python
"""Proxy-based data curation (is harm a property of the data?).

Harm scores are averaged over *proxy* runs (other seeds and/or other architectures, same data
split). The top-q% samples are removed before training task B in held-out *target* runs, which never
contributed to the scores. Compared against random removal with the same budget.

    python scripts/curation.py --runs ./runs --proxy pmnist-domain-mlp-finetune-s{0..4} \
        --targets pmnist-domain-mlp-finetune-s{5..9} pmnist-domain-cnn-finetune-cnn-s0 --q 0.05 0.1 0.2
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flgr.experiments import _needs, _results, _save_results, retrain_avg, setup  # noqa: E402
from flgr.utils import hash_uniform  # noqa: E402


def harm(run_dir):
    f = os.path.join(run_dir, "ledger", "task1.pt")
    return torch.load(f, map_location="cpu", weights_only=False)["data"].sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--proxy", nargs="+", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--q", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument("--tag", default="proxy")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    H = torch.stack([harm(os.path.join(a.runs, p)) for p in a.proxy])      # [n_proxy, N]
    # rank-average across proxies (robust to scale differences between runs)
    ranks = torch.stack([h.argsort().argsort().float() for h in H]).mean(0)
    for tgt in a.targets:
        rd = os.path.join(a.runs, tgt)
        cfg = json.load(open(os.path.join(rd, "config.json")))
        tr = setup(cfg, rd, log=False)
        N = len(tr.sc.tasks[1].train_idx)
        assert N == len(ranks), (N, len(ranks))
        R = _results(tr, "curation_task1")
        R.setdefault("ref_acc", [json.load(open(os.path.join(rd, "eval", "task0.json")))["task_acc"][0]])
        if _needs(R, "none", a.reps):
            R["none"] = retrain_avg(tr, 1, R.get("none"), a.reps); _save_results(tr, "curation_task1", R)
        for q in a.q:
            k = int(round(q * N))
            for name, score in [(a.tag, ranks), ("random", hash_uniform(int(cfg["seed"]), 555, 1, 0, torch.arange(N)).squeeze(1))]:
                key = f"{name}@{q}"
                if not _needs(R, key, a.reps):
                    continue
                keep = torch.ones(N, dtype=torch.bool)
                keep[torch.argsort(score, descending=True)[:k]] = False
                R[key] = retrain_avg(tr, 1, R.get(key), a.reps, keep=keep.to(tr.device))
                R[key]["proxies"] = a.proxy if name != "random" else []
                _save_results(tr, "curation_task1", R)
                print(f"[curation] {tgt} {key}: old {R[key]['old_acc']:.4f} (none {R['none']['old_acc']:.4f}) new {R[key]['new_acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
