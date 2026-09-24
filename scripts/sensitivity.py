#!/usr/bin/env python
"""Response of the final old-group probe losses to re-weighting a subset S of the new data (RQ13c).

The loss terms of S are multiplied by (1 - eps) while batches, order and number of steps stay identical
(visiting order 0 = the tracked trajectory). For a smooth training map the response is linear for small
eps with slope -sum_{i in S} C_i (first-order); ReLU networks make the SGD update map discontinuous, and the
curve shows where single-trajectory counterfactuals stop being predictable.

    python scripts/sensitivity.py --runs ./runs --run pmnist-domain-mlp-finetune-s0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flgr.experiments import _results, _save_results, retrain, setup  # noqa: E402
from flgr.utils import flat_params, hash_uniform  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--eps", nargs="+", type=float, default=list(np.logspace(-5, 0, 16)))
    a = ap.parse_args()
    for name in a.run:
        rd = os.path.join(a.runs, name)
        cfg = json.load(open(os.path.join(rd, "config.json")))
        cfg["data_root"] = os.environ.get("FLGR_DATA", cfg.get("data_root", "./data"))
        tr = setup(cfg, rd, log=False)
        t = 1
        N = len(tr.sc.tasks[t].train_idx)
        C = torch.load(tr.p("ledger", f"task{t}.pt"), map_location="cpu", weights_only=False)["data"]
        R = _results(tr, f"sensitivity_task{t}")
        if "base" not in R:
            r = retrain(tr, t, rep=0)
            R["base"] = dict(probe_L=r["probe_L"], theta_norm=float(flat_params(tr.model).norm()))
        base = np.array(R["base"]["probe_L"])
        th_base = None
        u = hash_uniform(int(cfg["seed"]), 93, t, 0, torch.arange(N)).squeeze(1)
        h = C.sum(1)
        subsets = {"rand1": u < 0.01, "rand5": u < 0.05,
                   "top1": torch.zeros(N, dtype=torch.bool).index_fill_(0, torch.argsort(-h)[: max(1, N // 100)], True)}
        for sname, rem in subsets.items():
            pred = (-C[rem].sum(0)).tolist()
            for e in a.eps:
                key = f"{sname}@{e:.3g}"
                if key in R:
                    continue
                if th_base is None:
                    retrain(tr, t, rep=0)
                    th_base = flat_params(tr.model).clone()
                w = torch.ones(N, device=tr.device)
                w[rem.to(tr.device)] = 1.0 - e
                r = retrain(tr, t, rep=0, zero=w)
                dth = float((flat_params(tr.model) - th_base).norm())
                R[key] = dict(eps=e, subset=sname, n=int(rem.sum()), dL=(np.array(r["probe_L"]) - base).tolist(),
                              pred=pred, dtheta=dth)
                _save_results(tr, f"sensitivity_task{t}", R)
                print(f"[sens] {name} {key}: dL={sum(R[key]['dL']):+.3e} eps*pred={e * sum(pred):+.3e} |dtheta|={dth:.2e}",
                      flush=True)


if __name__ == "__main__":
    main()
