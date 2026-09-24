#!/usr/bin/env python
"""Command-line entry point.

    python scripts/run.py train   configs/pm5_mlp.yaml seed=0            # tracked run (resumable)
    python scripts/run.py scores  configs/pm5_mlp.yaml seed=0            # baseline attribution scores
    python scripts/run.py removal configs/pm5_mlp.yaml seed=0            # RQ2/RQ3/RQ7 removal-and-retrain
    python scripts/run.py lds     configs/pm5_mlp.yaml seed=0            # RQ2 linear datamodeling score
    python scripts/run.py params  configs/pm5_mlp.yaml seed=0            # RQ4 rollback / freeze-and-retrain
    python scripts/run.py surgery configs/pm5_mlp.yaml seed=0            # RQ3 class-targeted removal
    python scripts/run.py cf      configs/pm5_mlp.yaml seed=0            # RQ13 counterfactual validity (subset removal)
    python scripts/run.py all     configs/pm5_mlp.yaml seed=0            # train, scores, removal, surgery, lds, params

Every sub-command is idempotent and resumable: finished units of work are skipped.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flgr.config import load_config, run_name  # noqa: E402
from flgr import experiments  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "scores", "removal", "lds", "params", "surgery", "cf", "all"])
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    run_dir = a.run_dir or os.path.join(cfg.get("out_root", "./runs"), run_name(cfg))
    cmds = ["train", "scores", "removal", "surgery", "lds", "params"] if a.command == "all" else [a.command]
    for c in cmds:
        getattr(experiments, f"cmd_{c}")(cfg, run_dir)


if __name__ == "__main__":
    main()
