"""YAML configs with dotted command-line overrides (e.g. ``train.lr=0.01 seed=3``)."""
from __future__ import annotations

import copy
from typing import List

import yaml


def _set(d: dict, key: str, value):
    parts = key.split(".")
    for k in parts[:-1]:
        d = d.setdefault(k, {})
    d[parts[-1]] = value


def load_config(path: str, overrides: List[str] = ()) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    if base:
        import os
        parent = load_config(os.path.join(os.path.dirname(path), base))
        cfg = deep_merge(parent, cfg)
    for o in overrides:
        k, v = o.split("=", 1)
        _set(cfg, k, yaml.safe_load(v))
    return cfg


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def run_name(cfg: dict) -> str:
    l = cfg.get("learner", {}).get("name", "finetune")
    norm = cfg["model"].get("norm")
    return f"{cfg['benchmark']}-{cfg['setting']}-{cfg['model']['name']}{'-' + norm if norm else ''}-{l}-s{cfg['seed']}"
