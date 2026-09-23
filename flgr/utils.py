"""Small shared utilities: seeding, stateless hashing RNG, flat-parameter helpers, I/O."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import tempfile
from typing import Dict, Iterable, List

import numpy as np
import torch

# ----------------------------------------------------------------------------- seeding


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    st = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


# ----------------------------------------------------------------------------- stateless hash RNG
# Counterfactual retraining must keep every random decision about a *remaining* sample
# identical when other samples are removed. All per-sample randomness (augmentation,
# buffer admission) is therefore a pure function of (seed, stream, a, b, sample index).

def _splitmix64(x: np.ndarray) -> np.ndarray:
    x = (x + np.uint64(0x9E3779B97F4A7C15))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def hash_uniform(seed: int, stream: int, a: int, b: int, idx: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Deterministic U[0,1) numbers, shape [len(idx), k]; a pure function of (seed, stream, a, b, idx, j)."""
    ids = idx.detach().cpu().numpy().astype(np.uint64)
    key = np.uint64((((seed * 1_000_003 + stream) * 10_007 + a) * 101 + b) & 0xFFFFFFFFFFFF)
    out = np.empty((len(ids), k), dtype=np.float64)
    with np.errstate(over="ignore"):
        h0 = _splitmix64(_splitmix64(key ^ np.uint64(0xD1B54A32D192ED03)) ^ ids)
        for j in range(k):
            h = _splitmix64(h0 + np.uint64(j) * np.uint64(0x9E3779B97F4A7C15))
            out[:, j] = (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return torch.from_numpy(out).float().to(idx.device)


def fixed_permutation(n: int, seed: int, stream: int, a: int, b: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(int((seed * 1_000_003 + stream * 10_007 + a * 101 + b) % (2**63 - 1)))
    return torch.randperm(n, generator=g)


# ----------------------------------------------------------------------------- flat params


def params_of(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def flat(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def flat_params(model: torch.nn.Module) -> torch.Tensor:
    return flat(p.detach() for p in params_of(model))


def unflat_like(vec: torch.Tensor, like: List[torch.Tensor]) -> List[torch.Tensor]:
    out, i = [], 0
    for t in like:
        n = t.numel()
        out.append(vec[i:i + n].view_as(t))
        i += n
    return out


@torch.no_grad()
def set_flat_params(model: torch.nn.Module, vec: torch.Tensor) -> None:
    for p, v in zip(params_of(model), unflat_like(vec, params_of(model))):
        p.copy_(v)


def buffers_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.named_buffers()}


@torch.no_grad()
def load_buffers(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    for k, v in model.named_buffers():
        v.copy_(state[k])


def has_running_stats(model: torch.nn.Module) -> bool:
    return any(k.endswith("running_mean") for k, _ in model.named_buffers())


# ----------------------------------------------------------------------------- I/O


def atomic_torch_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    os.close(fd)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def append_jsonl(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


def env_info() -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def git_commit(repo_dir: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)
