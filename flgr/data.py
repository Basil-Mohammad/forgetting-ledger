"""Continual-learning scenarios with deterministic, removal-stable data pipelines.

A :class:`Scenario` owns the raw tensors of one dataset (kept on the compute device) and a
list of :class:`Task` objects. Everything that is random about a *sample* (split membership,
input permutation, augmentation) is a pure function of the seed and the sample index, so
that counterfactual retraining with some samples removed leaves all other samples untouched.
"""
from __future__ import annotations

import gzip
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .utils import fixed_permutation, hash_uniform

STREAM_SPLIT, STREAM_PERM, STREAM_AUG, STREAM_ORDER, STREAM_BUFFER = 1, 2, 3, 4, 5

_MIRRORS = {
    "mnist": "https://raw.githubusercontent.com/fgnt/mnist/master/{f}.gz",
    "fashion": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{f}.gz",
}
_IDX = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
_STATS = {  # per-channel mean / std
    "mnist": ([0.1307], [0.3081]),
    "fashion": ([0.2860], [0.3530]),
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]),
    "cifar100": ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
}
FASHION_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
CIFAR10_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# ----------------------------------------------------------------------------- raw loading

def _read_idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
    nd = buf[3]
    dims = [int.from_bytes(buf[4 + 4 * i: 8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(buf, dtype=np.uint8, offset=4 + 4 * nd).reshape(dims)


def _load_idx_dataset(name: str, root: str):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    for f in _IDX:
        p = os.path.join(d, f + ".gz")
        if not os.path.exists(p):
            urllib.request.urlretrieve(_MIRRORS[name].format(f=f), p)
    xtr, ytr, xte, yte = (_read_idx(os.path.join(d, f + ".gz")) for f in _IDX)
    return (torch.from_numpy(xtr.copy()).unsqueeze(1), torch.from_numpy(ytr.astype(np.int64)),
            torch.from_numpy(xte.copy()).unsqueeze(1), torch.from_numpy(yte.astype(np.int64)))


def _load_cifar_batches(d: str):
    import pickle
    def rd(f):
        with open(os.path.join(d, f), "rb") as fh:
            b = pickle.load(fh, encoding="bytes")
        return (torch.from_numpy(np.asarray(b[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)),
                torch.tensor(b[b"labels"], dtype=torch.int64))
    parts = [rd(f"data_batch_{i}") for i in range(1, 6)]
    xte, yte = rd("test_batch")
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]), xte, yte


def _load_cifar(name: str, root: str):
    local = os.path.join(root, name, "cifar-10-batches-py")
    if name == "cifar10" and os.path.exists(os.path.join(local, "data_batch_1")):
        return _load_cifar_batches(local)
    import torchvision  # imported lazily: not needed for the MNIST family
    cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    tr, te = cls(root, train=True, download=True), cls(root, train=False, download=True)
    f = lambda ds: (torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(ds.targets, dtype=torch.int64))
    return (*f(tr), *f(te))


def _synthetic(name: str, n_train: int = 1200, n_test: int = 400):
    """Random tensors with the dataset's shapes (for tests and smoke runs without downloads)."""
    g = torch.Generator().manual_seed(0)
    nc = 100 if name == "cifar100" else 10
    shape = (3, 32, 32) if name.startswith("cifar") else (1, 28, 28)
    ytr = torch.arange(n_train) % nc
    yte = torch.arange(n_test) % nc
    # class-dependent means make the task learnable
    base = torch.randint(0, 256, (nc,) + shape, generator=g).float()
    mk = lambda y: (0.6 * base[y] + 0.4 * torch.randint(0, 256, (len(y),) + shape, generator=g).float()).to(torch.uint8)
    return mk(ytr), ytr, mk(yte), yte


def load_raw(name: str, root: str):
    if root == ":synthetic:":
        return _synthetic(name)
    if name in ("mnist", "fashion"):
        return _load_idx_dataset(name, root)
    if name in ("cifar10", "cifar100"):
        return _load_cifar(name, root)
    raise ValueError(f"unknown dataset {name}")


# ----------------------------------------------------------------------------- scenario

@dataclass
class Task:
    tid: int
    classes: List[int]                   # dataset labels that belong to this task
    train_idx: torch.Tensor              # indices into Scenario.xtr
    probe_idx: torch.Tensor              # held-out indices into Scenario.xtr (never trained on)
    test_idx: torch.Tensor               # indices into Scenario.xte
    names: List[str] = field(default_factory=list)


class Scenario:
    """Sequence of tasks. ``setting`` in {"class", "task", "domain"}.

    Head layout: class-IL / task-IL use one logit per dataset class; domain-IL uses
    ``classes_per_task`` shared logits. ``target`` maps dataset labels into head space.
    """

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.name = cfg["benchmark"]
        self.seed = int(cfg["seed"])                         # init, visiting order, augmentation, buffer
        self.data_seed = int(cfg.get("data_seed", cfg["seed"]))
        self.setting = cfg["setting"]
        root = cfg.get("data_root", "./data")
        ds = {"pmnist": "mnist", "smnist": "mnist", "sfmnist": "fashion",
              "scifar10": "cifar10", "scifar100": "cifar100"}[self.name]
        self.dataset = ds
        xtr, ytr, xte, yte = load_raw(ds, root)
        self.xtr, self.ytr = xtr.to(device), ytr.to(device)
        self.xte, self.yte = xte.to(device), yte.to(device)
        mean, std = _STATS[ds]
        self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, -1, 1, 1)
        self.augment = bool(cfg.get("augment", ds.startswith("cifar")))
        self.flatten = cfg["model"]["name"] == "mlp"
        self.n_tasks = int(cfg["n_tasks"])
        self.class_names = FASHION_NAMES if ds == "fashion" else CIFAR10_NAMES if ds == "cifar10" else [str(i) for i in range(100 if ds == "cifar100" else 10)]
        self._build_tasks()

    # ------------------------------------------------------------------ construction
    def _build_tasks(self):
        cfg = self.cfg
        s = self.data_seed                                   # data split / permutations: data_seed
        n_classes = int(self.ytr.max().item()) + 1
        n_probe = int(cfg.get("probe_per_class", 20))
        n_train_cap = cfg.get("train_per_task")          # optional cap on training samples per task
        self.perms: Dict[int, torch.Tensor] = {}
        self.tasks: List[Task] = []
        if self.name == "pmnist":                       # every task: all 10 digits, own pixel permutation
            self.classes_per_task = 10
            for t in range(self.n_tasks):
                perm = torch.arange(784) if t == 0 else self._partial_perm(float(cfg.get("perm_frac", 1.0)), s, t)
                self.perms[t] = perm.to(self.device)
                order = fixed_permutation(len(self.ytr), s, STREAM_SPLIT, t).to(self.device)
                probe = torch.cat([order[self.ytr[order] == c][:n_probe] for c in range(10)])
                rest = order[~torch.isin(order, probe)]
                cap = cfg.get("first_task_train") if t == 0 and cfg.get("first_task_train") else n_train_cap
                if cap:
                    rest = rest[: int(cap)]
                self.tasks.append(Task(t, list(range(10)), rest, probe, torch.arange(len(self.yte), device=self.device)))
        else:
            cpt = n_classes // self.n_tasks
            self.classes_per_task = cpt
            class_order = list(range(n_classes))
            if cfg.get("shuffle_classes", False):
                class_order = fixed_permutation(n_classes, s, STREAM_SPLIT, 999).tolist()
            for t in range(self.n_tasks):
                cls = class_order[t * cpt:(t + 1) * cpt]
                tr, pr = [], []
                for c in cls:
                    idx = torch.nonzero(self.ytr == c).squeeze(1)
                    idx = idx[fixed_permutation(len(idx), s, STREAM_SPLIT, t, c).to(self.device)]
                    pr.append(idx[:n_probe])
                    tr.append(idx[n_probe:])
                tr = torch.cat(tr)
                tr = tr[fixed_permutation(len(tr), s, STREAM_SPLIT, t, 12345).to(self.device)]
                cap = cfg.get("first_task_train") if t == 0 and cfg.get("first_task_train") else n_train_cap
                if cap:
                    tr = tr[: int(cap)]
                te = torch.nonzero(torch.isin(self.yte, torch.tensor(cls, device=self.device))).squeeze(1)
                self.tasks.append(Task(t, cls, tr, torch.cat(pr), te, [self.class_names[c] for c in cls]))
        self.n_outputs = self.classes_per_task if self.setting == "domain" else n_classes

    @staticmethod
    def _partial_perm(frac: float, s: int, t: int) -> torch.Tensor:
        """Permutation that shuffles a random fraction ``frac`` of the 784 pixel positions among
        themselves (frac=1: full permutation; frac->0: identity). Controls input-space task overlap."""
        perm = torch.arange(784)
        k = int(round(frac * 784))
        if k >= 2:
            pos = fixed_permutation(784, s, STREAM_PERM, t, 1)[:k]
            perm[pos] = pos[fixed_permutation(k, s, STREAM_PERM, t, 2)]
        return perm

    # ------------------------------------------------------------------ inputs / targets
    def inputs(self, x_uint8: torch.Tensor, task: int, idx: Optional[torch.Tensor] = None,
               epoch: int = -1, train: bool = False) -> torch.Tensor:
        x = x_uint8.float().div_(255.0)
        if train and self.augment and idx is not None:
            x = self._augment(x, task, epoch, idx)
        x = (x - self.mean) / self.std
        if self.flatten:
            x = x.reshape(len(x), -1)
            if task in self.perms:
                x = x[:, self.perms[task]]
        elif task in self.perms:
            n, c, h, w = x.shape
            x = x.reshape(n, -1)[:, self.perms[task]].reshape(n, c, h, w)
        return x

    def _augment(self, x, task, epoch, idx):
        """Random crop (pad 4) + horizontal flip; randomness = hash(seed, task, epoch, idx)."""
        u = hash_uniform(self.seed, STREAM_AUG, task, epoch, idx, k=3)
        n, c, h, w = x.shape
        xp = torch.nn.functional.pad(x, (4, 4, 4, 4))
        ox = (u[:, 0] * 9).long().clamp_(max=8)
        oy = (u[:, 1] * 9).long().clamp_(max=8)
        ar = torch.arange(h, device=x.device)
        rows = (oy.view(-1, 1) + ar.view(1, -1))                     # [n,h]
        cols = (ox.view(-1, 1) + ar.view(1, -1))                     # [n,w]
        flip = u[:, 2] < 0.5
        cols = torch.where(flip.view(-1, 1), cols.flip(1), cols)
        bidx = torch.arange(n, device=x.device).view(n, 1, 1, 1)
        cidx = torch.arange(c, device=x.device).view(1, c, 1, 1)
        return xp[bidx, cidx, rows.view(n, 1, h, 1), cols.view(n, 1, 1, w)]

    def target(self, y: torch.Tensor) -> torch.Tensor:
        if self.setting == "domain":
            return y % self.classes_per_task if self.name != "pmnist" else y
        return y

    def task_of_label(self, y: torch.Tensor) -> torch.Tensor:
        if self.name == "pmnist":
            raise ValueError("labels do not identify tasks in pmnist")
        return y // self.classes_per_task

    # ------------------------------------------------------------------ logit masks
    def logit_mask(self, current_task: int, sample_task: Optional[torch.Tensor] = None, n: int = 1) -> Optional[torch.Tensor]:
        """Boolean mask of allowed logits.

        class-IL: all classes of tasks <= current_task. task-IL: classes of the sample's own task.
        domain-IL: no mask.
        """
        if self.setting == "domain":
            return None
        cpt = self.classes_per_task
        if self.setting == "class":
            m = torch.zeros(self.n_outputs, dtype=torch.bool, device=self.device)
            for t in range(current_task + 1):
                m[self.tasks[t].classes] = True
            return m.view(1, -1).expand(n, -1)
        assert sample_task is not None
        m = torch.zeros(len(sample_task), self.n_outputs, dtype=torch.bool, device=self.device)
        for t in sample_task.unique().tolist():
            rows = sample_task == t
            m[rows.nonzero().squeeze(1).view(-1, 1), torch.tensor(self.tasks[t].classes, device=self.device).view(1, -1)] = True
        return m

    # ------------------------------------------------------------------ batches
    def train_batch(self, task: int, idx: torch.Tensor, epoch: int):
        x = self.inputs(self.xtr[idx], task, idx, epoch, train=True)
        return x, self.target(self.ytr[idx])

    def probe_batch(self, task: int):
        t = self.tasks[task]
        return self.inputs(self.xtr[t.probe_idx], task), self.target(self.ytr[t.probe_idx]), self.ytr[t.probe_idx]

    def test_batch(self, task: int):
        t = self.tasks[task]
        return self.inputs(self.xte[t.test_idx], task), self.target(self.yte[t.test_idx]), self.yte[t.test_idx]

    def epoch_order(self, task: int, epoch: int) -> torch.Tensor:
        """Fixed visiting order of task samples (positions into Task.train_idx)."""
        n = len(self.tasks[task].train_idx)
        return fixed_permutation(n, self.seed, STREAM_ORDER, task, epoch).to(self.device)

    # ------------------------------------------------------------------ probe groups
    def groups(self, upto_task: int, group_by: str = "class") -> List[dict]:
        """Old-knowledge groups for tasks < upto_task: list of {name, task, label, pos} where
        ``pos`` indexes into the concatenated probe set of those tasks."""
        out, offset = [], 0
        for t in range(upto_task):
            tk = self.tasks[t]
            ylab = self.ytr[tk.probe_idx]
            if group_by == "task":
                out.append(dict(name=f"T{t}", task=t, label=-1, pos=torch.arange(len(ylab), device=self.device) + offset))
            else:
                for c in tk.classes:
                    pos = torch.nonzero(ylab == c).squeeze(1) + offset
                    nm = self.class_names[c] if self.name != "pmnist" else f"T{t}:{c}"
                    out.append(dict(name=nm, task=t, label=c, pos=pos))
            offset += len(ylab)
        return out
