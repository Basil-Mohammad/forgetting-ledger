"""Text scenarios for continual fine-tuning of language models.

The language model is used as a *verbalizer classifier*: every class has a one-token label word
(" World", " Sports", ...), the prompt ends with a cue ("Topic:"), and the logits of the next
token restricted to the label words of all tasks form the classifier output (as in prompt-based
fine-tuning, e.g. LM-BFF). This keeps the whole ledger / intervention machinery unchanged:
``x`` is a LongTensor of token ids (left padded with -1), ``y`` a class index.

Benchmark ``agnews_dbpedia`` (two tasks):
  task A: AG News topics          World | Sports | Business | Science
  task B: DBpedia entity types    Company | Athlete | Politician | Vehicle | Animal | Film
Several B classes are semantically related to A classes (Company~Business, Athlete~Sports,
Politician~World, Vehicle/Animal~Science), which allows the class-pair analysis of the paper
("does a similar new class protect or harm an old class?") in a language model.
"""
from __future__ import annotations

import hashlib
import os
from typing import List, Optional

import torch

from .data import STREAM_SPLIT, Scenario, Task
from .utils import atomic_torch_save, fixed_permutation

AG_LABELS = ["World", "Sports", "Business", "Science"]                   # ag_news labels 0..3
DBP_KEEP = {0: "Company", 3: "Athlete", 4: "Politician", 5: "Vehicle", 9: "Animal", 12: "Film"}  # dbpedia_14 label -> word
PAD = -1


def _prompt_a(text: str) -> tuple:
    return "News article: ", text, "\nTopic:"


def _prompt_b(title: str, text: str) -> tuple:
    return "Wikipedia entry: ", f"{title}. {text}", "\nCategory:"


def _encode(tok, parts: tuple, L: int) -> List[int]:
    """prefix + (truncated) body + suffix, left padded with PAD to length L."""
    pre, body, suf = (tok(p, add_special_tokens=False)["input_ids"] for p in parts)
    room = L - len(pre) - len(suf)
    ids = pre + body[:max(0, room)] + suf
    ids = ids[-L:]
    return [PAD] * (L - len(ids)) + ids


def label_token_ids(tok, words: List[str]) -> List[int]:
    ids = []
    for w in words:
        t = tok(" " + w, add_special_tokens=False)["input_ids"]
        ids.append(t[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"label words do not start with distinct tokens: {list(zip(words, ids))}")
    return ids


def _load_hf(cfg: dict):
    """Tokenised AG News + DBpedia subsets (cached in data_root/text_cache)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    name = cfg["model"]["hf_name"]
    L = int(cfg.get("seq_len", 64))
    per_tr = int(cfg.get("pool_per_class", 1500))
    per_te = int(cfg.get("test_per_class", 400))
    root = cfg.get("data_root", "./data")
    key = hashlib.md5(f"{name}|{L}|{per_tr}|{per_te}|{cfg.get('data_seed', 0)}|v1".encode()).hexdigest()[:10]
    cache = os.path.join(root, "text_cache", f"agdbp_{key}.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    tok = AutoTokenizer.from_pretrained(name)
    s = int(cfg.get("data_seed", 0))
    xs, ys, xt, yt = [], [], [], []

    def take(ds, lab_col, want, per, split_seed):
        idx_by = {c: [] for c in want}
        order = fixed_permutation(len(ds), s, STREAM_SPLIT, split_seed).tolist()
        labs = ds[lab_col]
        for i in order:
            c = labs[i]
            if c in idx_by and len(idx_by[c]) < per:
                idx_by[c].append(i)
            if all(len(v) >= per for v in idx_by.values()):
                break
        return idx_by

    ag = load_dataset("fancyzhx/ag_news")
    db = load_dataset("fancyzhx/dbpedia_14")
    for split, per, X, Y, sd in [("train", per_tr, xs, ys, 1), ("test", per_te, xt, yt, 2)]:
        sel = take(ag[split], "label", list(range(4)), per, sd)
        for c, ids in sel.items():
            for i in ids:
                X.append(_encode(tok, _prompt_a(ag[split][i]["text"]), L)); Y.append(c)
        keep = list(DBP_KEEP)
        sel = take(db[split], "label", keep, per, sd + 10)
        for j, c in enumerate(keep):
            for i in sel[c]:
                r = db[split][i]
                X.append(_encode(tok, _prompt_b(r["title"], r["content"]), L)); Y.append(4 + j)
    out = dict(xtr=torch.tensor(xs), ytr=torch.tensor(ys), xte=torch.tensor(xt), yte=torch.tensor(yt),
               label_ids=label_token_ids(tok, AG_LABELS + list(DBP_KEEP.values())),
               vocab=len(tok), pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    atomic_torch_save(out, cache)                          # atomic: parallel jobs may build the cache at once
    return out


def _synthetic(cfg: dict):
    """Random token sequences with class-dependent keyword tokens (tests; no downloads)."""
    g = torch.Generator().manual_seed(0)
    L, V = int(cfg.get("seq_len", 64)), int(cfg["model"].get("vocab_size", 512))
    per_tr, per_te = int(cfg.get("pool_per_class", 200)), int(cfg.get("test_per_class", 50))

    def make(n_per):
        X, Y = [], []
        for c in range(10):
            x = torch.randint(20, V, (n_per, L), generator=g)
            kw = torch.randint(0, L - 4, (n_per, 3), generator=g)
            for k in range(3):
                x[torch.arange(n_per), kw[:, k]] = 2 + c              # class keyword token
            x[:, -1] = 1 + (c >= 4)                                    # task cue token
            nlead = torch.randint(0, L // 4, (n_per,), generator=g)
            for i in range(n_per):
                x[i, :nlead[i]] = PAD
            X.append(x); Y.append(torch.full((n_per,), c))
        return torch.cat(X), torch.cat(Y)
    xtr, ytr = make(per_tr)
    xte, yte = make(per_te)
    return dict(xtr=xtr, ytr=ytr, xte=xte, yte=yte, label_ids=list(range(V - 10, V)), vocab=V, pad_token_id=0)


class TextScenario(Scenario):
    """Two-task text scenario; see module docstring. Inputs are token ids (no augmentation)."""

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg, self.device = cfg, device
        self.name = cfg["benchmark"]
        self.seed = int(cfg["seed"])
        self.data_seed = int(cfg.get("data_seed", 0))
        self.setting = cfg.get("setting", "task")
        self.dataset = "text"
        raw = _synthetic(cfg) if cfg.get("data_root") == ":synthetic:" else _load_hf(cfg)
        self.label_ids, self.vocab, self.pad_token_id = raw["label_ids"], raw["vocab"], raw["pad_token_id"]
        self.xtr, self.ytr = raw["xtr"].to(device), raw["ytr"].to(device)
        self.xte, self.yte = raw["xte"].to(device), raw["yte"].to(device)
        self.augment, self.flatten, self.perms = False, False, {}
        self.n_tasks = 2
        self.class_names = AG_LABELS + list(DBP_KEEP.values())
        self._build_text_tasks()

    def _build_text_tasks(self):
        cfg, s = self.cfg, self.data_seed
        n_probe = int(cfg.get("probe_per_class", 16))
        caps = [int(cfg.get("first_task_train_per_class", 1000)), int(cfg.get("train_per_class", 400))]
        self.tasks = []
        for t, cls in enumerate([[0, 1, 2, 3], [4, 5, 6, 7, 8, 9]]):
            tr, pr = [], []
            for c in cls:
                idx = torch.nonzero(self.ytr == c).squeeze(1)
                idx = idx[fixed_permutation(len(idx), s, STREAM_SPLIT, t, c).to(self.device)]
                pr.append(idx[:n_probe])
                tr.append(idx[n_probe:n_probe + caps[t]])
            tr = torch.cat(tr)
            tr = tr[fixed_permutation(len(tr), s, STREAM_SPLIT, t, 12345).to(self.device)]
            te = torch.nonzero(torch.isin(self.yte, torch.tensor(cls, device=self.device))).squeeze(1)
            self.tasks.append(Task(t, cls, tr, torch.cat(pr), te, [self.class_names[c] for c in cls]))
        self.classes_per_task = 5                                  # unused for task-IL masks
        self.n_outputs = 10

    def inputs(self, x, task: int, idx: Optional[torch.Tensor] = None, epoch: int = -1, train: bool = False):
        return x

    def task_of_label(self, y: torch.Tensor) -> torch.Tensor:
        return (y >= 4).long()
