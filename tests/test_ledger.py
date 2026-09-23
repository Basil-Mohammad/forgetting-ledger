"""Correctness tests: completeness of the ledger, exact split across sources, resume, removal stability."""
import copy
import os

import pytest
import torch

from flgr.config import load_config
from flgr.experiments import setup

HERE = os.path.dirname(__file__)


def cfg_for(bench_file, **over):
    cfg = load_config(os.path.join(HERE, "..", "configs", bench_file))
    cfg["data_root"] = ":synthetic:"
    cfg["device"] = "cpu"
    cfg["probe_per_class"] = 5
    cfg["train"].update(epochs=1, batch_size=16, lr=0.02)
    cfg["ckpt_every"] = 0
    for k, v in over.items():
        d = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = v
    return cfg


CASES = [
    ("sc10_resnet.yaml", "finetune", "gn"),
    ("sc10_resnet.yaml", "er", "gn"),
    ("sc10_resnet.yaml", "derpp", "gn"),
    ("sc10_resnet.yaml", "ewc", "gn"),
    ("sc10_resnet.yaml", "agem", "gn"),
    ("sc10_resnet.yaml", "finetune", "bn"),
    ("sm5_mlp.yaml", "er", None),
]


@pytest.mark.parametrize("bench,learner,norm", CASES)
def test_completeness_and_exact_split(tmp_path, bench, learner, norm):
    over = {"learner.name": learner, "n_tasks": 2, "learner.buffer_size": 50}
    if norm:
        over["model.nf"] = 8
        over["model.norm"] = norm
    cfg = cfg_for(bench, **over)
    tr = setup(cfg, str(tmp_path), log=False)
    tr.run()
    led = torch.load(tmp_path / "ledger" / "task1.pt", weights_only=False)
    true = led["true_dL"]
    pred = led["path_g"] + led["stats"]
    rel = float((pred - true).abs().sum() / true.abs().sum())
    assert rel < 0.05, rel
    # sources sum exactly to the parameter path (per group)
    src = led["data"].sum(0) + led["reg"] + (led["mem"].sum(0) if led["mem"] is not None else 0)
    assert torch.allclose(src, led["path_g"], rtol=1e-4, atol=1e-6), (src, led["path_g"])
    # parameters / units / layers sum to the same total
    assert torch.allclose(led["param"].sum(), led["path_g"].sum(), rtol=1e-6, atol=1e-8)
    assert torch.allclose(led["unit"].sum(0), led["path_g"], rtol=1e-6, atol=1e-8)
    assert torch.allclose(led["layer"].sum(0), led["path_g"], rtol=1e-6, atol=1e-8)
    if norm == "bn":
        assert led["stats"].abs().sum() > 0
    if learner in ("er", "derpp"):
        assert led["mem"].abs().sum() > 0
    if learner == "ewc":
        assert led["reg"].abs().sum() > 0


def test_resume_is_exact(tmp_path):
    cfg = cfg_for("sm5_mlp.yaml", n_tasks=2, ckpt_every=7)
    a = setup(cfg, str(tmp_path / "a"), log=False); a.run()
    b = setup(cfg, str(tmp_path / "b"), log=False)
    # interrupt: train task 0 fully and task 1 partially by raising after some steps
    orig = b._tracked_step
    count = {"n": 0}

    def boom(*args, **kw):
        count["n"] += 1
        if count["n"] == 10:
            raise KeyboardInterrupt
        return orig(*args, **kw)
    b._tracked_step = boom
    with pytest.raises(KeyboardInterrupt):
        b.run()
    c = setup(cfg, str(tmp_path / "b"), log=False); c.run()
    la = torch.load(tmp_path / "a" / "ledger" / "task1.pt", weights_only=False)
    lb = torch.load(tmp_path / "b" / "ledger" / "task1.pt", weights_only=False)
    for k in ("data", "param", "true_dL", "path_g"):
        assert torch.allclose(la[k], lb[k], atol=1e-9), k


def test_removal_keeps_other_samples_identical():
    from flgr.utils import hash_uniform
    idx = torch.arange(100)
    a = hash_uniform(3, 3, 1, 2, idx, k=3)
    b = hash_uniform(3, 3, 1, 2, idx[50:], k=3)
    assert torch.equal(a[50:], b)


def test_hash_buffer_stable_under_removal():
    from flgr.learners import HashBuffer
    b1, b2 = HashBuffer(20, 0, 10, "cpu"), HashBuffer(20, 0, 10, "cpu")
    all_idx = torch.arange(200)
    removed = torch.arange(0, 200, 7)
    kept = all_idx[~torch.isin(all_idx, removed)]
    for i in range(0, 200, 25):
        b1.add(all_idx[i:i + 25], 0)
        k = kept[(kept >= i) & (kept < i + 25)]
        b2.add(k, 0)
    s1 = set(b1.gidx.tolist()) - set(removed.tolist())
    assert s1 <= set(b2.gidx.tolist())


def test_adam_ledger_is_complete(tmp_path):
    cfg = cfg_for("sm5_mlp.yaml", n_tasks=2)
    cfg["train"].update(optimizer="adam", lr=0.002)
    tr = setup(cfg, str(tmp_path), log=False)
    tr.run()
    led = torch.load(tmp_path / "ledger" / "task1.pt", weights_only=False)
    rel = float((led["path_g"] - led["true_dL"]).abs().sum() / led["true_dL"].abs().sum())
    assert rel < 0.05, rel
    src = led["data"].sum(0) + led["reg"]                       # new samples + momentum carried over
    assert torch.allclose(src, led["path_g"], rtol=1e-4, atol=1e-6), (src, led["path_g"])
    assert led["reg"].abs().sum() > 0
