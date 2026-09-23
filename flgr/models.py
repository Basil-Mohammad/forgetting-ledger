"""Backbones: MLP and the reduced ResNet-18 (nf=20) used by GEM / A-GEM / ER / DER."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_out: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden] * depth
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = nn.Linear(hidden, n_out)

    def features(self, x):
        for l in self.layers:
            x = F.relu(l(x))
        return x

    def forward(self, x):
        return self.head(self.features(x))


def _norm(kind: str, c: int) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(c)
    if kind == "gn":
        return nn.GroupNorm(num_groups=min(4, c), num_channels=c)
    if kind == "none":
        return nn.Identity()
    raise ValueError(kind)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin: int, cout: int, stride: int, norm: str):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = _norm(norm, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = _norm(norm, cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), _norm(norm, cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18Reduced(nn.Module):
    """ResNet-18 with nf=20 base width (Lopez-Paz & Ranzato 2017; ~1.1M params)."""

    def __init__(self, n_out: int, nf: int = 20, norm: str = "gn", in_ch: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, nf, 3, 1, 1, bias=False)
        self.bn1 = _norm(norm, nf)
        cfg = [(nf, 1), (nf * 2, 2), (nf * 4, 2), (nf * 8, 2)]
        layers, cin = [], nf
        for cout, stride in cfg:
            layers.append(nn.Sequential(BasicBlock(cin, cout, stride, norm), BasicBlock(cout, cout, 1, norm)))
            cin = cout
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.head = nn.Linear(nf * 8, n_out)

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return F.adaptive_avg_pool2d(out, 1).flatten(1)

    def forward(self, x):
        return self.head(self.features(x))


def build_model(cfg: dict, scenario) -> nn.Module:
    m = cfg["model"]
    if m["name"] == "mlp":
        return MLP(784, scenario.n_outputs, hidden=int(m.get("hidden", 256)), depth=int(m.get("depth", 2)))
    if m["name"] == "resnet18r":
        return ResNet18Reduced(scenario.n_outputs, nf=int(m.get("nf", 20)), norm=m.get("norm", "gn"))
    raise ValueError(m["name"])


# ----------------------------------------------------------------------------- unit structure

def unit_index(model: nn.Module) -> Tuple[torch.Tensor, List[dict], torch.Tensor, List[str]]:
    """Map every trainable scalar parameter to a *unit* and a *layer*.

    A unit is an output neuron (Linear) or an output channel (Conv / norm affine). Returns
    (unit_of_param [P], unit_meta, layer_of_param [P], layer_names).
    """
    unit_of, layer_of, meta, lnames = [], [], [], []
    u0 = 0
    for li, (name, p) in enumerate((n, p) for n, p in model.named_parameters() if p.requires_grad):
        lnames.append(name)
        n_units = p.shape[0]
        per = p.numel() // n_units
        unit_of.append((torch.arange(n_units).repeat_interleave(per) + u0))
        layer_of.append(torch.full((p.numel(),), li, dtype=torch.long))
        # weights and biases / affine params of the same module share units
        module = name.rsplit(".", 1)[0]
        for j in range(n_units):
            meta.append(dict(param=name, module=module, index=j))
        u0 += n_units
    return torch.cat(unit_of), meta, torch.cat(layer_of), lnames


def merge_units_by_module(meta: List[dict], unit_of: torch.Tensor) -> Tuple[torch.Tensor, List[dict]]:
    """Merge weight/bias units of the same module and output index into one unit."""
    key_to_new: Dict[Tuple[str, int], int] = {}
    new_meta, remap = [], torch.empty(len(meta), dtype=torch.long)
    for i, m in enumerate(meta):
        k = (m["module"], m["index"])
        if k not in key_to_new:
            key_to_new[k] = len(new_meta)
            new_meta.append(dict(module=m["module"], index=m["index"]))
        remap[i] = key_to_new[k]
    return remap[unit_of], new_meta
