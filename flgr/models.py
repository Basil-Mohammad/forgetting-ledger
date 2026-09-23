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


class SmallCNN(nn.Module):
    """Three conv blocks (32-64-64) + FC-128; ~0.2M parameters on CIFAR, ~0.1M on MNIST."""

    def __init__(self, n_out: int, in_ch: int, size: int, width: int = 32, norm: str = "none"):
        super().__init__()
        c1, c2 = width, 2 * width
        self.conv1, self.n1 = nn.Conv2d(in_ch, c1, 3, padding=1), _norm(norm, c1)
        self.conv2, self.n2 = nn.Conv2d(c1, c2, 3, padding=1), _norm(norm, c2)
        self.conv3, self.n3 = nn.Conv2d(c2, c2, 3, padding=1), _norm(norm, c2)
        s = size // 2 // 2 // 2
        self.fc = nn.Linear(c2 * s * s, 128)
        self.head = nn.Linear(128, n_out)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.n1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.n2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.n3(self.conv3(x))), 2)
        return F.relu(self.fc(x.flatten(1)))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(cfg: dict, scenario) -> nn.Module:
    m = cfg["model"]
    if m["name"] == "mlp":
        return MLP(784, scenario.n_outputs, hidden=int(m.get("hidden", 256)), depth=int(m.get("depth", 2)))
    if m["name"] == "cnn":
        cifar = scenario.dataset.startswith("cifar")
        return SmallCNN(scenario.n_outputs, 3 if cifar else 1, 32 if cifar else 28, int(m.get("width", 32)), m.get("norm", "none"))
    if m["name"] == "resnet18r":
        return ResNet18Reduced(scenario.n_outputs, nf=int(m.get("nf", 20)), norm=m.get("norm", "gn"))
    raise ValueError(m["name"])


# ----------------------------------------------------------------------------- unit structure

def unit_index(model: nn.Module) -> Tuple[torch.Tensor, List[dict], torch.Tensor, List[str]]:
    """Map every trainable scalar parameter to a *unit* and a *layer*.

    A unit is one output neuron of a Linear layer or one output channel of a Conv / norm layer;
    the weight row and the bias entry of the same output share the unit. Returns
    (unit_of_param [P], unit_meta [U], layer_of_param [P], layer_names [L]).
    """
    unit_of, layer_of, meta, lnames = [], [], [], []
    key_to_unit: Dict[Tuple[str, int], int] = {}
    for li, (name, p) in enumerate((n, p) for n, p in model.named_parameters() if p.requires_grad):
        lnames.append(name)
        module = name.rsplit(".", 1)[0]
        n_units = p.shape[0]
        per = p.numel() // n_units
        ids = []
        for j in range(n_units):
            k = (module, j)
            if k not in key_to_unit:
                key_to_unit[k] = len(meta)
                meta.append(dict(module=module, index=j))
            ids.append(key_to_unit[k])
        unit_of.append(torch.tensor(ids, dtype=torch.long).repeat_interleave(per))
        layer_of.append(torch.full((p.numel(),), li, dtype=torch.long))
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
