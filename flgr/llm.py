"""A causal language model with LoRA adapters, used as a verbalizer classifier.

Only the LoRA matrices are trainable, so every ledger quantity (parameter path, per-sample
credit, units, layers) lives in the adapter space, exactly as in parameter-efficient fine-tuning
of large models. The model is kept in float32 and uses eager attention, because the ledger needs
double backward (per-sample credits) and exact endpoint losses (certified quadrature).
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

from .text import PAD


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B A x, with W, b frozen; A ~ Kaiming, B = 0 (Hu et al., 2022)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, seed: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        g = torch.Generator().manual_seed(seed)
        a = torch.empty(r, base.in_features)
        bound = 1.0 / math.sqrt(base.in_features)
        a.uniform_(-bound, bound, generator=g)
        self.lora_A = nn.Parameter(a.to(base.weight.device, base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=base.weight.dtype))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


class LLMClassifier(nn.Module):
    no_vmap = True                     # per-sample gradients are obtained by double backward, not vmap
    eval_chunk = 256

    def __init__(self, cfg: dict, scenario):
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM
        m = cfg["model"]
        if cfg.get("data_root") == ":synthetic:" or m.get("hf_name", "") == "tiny":
            conf = AutoConfig.for_model("gpt_neox", vocab_size=scenario.vocab, hidden_size=int(m.get("tiny_hidden", 64)),
                                        num_hidden_layers=int(m.get("tiny_layers", 2)), num_attention_heads=4,
                                        intermediate_size=4 * int(m.get("tiny_hidden", 64)), max_position_embeddings=256,
                                        attention_dropout=0.0, hidden_dropout=0.0)
            torch.manual_seed(1234)
            lm = AutoModelForCausalLM.from_config(conf, attn_implementation="eager")
        else:
            try:                                            # transformers >= 4.56 uses `dtype`
                lm = AutoModelForCausalLM.from_pretrained(m["hf_name"], dtype=torch.float32, attn_implementation="eager")
            except TypeError:
                lm = AutoModelForCausalLM.from_pretrained(m["hf_name"], torch_dtype=torch.float32, attn_implementation="eager")
        for k in ("attention_dropout", "hidden_dropout", "attn_pdrop", "resid_pdrop", "embd_pdrop", "dropout"):
            if hasattr(lm.config, k):
                setattr(lm.config, k, 0.0)
        for p in lm.parameters():
            p.requires_grad_(False)
        targets: List[str] = list(m.get("lora_targets", ["query_key_value"]))
        r, alpha = int(m.get("lora_r", 8)), float(m.get("lora_alpha", 16))
        n = 0
        for name, mod in list(lm.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.Linear) and cname in targets:
                    setattr(mod, cname, LoRALinear(child, r, alpha, seed=int(cfg["seed"]) * 1000 + n)); n += 1
        if n == 0:
            raise ValueError(f"no LoRA target modules named {targets}")
        self.lm = lm
        self.backbone = lm.base_model
        self.register_buffer("label_ids", torch.tensor(scenario.label_ids, dtype=torch.long), persistent=False)
        self.n_lora = n

    # ------------------------------------------------------------------ forward
    def features(self, x: torch.Tensor) -> torch.Tensor:
        if len(x) > self.eval_chunk and not torch.is_grad_enabled():
            return torch.cat([self.features(x[i:i + self.eval_chunk]) for i in range(0, len(x), self.eval_chunk)])
        mask = x != PAD
        ids = x.clamp(min=0)
        pos = (mask.long().cumsum(1) - 1).clamp(min=0)            # left padding: positions start at the first token
        h = self.backbone(input_ids=ids, attention_mask=mask.long(), position_ids=pos).last_hidden_state
        return h[:, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        W = self.lm.get_output_embeddings().weight[self.label_ids]           # [K, d]  (frozen)
        return h @ W.T

    # ------------------------------------------------------------------ state: adapters only
    def state_dict(self, *args, **kwargs):
        full = super().state_dict(*args, **kwargs)
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        prefix = kwargs.get("prefix", "")
        return type(full)((k, v) for k, v in full.items() if k[len(prefix):] in keep)

    def load_state_dict(self, state_dict, strict: bool = False, assign: bool = False):
        return super().load_state_dict(state_dict, strict=False)
