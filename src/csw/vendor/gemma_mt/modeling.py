# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model construction / loading / saving helpers (torch + transformers; imported lazily by the trainers)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def tiny_gemma3_config(vocab_size: int = 512, **overrides) -> Any:
    """A few-layer random Gemma3 text config for offline tests: 6 layers = 5 sliding-window layers + 1 global one,
    window 8 (smaller than the test sequences, so the sliding mask matters)."""
    from transformers import Gemma3TextConfig

    kw: Dict[str, Any] = dict(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, sliding_window=8, max_position_embeddings=512, query_pre_attn_scalar=16,
        rope_local_base_freq=10000.0, pad_token_id=0, bos_token_id=2, eos_token_id=[1, 106],
    )
    kw.update(overrides)
    return Gemma3TextConfig(**kw)


def build_tiny_model(seed: int = 0, attn_implementation: str = "sdpa", dtype=None, **overrides):
    import torch
    from transformers import Gemma3ForCausalLM

    if attn_implementation == "varlen":
        from csw.vendor.gemma_mt.attention import register_varlen_attention

        register_varlen_attention()
    cfg = tiny_gemma3_config(**overrides)
    cfg._attn_implementation = attn_implementation
    torch.manual_seed(seed)
    model = Gemma3ForCausalLM(cfg)
    if dtype is not None:
        model = model.to(dtype)
    return model


def load_model(name_or_path: str, *, attn_implementation: str = "sdpa", dtype=None, device: Optional[str] = None):
    """`name_or_path` = Hub id / local dir, or the literal "tiny-random" (offline smoke tests)."""
    import torch

    dtype = dtype or torch.float32
    if name_or_path == "tiny-random":
        model = build_tiny_model(attn_implementation=attn_implementation, dtype=dtype)
    else:
        from transformers import AutoModelForCausalLM

        impl = attn_implementation
        if impl == "varlen":
            from csw.vendor.gemma_mt.attention import register_varlen_attention

            register_varlen_attention()
        if impl == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[gemma_mt] flash_attn is not installed: falling back to sdpa", flush=True)
                impl = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=dtype, attn_implementation=impl)
        if not hasattr(model, "lm_head") or not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise ValueError(f"{name_or_path} is not a text-only Gemma3ForCausalLM (got {type(model).__name__}); "
                             "this module fine-tunes google/gemma-3-270m-it")
    if device:
        model = model.to(device)
    return model


def sliding_window_of(model: Any) -> Optional[int]:
    return getattr(model.config, "sliding_window", None)


def save_model(model: Any, tokenizer: Any, path: str, stop_ids=None, dtype=None, greedy_generation_config: bool = True) -> None:
    """save_pretrained (safetensors) + tokenizer + a generation_config whose eos list contains <end_of_turn>."""
    import copy

    os.makedirs(path, exist_ok=True)
    m = getattr(model, "module", model)
    if stop_ids is not None:
        m.generation_config.eos_token_id = list(stop_ids)
    if dtype is not None:
        m = copy.deepcopy(m).to(dtype)
    m.save_pretrained(path, safe_serialization=True)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(path)
    if greedy_generation_config:                  # D-67 (R12): exported checkpoints carry a GREEDY generation_config (the Gemma default samples with top_k 64 / top_p 0.95: -1.4 chrF++ for a bare generate())
        from csw.vendor.gemma_mt.format import write_greedy_generation_config

        write_greedy_generation_config(path, stop_ids=list(stop_ids) if stop_ids is not None else None)
