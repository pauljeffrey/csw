"""Depth up-scaling of a csw checkpoint: new local-attention layers interleaved between the original ones, initialised as the identity.
    python -m csw.sft.upscale --src work/base2 --out work/base_up --every 3 [--verify]
After every `every` original layers one new layer is inserted (18 layers, every 3 -> 24).  A new layer starts as a copy of the layer before it with the attention output projection and
the MLP down projection set to zero; both sub-blocks end in a norm before the residual add, so a zero projection adds exactly nothing and the up-scaled model reproduces the source logits.
New layers use sliding-window attention; the original layers keep their own type (including the full-attention ones)."""
import argparse, copy, json, re, sys
from pathlib import Path


def plan(types: list[str], every: int, per: int = 1):
    """-> (new layer_types, for each new index the source index and whether it is a new identity layer)"""
    out, src = [], []
    for i, t in enumerate(types):
        out.append(t); src.append((i, False))
        if (i + 1) % every == 0:
            for _ in range(per): out.append("sliding_attention"); src.append((i, True))
    return out, src


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); ap.add_argument("--every", type=int, default=3); ap.add_argument("--per-group", type=int, default=1, help="new layers inserted after each group (2 with every 3: 18 -> 30 layers)")
    ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(a.src); old = AutoModelForCausalLM.from_pretrained(a.src, torch_dtype=torch.float32)
    types, src = plan(list(cfg.layer_types), a.every, a.per_group)
    new_cfg = copy.deepcopy(cfg); new_cfg.num_hidden_layers = len(types); new_cfg.layer_types = types
    new = AutoModelForCausalLM.from_config(new_cfg, torch_dtype=torch.float32)
    sd_old = old.state_dict(); sd_new = new.state_dict(); layer_re = re.compile(r"^(model\.layers\.)(\d+)(\..*)$"); loaded = set()
    for k in sd_new:
        m = layer_re.match(k)
        if not m:
            sd_new[k].copy_(sd_old[k]); loaded.add(k); continue
        j = int(m.group(2)); i, is_new = src[j]
        w = sd_old[f"{m.group(1)}{i}{m.group(3)}"].clone()
        if is_new and (".self_attn.o_proj." in k or ".mlp.down_proj." in k): w.zero_()
        sd_new[k].copy_(w); loaded.add(k)
    assert loaded == set(sd_new), "unmapped parameters"
    new.load_state_dict(sd_new)
    n_old = sum(p.numel() for p in old.parameters()); n_new = sum(p.numel() for p in new.parameters())
    print(f"layers {len(cfg.layer_types)} -> {len(types)}; parameters {n_old/1e6:.1f}M -> {n_new/1e6:.1f}M (+{(n_new - n_old)/1e6:.1f}M); new layer indices {[j for j, (_, nw) in enumerate(src) if nw]}")
    if a.verify:
        tok = AutoTokenizer.from_pretrained(a.src)
        x = tok(["<hm> <yor>+<eng>:  The market was busy this morning and everybody was talking.", "<ood> <hau>:  Ya yi aiki tuƙuru amma bai samu komai ba."], return_tensors="pt", padding=True, add_special_tokens=True)
        old.eval(); new.eval()
        with torch.no_grad(): d = (old(**x).logits - new(**x).logits).abs().max().item()
        print(f"max |logit difference| old vs up-scaled: {d:.3e}"); assert d < 1e-3, "identity init failed"
    Path(a.out).mkdir(parents=True, exist_ok=True); new.save_pretrained(a.out, safe_serialization=True); AutoTokenizer.from_pretrained(a.src).save_pretrained(a.out)
    json.dump({"source": a.src, "every": a.every, "per_group": a.per_group, "layer_types": types, "new_layers": [j for j, (_, nw) in enumerate(src) if nw], "params_old": n_old, "params_new": n_new}, open(Path(a.out) / "upscale_info.json", "w"), indent=1)


if __name__ == "__main__":
    main()
