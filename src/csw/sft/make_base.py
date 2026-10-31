"""Build the csw starting point: gemma-3-270m-translate-rl2c with the code-switcher tags as ONE token each.
python -m csw.sft.make_base --out work/base
Tokenizer slots named <unusedN> (6,242 of them in the Gemma-3 vocabulary) are renamed to <hm> <ood> <dev> </dev> and the language tags (<yor> <eng> ...); nothing is added, the vocabulary stays 262,144.
Each renamed row of the (tied) embedding matrix starts as the mean of the embeddings of the pieces the same string had before, so training does not start from noise."""
import argparse, json, os, sys
from pathlib import Path
from csw.sft import fmt2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--repo", default="BeardedMonster/gemma-3-270m-translate-rl2c"); ap.add_argument("--subdir", default="latest", help="folder inside the repo that holds the model ( '' for a plain model repo such as google/gemma-3-270m)"); a = ap.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_env = os.environ.get("HF_TOKEN")
    d = Path(snapshot_download(a.repo, allow_patterns=[f"{a.subdir}/*" if a.subdir else "*"], token=tok_env)) / a.subdir
    old = AutoTokenizer.from_pretrained(str(d))
    specials = fmt2.special_tokens(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tj = json.load(open(d / "tokenizer.json")); cfg = json.load(open(d / "tokenizer_config.json"))
    unused = sorted(((int(v), k) for k, v in tj["model"]["vocab"].items() if k.startswith("<unused")))
    assert len(unused) > len(specials)
    ren = {unused[i][1]: specials[i] for i in range(len(specials))}; ids = {specials[i]: unused[i][0] for i in range(len(specials))}
    tj["model"]["vocab"] = {ren.get(k, k): v for k, v in tj["model"]["vocab"].items()}
    for a_ in tj["added_tokens"]:
        a_["content"] = ren.get(a_["content"], a_["content"])
    for k, v in (cfg.get("added_tokens_decoder") or {}).items():
        v["content"] = ren.get(v["content"], v["content"])
    json.dump(tj, open(out / "tokenizer.json", "w"), ensure_ascii=False); json.dump(cfg, open(out / "tokenizer_config.json", "w"), ensure_ascii=False, indent=1)
    for f in d.iterdir():
        if f.name not in ("tokenizer.json", "tokenizer_config.json", "model.safetensors", "training_state.json", "README.md"):
            (out / f.name).write_bytes(f.read_bytes())
    model = AutoModelForCausalLM.from_pretrained(str(d), torch_dtype=torch.float32)
    E = model.get_input_embeddings().weight
    with torch.no_grad():
        for s, i in ids.items():
            pieces = old(s, add_special_tokens=False)["input_ids"]
            E[i] = E[pieces].mean(0)
    model.save_pretrained(str(out), safe_serialization=True)
    new = AutoTokenizer.from_pretrained(str(out))
    for s in specials:
        got = new(s, add_special_tokens=False)["input_ids"]; assert got == [ids[s]], (s, got, ids[s])
    t = new("<ood> Yoruba from Yoruba+English:  hello", add_special_tokens=False)["input_ids"]
    print("ok", len(specials), "tokens;", new.convert_ids_to_tokens(t)); json.dump(ids, open(out / "csw_token_ids.json", "w"), indent=1)


if __name__ == "__main__":
    main()
