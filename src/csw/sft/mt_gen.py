"""Translations of English sentences by the translator model (BeardedMonster/gemma-3-270m-translate-rl2c) for the evaluator's natural-low-quality data.
python -m csw.sft.mt_gen --model work/translator --src r4/mt_src.jsonl --out r4/mt_out.jsonl --mode greedy|sample [--bs 96] [--max-new 160]
Prompt = the translation replay format: 'Translate the text to <Language>:  <english>' in the chat wrapper; the completion starts with the ISO code, which is removed."""
import argparse, json, sys, time
from pathlib import Path
from csw.sft import fmt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); ap.add_argument("--mode", default="greedy"); ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--max-new", type=int, default=160); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"; model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    rows = [json.loads(l) for l in open(a.src, encoding="utf-8")]
    if a.limit: rows = rows[: a.limit]
    rows.sort(key=lambda r: len(r["eng"])); stop = [tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id]; t0 = time.time()
    with open(a.out, "w", encoding="utf-8") as fo:
        for s in range(0, len(rows), a.bs):
            b = rows[s:s + a.bs]
            enc = tok([f"<bos><start_of_turn>user\nTranslate the text to {fmt.NAMES.get(r['lang'], r['lang'])}:  {r['eng']}<end_of_turn>\n<start_of_turn>model\n" for r in b], add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
            kw = dict(do_sample=False) if a.mode == "greedy" else dict(do_sample=True, temperature=1.0, top_p=0.95)
            with torch.no_grad(): o = model.generate(**enc, max_new_tokens=a.max_new, eos_token_id=stop, pad_token_id=tok.pad_token_id, **kw)
            for i, r in enumerate(b):
                t = tok.decode(o[i, enc["input_ids"].shape[1]:], skip_special_tokens=True).replace("<end_of_turn>", "").strip()
                if t.split(" ", 1)[0] == r["lang"]: t = t.split(" ", 1)[1] if " " in t else ""
                fo.write(json.dumps(dict(r, m=t, mode=a.mode), ensure_ascii=False) + "\n")
            fo.flush()
            if (s // a.bs) % 20 == 0: print(f"{s+len(b)}/{len(rows)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
