"""Generate for the held-out test prompts and keep everything (for the website of human preferences and for QE scoring).
python -m csw.sft.gen_test --model work/runs/sft1/best --data work/sft/test.jsonl --out work/gens_sft1.jsonl [--per-lang-task 60] [--samples 2]
Per prompt: sample 0 = greedy (repetition penalty 1.05), samples 1.. = temperature 0.7 top-p 0.95."""
import argparse, collections, json, random, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--per-lang-task", type=int, default=60); ap.add_argument("--samples", type=int, default=2); ap.add_argument("--bs", type=int, default=96); ap.add_argument("--max-new", type=int, default=700); ap.add_argument("--start", type=int, default=0); ap.add_argument("--rep-pen", type=float, default=1.05); ap.add_argument("--end", type=int, default=0); a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    cnt = collections.Counter(); items = []
    for l in open(a.data, encoding="utf-8"):
        e = json.loads(l)
        if cnt[(e["lang"], e["task"])] < a.per_lang_task: cnt[(e["lang"], e["task"])] += 1; items.append(e)
    items.sort(key=lambda e: len(e["prompt"])); items = items[a.start:(a.end or None)]; print(len(items), "prompts", flush=True)
    stop = [tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id]; t0 = time.time()
    with open(a.out, "w", encoding="utf-8") as fo:
        for s in range(0, len(items), a.bs):
            b = items[s:s + a.bs]
            enc = tok([f"<bos><start_of_turn>user\n{e['prompt']}<end_of_turn>\n<start_of_turn>model\n" for e in b], add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
            gens = [[] for _ in b]
            for k in range(a.samples):
                kw = dict(do_sample=False, repetition_penalty=a.rep_pen) if k == 0 else dict(do_sample=True, temperature=0.7, top_p=0.95)
                with torch.no_grad():
                    o = model.generate(**enc, max_new_tokens=a.max_new, eos_token_id=stop, pad_token_id=tok.pad_token_id, **kw)
                for i in range(len(b)): gens[i].append(tok.decode(o[i, enc["input_ids"].shape[1]:], skip_special_tokens=False).replace("<end_of_turn>", "").replace("<eos>", "").replace("<pad>", "").strip())
            for e, g in zip(b, gens): fo.write(json.dumps({"id": e["id"], "lang": e["lang"], "task": e["task"], "prompt": e["prompt"], "reference": e["completion"], "switch": e.get("switch"), "generations": g, "model": a.model}, ensure_ascii=False) + "\n")
            fo.flush(); print(f"{s+len(b)}/{len(items)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
