"""vLLM version of csw.sft.gen_test: same prompts, same output schema, same sampling (sample 0 greedy, sample 1 temperature 0.7 top-p 0.95), one process that keeps the GPU full with continuous batching.
python -m csw.sft.gen_vllm --model work/runs/abl_F/best --data work/ablation/A/test_Edone.jsonl --out gens_abl/F_test_b0.jsonl [--rep-pen 1.0] [--gpu-util 0.85]"""
import argparse, collections, json, time


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--per-lang-task", type=int, default=100); ap.add_argument("--samples", type=int, default=2); ap.add_argument("--max-new", type=int, default=700); ap.add_argument("--rep-pen", type=float, default=1.0)
    ap.add_argument("--gpu-util", type=float, default=0.85); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    cnt = collections.Counter(); items = []
    for l in open(a.data, encoding="utf-8"):
        e = json.loads(l)
        if cnt[(e["lang"], e["task"])] < a.per_lang_task: cnt[(e["lang"], e["task"])] += 1; items.append(e)
    ids = [tok(f"<bos><start_of_turn>user\n{e['prompt']}<end_of_turn>\n<start_of_turn>model\n", add_special_tokens=False)["input_ids"] for e in items]
    stop = [tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id]
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_util, max_model_len=2048, seed=a.seed, enable_prefix_caching=False)
    t0 = time.time(); gens = [[] for _ in items]
    for k in range(a.samples):
        sp = SamplingParams(temperature=0.0, repetition_penalty=a.rep_pen, max_tokens=a.max_new, stop_token_ids=stop, skip_special_tokens=False) if k == 0 else \
            SamplingParams(temperature=0.7, top_p=0.95, max_tokens=a.max_new, stop_token_ids=stop, skip_special_tokens=False, seed=a.seed + k)
        outs = llm.generate([{"prompt_token_ids": x} for x in ids], sp, use_tqdm=False)
        for i, o in enumerate(outs): gens[i].append(tok.decode(o.outputs[0].token_ids, skip_special_tokens=False).replace("<end_of_turn>", "").replace("<eos>", "").replace("<pad>", "").strip())
        print(f"sample {k} done {len(items)} prompts {time.time()-t0:.0f}s", flush=True)
    with open(a.out, "w", encoding="utf-8") as fo:
        for e, g in zip(items, gens): fo.write(json.dumps({"id": e["id"], "lang": e["lang"], "task": e["task"], "prompt": e["prompt"], "reference": e["completion"], "switch": e.get("switch"), "generations": g, "model": a.model}, ensure_ascii=False) + "\n")
    print("DONE", len(items), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
