"""N sampled completions per test prompt (temperature 0.7, top-p 0.95), one vLLM process per model.
python -m csw.sft.gen_samples --model DIR --data final_v5/test_off.jsonl --out gens/NAME_samples8.jsonl [--n 8]"""
import argparse, json, time


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--top-p", type=float, default=0.95); ap.add_argument("--max-new", type=int, default=700)
    ap.add_argument("--gpu-util", type=float, default=0.28); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model); items = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    ids = [tok(f"<bos><start_of_turn>user\n{e['prompt']}<end_of_turn>\n<start_of_turn>model\n", add_special_tokens=False)["input_ids"] for e in items]
    stop = [tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id]
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_util, max_model_len=2048, seed=a.seed, enable_prefix_caching=True)
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_new, stop_token_ids=stop, skip_special_tokens=False, seed=a.seed)
    t0 = time.time(); outs = llm.generate([{"prompt_token_ids": x} for x in ids], sp, use_tqdm=False); clean = lambda t: t.replace("<end_of_turn>", "").replace("<eos>", "").replace("<pad>", "").strip()
    with open(a.out, "w", encoding="utf-8") as fo:
        for e, o in zip(items, outs): fo.write(json.dumps({"id": e["id"], "lang": e["lang"], "task": e["task"], "switch": e.get("switch"), "prompt": e["prompt"], "samples": [clean(c.text) for c in o.outputs], "model": a.model}, ensure_ascii=False) + "\n")
    print("DONE", len(items), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
