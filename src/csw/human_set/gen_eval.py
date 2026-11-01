"""8 candidates (1 greedy + 7 sampled) for the test-set prompts, same schema as gen_main.py.
python gen_eval.py MODEL OUT CONFIG_JSON TASKS_COMMA   (tasks are the six test-set tasks; eng_m uses the translation model, the rest the RL model)"""
import json, sys, time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
model, out, cfg, tasks = sys.argv[1:5]; cfg = json.loads(cfg); tasks = tasks.split(",")
FAM = {"eng_m": "m_text", "eng_hm": "hm", "m_hm": "hm", "eng_ood": "ood", "m_ood": "ood", "hm_ood": "ood"}
rows = [json.loads(l) for l in open("/workspace/csw_hs/data/test_off.jsonl")]
rows = [r for r in rows if r["task"] in tasks and not (r["task"] == "eng_m" and r["lang"] in ("ful", "gpe", "nde"))]
tok = AutoTokenizer.from_pretrained(model)
ids = [tok(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"] for r in rows]
stop = [tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id]
llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=cfg.pop("gpu", 0.4), max_model_len=2048, seed=0, enable_prefix_caching=True, max_num_seqs=1024, max_num_batched_tokens=32768)
clean = lambda t: t.replace("<end_of_turn>", "").replace("<eos>", "").replace("<pad>", "").strip()
t0 = time.time(); P = [{"prompt_token_ids": x} for x in ids]
g = llm.generate(P, SamplingParams(temperature=0.0, max_tokens=768, stop_token_ids=stop, skip_special_tokens=False), use_tqdm=False)
print(f"greedy {time.time() - t0:.0f}s", flush=True)
s = llm.generate(P, SamplingParams(n=7, max_tokens=768, stop_token_ids=stop, skip_special_tokens=False, seed=7, **cfg), use_tqdm=False)
with open(out, "w", encoding="utf-8") as fo:
    for r, a, b in zip(rows, g, s):
        src = r["prompt"].split(":", 1)[1].strip()
        cands = [{"k": 0, "greedy": True, "text": clean(a.outputs[0].text), "finish": a.outputs[0].finish_reason}] + [{"k": j + 1, "greedy": False, "text": clean(c.text), "finish": c.finish_reason} for j, c in enumerate(b.outputs)]
        fo.write(json.dumps({"id": r["id"], "task": FAM[r["task"]], "task6": r["task"], "lang": r["lang"], "switch": r.get("switch") if r["task"] in ("eng_hm", "m_hm") else None, "english": src, "prompt": r["prompt"], "candidates": cands}, ensure_ascii=False) + "\n")
print(f"DONE {len(rows)} rows {time.time() - t0:.0f}s", flush=True)
