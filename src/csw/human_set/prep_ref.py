"""Reference completions of the test set as one-candidate rows (for evaluator scoring)."""
import json, sys, importlib.util, re
sp = importlib.util.spec_from_file_location("ec", "/workspace/csw_hs/eval_chrf.py"); ec = importlib.util.module_from_spec(sp); sp.loader.exec_module(ec)
FAM = {"eng_hm": "hm", "m_hm": "hm", "eng_ood": "ood", "m_ood": "ood", "hm_ood": "ood"}
with open("/workspace/csw_hs/out/eval_ref.jsonl", "w", encoding="utf-8") as fo:
    for l in open("/workspace/csw_hs/data/test_off.jsonl"):
        r = json.loads(l); text, devs, _, has = ec.parts(r["completion"])
        fo.write(json.dumps({"id": r["id"], "task": FAM[r["task"]], "task6": r["task"], "lang": r["lang"], "switch": r.get("switch") if r["task"] in ("eng_hm", "m_hm") else None, "english": r["prompt"].split(":", 1)[1].strip(), "candidates": [{"k": 0, "greedy": False, "text": text, "devices": devs, "tags": re.findall(r"<([a-z]{3,5})>", r["completion"].split("</dev>")[0])}]}, ensure_ascii=False) + "\n")
