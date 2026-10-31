"""Chain batches (single-field compact prompts) -> evaluator records.  python -m csw.evaluator.collect_compact --kind hm --dir work/chains_hm --rows work/chains/hm_rows.jsonl --out work/chains_hm/records.jsonl"""
from csw.paths import DATA_ROOT
import argparse, glob, json, re, sys
from pathlib import Path
HERE = DATA_ROOT
from csw.judging import compact_judge as C
from csw.sft import devices as dv


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", choices=("hm", "ood"), required=True); ap.add_argument("--dir", required=True); ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    rows = {}
    for l in open(a.rows, encoding="utf-8"):
        r = json.loads(l); rows[r["id"]] = r
    lex = dv.Lexicon.load("work/lexicon.json"); out = []; req = bad = 0; tin = tout = 0
    for f in sorted(glob.glob(f"{a.dir}/results/stage_s*.raw.jsonl")):
        for l in open(f, encoding="utf-8"):
            o = json.loads(l); req += 1; resp = o.get("response") or {}; um = resp.get("usageMetadata") or {}; tin += um.get("promptTokenCount", 0); tout += um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)
            try: txt = "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"] if "text" in p and not p.get("thought")); rep = json.loads(re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip())
            except Exception: bad += 1; continue
            v = C.to_vec(a.kind, rep); r = rows.get(o["key"])
            if v is None or r is None: bad += 1; continue
            try: plan = dv.hm_devices(r, lex)["devices"]
            except Exception: plan = []
            out.append({"id": r["id"], "lang": r["lang"], "pin": (r.get("scenario") or {}).get("switch") or "", "eng": r["eng"], "m": r["m_text"], "hm": r["hm"], "ood": r["ood"], "plan": plan[:10], "idioms": [d for d in (r.get("devices") or []) if isinstance(d, str)][:8], "draft": "",
                        "y": v, "yj": {"flash_strict": v}, "judges": ["flash_strict"], "tgts": [a.kind], "grader": "strict", "chain_base": r.get("chain_base"), "chain_op": r.get("chain_op"), "chain_level": r.get("chain_level"), "heldout": r.get("heldout")})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"requests {req}; unusable {bad}; records {len(out)}; tokens in {tin:,} out {tout:,}; cost ${(tin * 0.375 + tout * 1.875) / 1e6:.2f}")


if __name__ == "__main__":
    main()
