"""Parse M1..M6 batch results (8 pairs per request, written by batch_driver as results/*.raw.jsonl) into labels.jsonl.
python -m csw.judging.collect_m_raw --dir work/label_nb"""
import argparse, glob, json, re, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args(); d = Path(a.dir)
    meta = json.loads((d / "meta.json").read_text()); pairs = {}
    for l in open(d / "pairs.jsonl", encoding="utf-8"):
        x = json.loads(l); pairs[x["id"]] = x
    n = bad = req = 0; tin = tout = 0
    with open(d / "labels.jsonl", "w", encoding="utf-8") as fo:
        for f in sorted(glob.glob(str(d / "results" / "stage_s*.raw.jsonl"))):
            for l in open(f, encoding="utf-8"):
                o = json.loads(l); req += 1; resp = o.get("response") or {}; um = resp.get("usageMetadata") or {}; tin += um.get("promptTokenCount", 0); tout += um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)
                try: txt = "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"] if "text" in p and not p.get("thought")); g = json.loads(re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()).get("g")
                except Exception: g = None
                ids = meta.get(o["key"])
                if not ids or not isinstance(g, list) or len(g) != len(ids): bad += 1; continue
                for i, v in zip(ids, g):
                    if isinstance(v, list) and len(v) == 6 and all(x is None or (isinstance(x, (int, float)) and 1 <= x <= 5) for x in v):
                        p = pairs[i]; fo.write(json.dumps({"id": i, "lang": p["lang"], "eng": p["eng"], "m": p["m"], "origin": p.get("origin"), "why": p.get("why"), "qe": p.get("qe"), "op": p.get("op"), "level": p.get("level"), "base": p.get("base"), "heldout": p.get("heldout"), "g": v}, ensure_ascii=False) + "\n"); n += 1
    print(f"requests {req}; unusable requests {bad}; labelled pairs {n}; tokens in {tin:,} out {tout:,}; cost ${(tin * 0.375 + tout * 1.875) / 1e6:.2f}")


if __name__ == "__main__":
    main()
