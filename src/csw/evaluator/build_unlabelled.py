"""Unlabelled pool for the semi-supervised QE: generated rows that no judge graded and whose English sentence is not in the labelled val / test sets.
python -m csw.evaluator.build_unlabelled --data work/qe_data --rows work/post_rows --out work/qe_data/unlabelled.jsonl --per-lang 1100"""
from csw.paths import DATA_ROOT
import argparse, collections, hashlib, json, random, sys
from pathlib import Path
HERE = DATA_ROOT
from csw.sft import devices as dv


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="work/qe_data"); ap.add_argument("--rows", default="work/post_rows"); ap.add_argument("--out", default="work/qe_data/unlabelled.jsonl")
    ap.add_argument("--per-lang", type=int, default=1100); ap.add_argument("--seed", type=int, default=3); a = ap.parse_args()
    seen, protect = set(), set()
    for s in ("train", "val", "test"):
        for l in open(Path(a.data) / f"{s}.jsonl", encoding="utf-8"):
            r = json.loads(l); seen.add(r["id"])
            if s != "train": protect.add(r["sha"])
    lex = dv.Lexicon.load("work/lexicon.json"); by = collections.defaultdict(list); st = collections.Counter()
    for f in sorted(Path(a.rows).glob("*.raw.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); st["rows"] += 1
            if r["id"] in seen: st["labelled"] += 1; continue
            if not all(isinstance(r.get(k), str) and r.get(k) for k in ("eng", "m_text", "hm", "ood")): st["missing_field"] += 1; continue
            sha = hashlib.sha1(" ".join(r["eng"].lower().split()).encode()).hexdigest()
            if sha in protect: st["english_in_val_or_test"] += 1; continue
            by[r["lang"]].append((r, sha))
    rnd = random.Random(a.seed); out = []
    for lang, v in sorted(by.items()):
        rnd.shuffle(v)
        for r, sha in v[: a.per_lang]:
            try: plan = dv.hm_devices(r, lex)["devices"]
            except Exception: plan = []
            idioms = [d for d in (r.get("devices") or []) if isinstance(d, str)][:8]
            out.append({"id": r["id"], "lang": lang, "pin": (r.get("scenario") or {}).get("switch") or "", "eng": r["eng"], "m": r["m_text"], "hm": r["hm"], "ood": r["ood"],
                        "plan": plan[:10], "idioms": idioms, "draft": " ".join((r.get("eng_initial") or "").split()), "sha": sha})
    rnd.shuffle(out)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(st), "available per language", {k: len(v) for k, v in sorted(by.items())}, "-> wrote", len(out))


if __name__ == "__main__":
    main()
