"""Pool of UNUSED cleared pairs for the evaluator's low end: candidates (build_replay2 --cap 3500) minus everything already used (replay4, label_more), same licence / decon rules as build_replay4 but a relaxed length-ratio filter
(misaligned pairs are exactly what we want), restricted to the languages AfriCOMET-1.1 supports.
python -m csw.sft.build_nb_pool --cand r4/cand.jsonl --used replay4.jsonl label_more_pairs.jsonl --out r4/nb_pool.jsonl"""
import argparse, collections, hashlib, json
SUPPORTED = {"pcm", "yor", "hau", "ibo", "fon", "ewe", "twi", "afr", "zul", "sna", "nya", "xho", "sot", "kin", "tsn", "orm", "wol", "som", "swh"}      # AfriCOMET-1.1 languages that are also csw languages


def sha(t): return hashlib.sha1(" ".join(t.lower().split()).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cand", required=True); ap.add_argument("--used", nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    used = set()
    for f in a.used:
        for l in open(f, encoding="utf-8"):
            x = json.loads(l); used.add((sha(x["eng"]), x["lang"]))
    seen = set(); st = collections.Counter(); out = []
    for l in open(a.cand, encoding="utf-8"):
        c = json.loads(l); lang = c["iso"]
        if lang not in SUPPORTED: st["language_not_supported_by_africomet"] += 1; continue
        s = c["source_dataset"]
        if c["decon_flag"] or c["licence"] in ("non_commercial", "share_alike") or "finetranslations" in s.lower() or "SSA-MTE" in s: st["licence_or_flag"] += 1; continue
        e, m = c["english"], c["target"]; k = (sha(e), lang)
        if k in used or k in seen: st["already_used_or_duplicate"] += 1; continue
        we, wm = e.split(), m.split()
        if len(we) < 3 or len(wm) < 2 or len(e) > 600 or len(m) > 900 or not (0.2 <= len(m) / max(1, len(e)) <= 5) or e.strip().lower() == m.strip().lower(): st["length_sanity"] += 1; continue
        seen.add(k); out.append({"lang": lang, "eng": e, "m": m, "origin": s, "licence": c["licence"]})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per = collections.Counter(r["lang"] for r in out); print(dict(st), "pool", len(out), dict(sorted(per.items())))


if __name__ == "__main__":
    main()
