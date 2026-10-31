"""Human-rated SSA-MTE rows as evaluator training / dev data (translation only).  The human z-score trains the MEAN of M1..M6 (one scalar per row, not six labels):
rec["h"] = 3.2 + 0.8*z clipped to [1,5].  Train rows whose English sentence also occurs in the SSA-MTE test or dev split are dropped (no leakage).
python -m csw.evaluator.build_human --raw work/ssa_mte_full --out work/human --train-per-lang 2500 --dev-per-lang 200"""
import argparse, hashlib, json, random
from pathlib import Path
ISO = {"Amharic": "amh", "Emakhuwa": "vmw", "Ewe": "ewe", "Hausa": "hau", "Igbo": "ibo", "Kikuyu": "kik", "Kinyarwanda": "kin", "Lingala": "lin", "Luo": "luo", "Nyanja": "nya", "Sesotho": "sot", "Swahili": "swh", "Twi": "twi", "Wolof": "wol", "Yoruba": "yor", "Zulu": "zul"}
EXCLUDE_SYSTEMS = {"NLLB"}          # NLLB weights are CC-BY-NC and NLLB output is not used in anything that could reach production
norm = lambda s: " ".join(s.lower().split())


def rows(raw, split, name):
    for l in open(Path(raw) / f"{split}_{name}.json", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("src_sent") and r.get("tgt_sent") and r.get("z_score") is not None and r.get("system") not in EXCLUDE_SYSTEMS: yield r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True); ap.add_argument("--train-per-lang", type=int, default=2500); ap.add_argument("--dev-per-lang", type=int, default=200); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rnd = random.Random(5); held = set()
    for name in ISO:
        for sp in ("test", "dev"): held |= {norm(r["src_sent"]) for r in rows(a.raw, sp, name)}   # (test/dev sentences are still excluded from train whatever system produced them)
    stats = {}
    for split, cap, fn in (("train", a.train_per_lang, "train.jsonl"), ("dev", a.dev_per_lang, "dev.jsonl")):
        recs = []
        for name, c in ISO.items():
            rs = [r for r in rows(a.raw, split, name) if split == "dev" or norm(r["src_sent"]) not in held]
            if split == "dev": rs = [r for r in rs if True]
            rnd.shuffle(rs)
            for r in rs[:cap]:
                recs.append({"id": f"h-{split}-{c}-{hashlib.sha1((r['src_sent'] + r['tgt_sent']).encode()).hexdigest()[:10]}", "lang": c, "pin": "", "eng": r["src_sent"], "m": r["tgt_sent"], "plan": [], "idioms": [], "draft": "", "y": [None] * 27,
                             "h": float(min(5.0, max(1.0, 3.2 + 0.8 * r["z_score"]))), "hz": r["z_score"], "tgts": ["m"]})
        with open(out / fn, "w", encoding="utf-8") as f:
            for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[split] = len(recs)
    print(stats, "held-out english sentences", len(held))


if __name__ == "__main__":
    main()
