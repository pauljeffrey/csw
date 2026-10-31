"""Data for a continuous fine-tuning stage: new strict-graded records + a replay sample of the old training data (so nothing is forgotten) + a strict dev slice.
python -m csw.evaluator.build_cft --base work/qe_data_v3 --new work/stage2_both/records.jsonl [--new work/..] --out work/cft1 --old-gen 12000 --old-tr 8000 --strict-dev 500"""
import argparse, json, random, shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--new", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--old-gen", type=int, default=12000); ap.add_argument("--old-tr", type=int, default=8000); ap.add_argument("--strict-dev", type=int, default=500); ap.add_argument("--seed", type=int, default=5); a = ap.parse_args()
    rnd = random.Random(a.seed); base = Path(a.base); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    old = [json.loads(l) for l in open(base / "train.jsonl", encoding="utf-8")]; gen = [r for r in old if not r.get("tgts")]; tr = [r for r in old if r.get("tgts")]
    rnd.shuffle(gen); rnd.shuffle(tr); new = []
    for f in a.new: new += [json.loads(l) for l in open(f, encoding="utf-8")]
    rnd.shuffle(new); dev_new, new = new[: a.strict_dev], new[a.strict_dev:]
    for r in old: r.setdefault("grader", "std")
    train = gen[: a.old_gen] + tr[: a.old_tr] + new; rnd.shuffle(train)
    val = [json.loads(l) for l in open(base / "val.jsonl", encoding="utf-8")] + dev_new
    for name, rows in (("train", train), ("val", val)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for n in ("test.jsonl", "unlabelled.jsonl"):
        if (base / n).exists(): shutil.copy(base / n, out / n)
    print(f"train {len(train)} rows: {min(a.old_gen, len(gen))} old generated + {min(a.old_tr, len(tr))} old translation + {len(new)} new strict; val {len(val)} (incl. {len(dev_new)} strict dev rows)")


if __name__ == "__main__":
    main()
