"""Fine-tuning stage 2 data: new strict translation labels (natural-bad pairs + chains) and the hm / ood chains, plus a replay sample of stage 1's training set.  Held-out chains and a small strict dev slice go to val.
python -m csw.evaluator.build_cft2 --base work/cft1 --out work/cft2 --tr work/label_nb/labels.jsonl work/chains_tr/labels.jsonl --hmood work/chains_hm/records.jsonl work/chains_ood/records.jsonl --replay 20000"""
import argparse, json, random, shutil
from pathlib import Path


def tr_record(x):
    y = [None] * 27; y[:6] = [float(v) if v is not None else None for v in x["g"]]
    return {"id": x["id"], "lang": x["lang"], "pin": "", "eng": x["eng"], "m": x["m"], "plan": [], "idioms": [], "draft": "", "y": y, "yj": {"flash_strict": y}, "judges": ["flash_strict"], "tgts": ["m"], "grader": "strict",
            "why": x.get("why"), "chain_op": x.get("op"), "chain_level": x.get("level"), "chain_base": x.get("base"), "heldout": bool(x.get("heldout"))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tr", nargs="+", required=True); ap.add_argument("--hmood", nargs="*", default=[])
    ap.add_argument("--drop-ops", nargs="*", default=[], help="corruption operators whose corrupted (level>0) items are left out because the judge does not grade them reliably"); ap.add_argument("--replay", type=int, default=20000); ap.add_argument("--dev-frac", type=float, default=0.03); ap.add_argument("--seed", type=int, default=6); a = ap.parse_args()
    rnd = random.Random(a.seed); base = Path(a.base); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); new = []
    drop = set(a.drop_ops); dropped = 0
    for f in a.tr:
        for l in open(f, encoding="utf-8"):
            x = json.loads(l)
            if x.get("level") and x.get("op") in drop: dropped += 1; continue
            new.append(tr_record(x))
    print("dropped corrupted items of unreliable operators:", dropped)
    for f in a.hmood: new += [json.loads(l) for l in open(f, encoding="utf-8")]
    held = [r for r in new if r.get("heldout")]; new = [r for r in new if not r.get("heldout")]; rnd.shuffle(new); dev = new[: int(a.dev_frac * len(new))]; new = new[len(dev):]
    old = [json.loads(l) for l in open(base / "train.jsonl", encoding="utf-8")]; rnd.shuffle(old); train = old[: a.replay] + new; rnd.shuffle(train)
    val = [json.loads(l) for l in open(base / "val.jsonl", encoding="utf-8")] + dev
    for name, rows in (("train", train), ("val", val), ("heldout_chains", held)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for n in ("test.jsonl", "unlabelled.jsonl"):
        if (base / n).exists(): shutil.copy(base / n, out / n)
    print(f"train {len(train)} = {min(a.replay, len(old))} replay + {len(new)} new; val {len(val)} (+{len(dev)} new-type dev rows); held-out chain items {len(held)}")


if __name__ == "__main__":
    main()
