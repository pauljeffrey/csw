"""Flash-labelled human replay pairs -> evaluator training records (translation field only: M1..M6 labelled, every other dimension masked; tgts = ["m"]).
python -m csw.evaluator.build_replay_records --labels work/replay_flash/labels.jsonl --base work/qe_data_v2 --out work/qe_data_v2r
English sentences that occur in the evaluator's val or test rows are dropped (no overlap with the held-out sets)."""
import argparse, hashlib, json, shutil
from pathlib import Path


def sha(t): return hashlib.sha1(" ".join(t.lower().split()).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--labels", required=True); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    base, out = Path(a.base), Path(a.out); out.mkdir(parents=True, exist_ok=True); protect = set()
    for s in ("val", "test"):
        for l in open(base / f"{s}.jsonl", encoding="utf-8"): protect.add(json.loads(l)["sha"])
    new = []; drop = 0
    for l in open(a.labels, encoding="utf-8"):
        x = json.loads(l)
        if sha(x["eng"]) in protect: drop += 1; continue
        y = [None] * 27; y[:6] = [float(v) if v is not None else None for v in x["g"]]
        new.append({"id": x["id"], "lang": x["lang"], "pin": "", "eng": x["eng"], "m": x["m"], "plan": [], "idioms": [], "draft": "", "y": y, "yj": {"flash": y}, "judges": ["flash"], "tgts": ["m"], "sha": sha(x["eng"]), "origin": x.get("origin")})
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for l in open(base / "train.jsonl", encoding="utf-8"): f.write(l)
        for r in new: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for n in ("val.jsonl", "test.jsonl", "unlabelled.jsonl"): shutil.copy(base / n, out / n)
    print(f"replay records added {len(new)} (dropped {drop} whose English is in val/test) -> {out}/train.jsonl")


if __name__ == "__main__":
    main()
