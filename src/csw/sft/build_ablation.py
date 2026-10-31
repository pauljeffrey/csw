"""Three SFT arms on the SAME stratified subsample of the SFT build, differing only in the format of the ood tasks (eng_ood, m_ood, hm_ood); hm, eng_m and replay are identical.
    python -m csw.sft.build_ablation --sft work/csw_sft_v3 --rows work/post_rows --out work/ablation --frac 0.2
  A  standard        <yor><dev>idiom one | idiom two</dev> text
  B  glossed devices <yor><dev>idiom one (expression: meaning) | idiom two (term: meaning)</dev> text                    (the generator's own English glosses)
  C  draft           <yor><draft>English draft (eng_initial)</draft><dev>idiom one | idiom two</dev> text; the prompt starts with `<draft> ` to ask for the draft. A random half of the ood
                     examples carry the flag (and the draft), the other half are identical to A, so one model learns both modes and the draft can be switched off at inference.
Test/val: A and B use the standard prompts; C writes test_on.jsonl (flag on every ood prompt) and test_off.jsonl (standard).  Every arm keeps the reference completion untouched as `completion`;
scoring (csw.sft.eval_chrf) strips tags, draft and devices and compares the final text only, so the formats do not distort chrF.  Examples with a missing gloss or eng_initial stay standard."""
from __future__ import annotations
import argparse, collections, json, random, re, sys
from pathlib import Path
from csw.sft import fmt2
from csw.sft import devices as dv

OOD_TASKS = ("eng_ood", "m_ood", "hm_ood")


def load_rows(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*.raw.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); out[r["id"]] = r
    return out


def gloss_completion(comp: str, row: dict) -> str | None:
    o = dv.ood_devices(row); pairs = dict(zip(o["devices"], o["glossed"])) if o.get("glossed") else {}
    m = re.search(r"<dev>(.*?)</dev>", comp, flags=re.S)
    if not m or not pairs: return None
    items = [x.strip() for x in m.group(1).split(" | ")]
    if not all(i in pairs for i in items): return None
    return comp[:m.start()] + "<dev>" + " | ".join(pairs[i] for i in items) + "</dev>" + comp[m.end():]


def draft_completion(comp: str, row: dict) -> str | None:
    e = " ".join((row.get("eng_initial") or "").split())
    if len(e) < 10 or "<dev>" not in comp: return None
    return comp.replace("<dev>", f"{fmt2.DRAFT_O}{e}{fmt2.DRAFT_C}<dev>", 1)


def subsample(rows: list[dict], frac: float, rnd: random.Random) -> list[dict]:
    by = collections.defaultdict(list)
    for r in rows: by[(r["task"], r["lang"])].append(r)
    out = []
    for k, v in by.items():
        n = max(1, round(len(v) * frac)); out += rnd.sample(v, min(n, len(v)))
    rnd.shuffle(out); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sft", required=True); ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.2); ap.add_argument("--draft-share", type=float, default=0.5); ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--arms", default="ABC", help="which arms to write, e.g. C for the final build"); ap.add_argument("--draft-tasks", default=",".join(OOD_TASKS), help="tasks that may carry the draft in arm C (final build: eng_ood)"); a = ap.parse_args()
    draft_tasks = set(a.draft_tasks.split(","))
    sft, out = Path(a.sft), Path(a.out); rnd = random.Random(a.seed); rows = load_rows(Path(a.rows)); st = collections.Counter()
    ld = lambda n: [json.loads(l) for l in open(sft / n, encoding="utf-8") if l.strip()]
    train, val, test = ld("train.jsonl"), ld("val.jsonl"), ld("test.jsonl")
    train, val = subsample(train, a.frac, rnd), subsample(val, a.frac, rnd)
    for arm in a.arms: (out / arm).mkdir(parents=True, exist_ok=True)

    def variant(e: dict, arm: str, on: bool | None):
        if e["task"] not in OOD_TASKS: return e
        rid = e["id"].rsplit(":", 1)[0]; row = rows.get(rid) or rows.get(re.sub(r"-fx[fl]$", "", rid))     # judge-corrected copies use their parent row's draft and glosses
        if row is None: st[f"{arm}_no_row"] += 1; return e
        if arm == "B":
            c = gloss_completion(e["completion"], row)
            if c is None: st["B_no_gloss"] += 1; return e
            st["B_glossed"] += 1; return dict(e, completion=c)
        if arm == "C":
            if e["task"] not in draft_tasks: st["C_task_standard"] += 1; return e
            use = rnd.random() < a.draft_share if on is None else on
            if not use: st["C_off"] += 1; return e
            c = draft_completion(e["completion"], row)
            if c is None: st["C_no_draft"] += 1; return e
            st["C_on"] += 1; return dict(e, prompt=f"{fmt2.DRAFT_O} " + e["prompt"], completion=c)
        return e

    def write(path: Path, items):
        with open(path, "w", encoding="utf-8") as f:
            for e in items: f.write(json.dumps(e, ensure_ascii=False) + "\n")
    for arm in a.arms:
        write(out / arm / "train.jsonl", [variant(e, arm, None) for e in train]); write(out / arm / "val.jsonl", [variant(e, arm, None) for e in val])
    if "A" in a.arms: write(out / "A" / "test.jsonl", test)
    if "B" in a.arms: write(out / "B" / "test.jsonl", [variant(e, "B", None) for e in test])
    if "C" in a.arms: write(out / "C" / "test_on.jsonl", [variant(e, "C", True) for e in test]); write(out / "C" / "test_off.jsonl", test)
    info = {"frac": a.frac, "train": len(train), "val": len(val), "test": len(test), "stats": dict(st), "ood_train": sum(e["task"] in OOD_TASKS for e in train)}
    (out / "ablation_info.json").write_text(json.dumps(info, indent=1)); print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
