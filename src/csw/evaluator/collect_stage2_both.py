"""Parse the strict compact BOTH-prompt results (new hm/ood rows) into evaluator records, keep the spans and edits, report usage.
python -m csw.evaluator.collect_stage2_both --dir work/stage2_both --unl work/qe_data/unlabelled.jsonl --out work/stage2_both/records.jsonl"""
import argparse, collections, glob, json, re, statistics, sys
from pathlib import Path
from csw.judging import compact_judge as C


def parse(t):
    if not t: return None
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    try: return json.loads(t)
    except Exception: return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--unl", required=True); ap.add_argument("--out", required=True); ap.add_argument("--price-in", type=float, default=0.375); ap.add_argument("--price-out", type=float, default=1.875); a = ap.parse_args()
    unl = {json.loads(l)["id"]: json.loads(l) for l in open(a.unl, encoding="utf-8")}; st = collections.Counter(); out = []; tin = tout = 0
    for f in sorted(glob.glob(f"{a.dir}/results/stage_s*.raw.jsonl")):
        for l in open(f, encoding="utf-8"):
            o = json.loads(l); st["requests"] += 1; resp = o.get("response") or {}; um = resp.get("usageMetadata") or {}
            tin += um.get("promptTokenCount", 0); tout += um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)
            try: txt = "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"] if "text" in p and not p.get("thought"))
            except Exception: st["no_text"] += 1; continue
            rep = parse(txt)
            if rep is None: st["unparsed"] += 1; continue
            if rep.get("r") == "n" or rep.get("nn") == 1 and rep.get("r") != "y": st["cannot_read"] += 1
            v = C.to_vec("both", rep)
            if v is None: st["bad_grades"] += 1; continue
            r = unl.get(o["key"])
            if not r: st["no_row"] += 1; continue
            out.append(dict(r, y=v, yj={"flash_strict": v}, judges=["flash_strict"], tgts=["hm", "ood"], grader="strict", spans=rep.get("e") or [], edits=rep.get("ed") or [], nn=rep.get("nn", 0), judge_reply_r=rep.get("r")))
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(st), "usable", len(out), f"| tokens in {tin:,} out {tout:,} ({tout/max(1,st['requests']):.0f}/row) -> cost ${(tin*a.price_in+tout*a.price_out)/1e6:.2f}")
    def md(r, lo, hi): x = [r["y"][i] for i in range(lo, hi) if r["y"][i] is not None]; return sum(x) / len(x) if x else None
    for name, lo, hi in (("hm (H1-H9)", 6, 15), ("ood (O1-O7)", 15, 22)):
        ms = [m for m in (md(r, lo, hi) for r in out) if m is not None]
        print(f"  {name}: mean {statistics.mean(ms):.2f}, sd {statistics.pstdev(ms):.2f}, below 3: {100*sum(m<3 for m in ms)/len(ms):.0f}%, below 2: {100*sum(m<2 for m in ms)/len(ms):.1f}%, 4.5+: {100*sum(m>=4.5 for m in ms)/len(ms):.0f}%")
    print("  rows with edits:", sum(bool(r['edits']) for r in out), "| mean edits per edited row:", round(statistics.mean([len(r['edits']) for r in out if r['edits']] or [0]), 1), "| rows with spans:", sum(bool(r['spans']) for r in out))
    print("  languages:", len({r['lang'] for r in out}))


if __name__ == "__main__":
    main()
