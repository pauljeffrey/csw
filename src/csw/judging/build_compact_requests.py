"""Gemini batch request shards with the dedicated compact prompts.
python -m csw.judging.build_compact_requests --kind both --ids work/new_rows_ids.json --rows work/post_rows --out work/stage2_both --shard 900
kinds: both (hm + ood of one row, spans and edits), hm, ood.  Each shard dir holds gemini.jsonl; key = row id."""
from csw.paths import DATA_ROOT
import argparse, json, sys
from pathlib import Path
HERE = DATA_ROOT
from csw.judging import compact_judge as C
from csw.sft import devices as dv


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", choices=("both", "hm", "ood"), required=True); ap.add_argument("--ids", required=True); ap.add_argument("--rows", default="work/post_rows"); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=900); ap.add_argument("--max-out", type=int, default=1800); ap.add_argument("--rows-file", default="", help="jsonl of rows (e.g. corrupted variants) instead of post_rows"); a = ap.parse_args()
    ids = json.load(open(a.ids)); want = set(ids if isinstance(ids, list) else ids.keys()); rows = {}
    if a.rows_file:
        for l in open(a.rows_file, encoding="utf-8"):
            r = json.loads(l); rows[r["id"]] = r
    else:
        for f in sorted(Path(a.rows).glob("*.raw.jsonl")):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                if r["id"] in want: rows[r["id"]] = r
    lex = dv.Lexicon.load("work/lexicon.json"); sysm = {"both": C.SYS_BOTH, "hm": C.SYS_HM, "ood": C.SYS_OOD}[a.kind]; usr = {"both": C.user_both, "hm": C.user_hm, "ood": C.user_ood}[a.kind]; lines = []; skipped = 0
    for rid in ids if isinstance(ids, list) else list(ids):
        r = rows.get(rid)
        need = {"both": ("eng", "hm", "ood"), "hm": ("eng", "hm"), "ood": ("eng", "ood")}[a.kind]
        if not r or not all(isinstance(r.get(k), str) and r.get(k) for k in need): skipped += 1; continue
        lines.append({"key": rid, "request": {"contents": [{"role": "user", "parts": [{"text": usr(r, lex)}]}], "system_instruction": {"parts": [{"text": sysm}]},
                                              "generation_config": {"temperature": 0, "max_output_tokens": a.max_out, "response_mime_type": "application/json", "thinking_config": {"thinking_level": "low"}}}})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for s in range(0, len(lines), a.shard):
        d = out / f"stage_s{s // a.shard:02d}"; d.mkdir(exist_ok=True)
        with open(d / "gemini.jsonl", "w", encoding="utf-8") as f:
            for x in lines[s:s + a.shard]: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    (out / "kind.txt").write_text(a.kind); print("skipped (missing text)", skipped, "|", a.kind, len(lines), "requests in", -(-len(lines) // a.shard), "shards ->", out)


if __name__ == "__main__":
    main()
