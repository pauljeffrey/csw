"""Keep the old rows (tagged when their hm is wrong) and add NEW rows with the regenerated hm.
  python -m csw.datagen.merge_rehm out_gb10 yor pcm
Output <lang>.v2.jsonl:
  - every original row, with `quality_tags` (e.g. ["hm_over_mixed_v1"]) when its hm breaks the item-3 definition
    (kept on purpose as negatives / preference pairs),
  - one NEW row per regenerated hm: id = <orig>-hm2, hm = new hm, parent_id, `hm_rejected` = the old hm, tags ["hm_v2"].
Heuristics (no model needed): english_share = share of hm words (ascii, >2 letters) that also occur in the plain `eng`;
over-mixed if share > 0.30; for Pidgin-like rows 'tag_only' if hm is >0.85 similar to m_text (only tags were added)."""
import difflib, json, re, sys, statistics as st
from pathlib import Path
W = re.compile(r"[^\W\d_]+", re.U)
tk = lambda t: [w.lower() for w in W.findall(t)]
jl = lambda p: [json.loads(x) for x in Path(p).read_text(encoding="utf-8").split("\n") if x.strip()] if Path(p).exists() else []


def metrics(eng, m_text, hm):
    e = set(tk(eng)); h = tk(hm); n = max(1, len(h))
    run = best = 0
    for w in h:
        run = run + 1 if (w in e and w.isascii() and len(w) > 2) else 0
        best = max(best, run)
    return {"english_share": round(sum(1 for w in h if w in e and w.isascii() and len(w) > 2) / n, 3),
            "longest_en_run": best, "sim_to_m_text": round(difflib.SequenceMatcher(None, tk(m_text), h).ratio(), 3)}


def main(od, langs):
    for lang in langs:
        raw, new = jl(f"{od}/{lang}.raw.jsonl"), {r["id"]: r for r in jl(f"{od}/{lang}.rehm.jsonl")}
        out, tagged, n_new, before, after = [], 0, 0, [], []
        for r in raw:
            m = metrics(r["eng"], r["m_text"], r["hm"]); r = dict(r, hm_metrics=m, quality_tags=[])
            pidginish = lang == "pcm"
            if (not pidginish and m["english_share"] > 0.30) or (pidginish and m["sim_to_m_text"] > 0.85):
                r["quality_tags"].append("hm_over_mixed_v1" if not pidginish else "hm_tags_only_v1"); tagged += 1
            out.append(r)
            if r["id"] in new:
                nr = dict(r, id=r["id"] + "-hm2", parent_id=r["id"], hm_rejected=r["hm"], hm=new[r["id"]]["hm_new"], quality_tags=["hm_v2"])
                nr["hm_metrics"] = metrics(r["eng"], r["m_text"], nr["hm"]); out.append(nr); n_new += 1
                before.append(m); after.append(nr["hm_metrics"])
        Path(f"{od}/{lang}.v2.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
        f = lambda L, k: round(st.mean(x[k] for x in L), 2) if L else None
        print(f"{lang}: {len(raw)} original rows ({tagged} tagged wrong) + {n_new} new rows | english_share {f(before,'english_share')} -> {f(after,'english_share')} | "
              f"longest run {f(before,'longest_en_run')} -> {f(after,'longest_en_run')} | sim_to_m_text {f(before,'sim_to_m_text')} -> {f(after,'sim_to_m_text')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
