"""Validate and deduplicate the 8 candidates of every row.  python postprocess.py IN.jsonl OUT.jsonl"""
import json, sys, re, collections, importlib.util, unicodedata
sp = importlib.util.spec_from_file_location("ec", "/workspace/csw_hs/eval_chrf.py"); ec = importlib.util.module_from_spec(sp); sp.loader.exec_module(ec)
inp, outp = sys.argv[1:3]
TAGS = re.compile(r"^((?:<[a-z]{3,5}>)+)")


def norm(t): return re.sub(r"[^\w ]", "", unicodedata.normalize("NFC", t).lower()).split()
def grams(t, n=5): s = " ".join(norm(t)); return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}
def jacc(a, b): return len(a & b) / max(1, len(a | b))
def loopy(t):
    w = norm(t); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(g) > 8 and 1 - len(set(g)) / len(g) > 0.25


def parse(row, c):
    """-> (final_text, devices, tags, reason or None)"""
    t = c["text"]
    if c["finish"] != "stop": return None, [], [], "truncated"
    if row["task"] == "m_text":
        m = re.match(r"^([a-z]{3,5})\s+(.*)$", t, re.S)
        if not m or m.group(1) != row["lang"]: return None, [], [], "bad_language_tag"
        return m.group(2).strip(), [], [m.group(1)], None
    text, devs, _, has = ec.parts(t); tags = re.findall(r"<([a-z]{3,5})>", TAGS.match(t).group(1)) if TAGS.match(t) else []
    if not has or len(devs) < 2 or not text: return None, devs, tags, "bad_format"
    if not tags or tags[0] != row["lang"]: return None, devs, tags, "bad_base_tag"
    if row["task"] == "hm" and (len(tags) < 2 or tags[1] != row["switch"]): return None, devs, tags, "bad_switch_tag"
    return text, devs, tags, None


stats = collections.Counter(); rows_out = []
for l in open(inp):
    row = json.loads(l); kept = []; dropped = collections.Counter(); le = len(row["english"])
    for c in row["candidates"]:
        text, devs, tags, why = parse(row, c)
        if not why and loopy(text): why = "loop"
        if not why and not (0.35 * le <= len(text) <= 4.0 * le): why = "length"
        if not why:
            g = grams(text)
            if any(jacc(g, k["_g"]) >= 0.9 for k in kept): why = "duplicate"
        if why: dropped[why] += 1; continue
        kept.append({"k": c["k"], "greedy": c["greedy"], "text": text, "devices": devs, "tags": tags, "raw": c["text"], "_g": g})
    for k in kept: k.pop("_g")
    for k, v in dropped.items(): stats[k] += v
    stats["kept"] += len(kept); stats["candidates"] += len(row["candidates"]); stats["greedy_kept"] += any(k["greedy"] for k in kept)
    row["candidates"] = kept; row["n_unique"] = len(kept); row["dropped"] = dict(dropped); rows_out.append(row)
with open(outp, "w", encoding="utf-8") as fo:
    for r in rows_out: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
n = len(rows_out); print(inp, "rows", n, "mean unique per row %.2f" % (stats["kept"] / max(1, n)), "greedy kept %.1f%%" % (100 * stats["greedy_kept"] / max(1, n)), dict(stats))
