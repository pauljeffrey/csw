"""Fresh English for the RL prompts, taken ONLY from the annotated English source datasets of the organisation named by CSW_ENGLISH_SOURCE_ORG and only the parts the triage model marks as easy to translate.

Run on a GPU/CPU box, low priority:   python -m csw.judging.fetch_english --out english_fresh.jsonl [--exclude-sha1 ours.txt] [--scale 1.0]

For every source the parquet shards are downloaded one at a time, scanned, and deleted again.  A text unit passes when its triage struct says
    recommended_pipeline == 'translate into African languages'   technical_depth in {plain language, school-level}   translation_difficulty in {very easy, easy}
    sensitivity == 'none'   english_form_dependent False   contains_personal_data False   core_labels_high_confidence True
Chat sets: the labels belong to an assistant turn (+ the turns before it), so both that assistant reply and the user message before it are used.
From each passing text, windows of whole sentences are cut with the length mix of the generator (1-3 / 2-5 / 6-12 / 13-20 sentences) and checked: ascii English,
no markup / code / URLs / contact data, plain sentences (<=35 words on average, short words), no profanity.  Output rows: english, n_sent, source, register.
"""
from __future__ import annotations

from csw.paths import env_value
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import unicodedata

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

TOK = env_value("HF_WRITE_TOKEN")
CACHE = os.environ.get("CSW_HF_CACHE", "hf_tmp")

# repo, kind, register, max shards, max windows from the source (at --scale 1)
ORG = env_value("CSW_ENGLISH_SOURCE_ORG")
SUFFIX = env_value("CSW_ENGLISH_SOURCE_SUFFIX", "")
SOURCES = json.load(open(env_value("CSW_ENGLISH_SOURCES_FILE"), encoding="utf-8")) if env_value("CSW_ENGLISH_SOURCES_FILE") else []    # [[repo, kind, register, max shards, max windows], ...]
OK_DEPTH = {"plain language, no jargon", "school-level terminology"}
OK_DIFF = {"very easy", "easy"}
BANDS = [((1, 3), 0.25), ((2, 5), 0.35), ((6, 12), 0.28), ((13, 20), 0.12)]
BAD = re.compile(r"(https?://|www\.|@\w+\.|\b\d{3}[-. ]\d{3,4}[-. ]\d{4}\b|```|<[a-z/][^>]*>|\{|\}|\\|\[\s*(WP|IP|EU|CW|TT|PM)\s*\]|^\s*[#>*\-\d]+[.)]?\s)", re.I | re.M)
PROFANE = re.compile(r"\b(fuck|shit|bitch|cunt|dick|pussy|nigg|whore|slut|rape|porn|sex|nude|kill yourself|suicide)\w*", re.I)
SENT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z\"'(])")
STOP = set("the a an and or but of to in on at for with from by is are was were be been it this that i you he she we they my your his her our their not no".split())
QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", " ": " "})


def norm(t: str) -> str:
    return " ".join(unicodedata.normalize("NFC", t).translate(QUOTES).split())


def passes_labels(t: dict) -> bool:
    return (t.get("recommended_pipeline") == "translate into African languages" and t.get("technical_depth") in OK_DEPTH and t.get("translation_difficulty") in OK_DIFF
            and t.get("sensitivity") == "none" and not t.get("english_form_dependent") and not t.get("contains_personal_data") and bool(t.get("core_labels_high_confidence")))


def good_window(w: str) -> bool:
    if len(w) < 25 or len(w) > 3000 or BAD.search(w) or PROFANE.search(w):
        return False
    if sum(ord(c) < 128 for c in w) / len(w) < 0.995:
        return False
    words = re.findall(r"[A-Za-z']+", w)
    if len(words) < 8 or sum(x.lower() in STOP for x in words) / len(words) < 0.28:
        return False
    if sum(map(len, words)) / len(words) > 5.3 or sum(c.isdigit() for c in w) / len(w) > 0.05:
        return False
    sents = [s for s in SENT.split(w) if s]
    return all(len(s.split()) <= 45 for s in sents) and len(words) / max(1, len(sents)) <= 30


def windows(text: str, rnd: random.Random, k: int) -> list[tuple[str, int]]:
    paras = [p for p in re.split(r"\n{2,}", text) if p.strip()]
    sents = []
    for p in paras:
        sents += [s.strip() for s in SENT.split(norm(p)) if s.strip()]
    out = []
    if not sents:
        return out
    for _ in range(k):
        r, acc = rnd.random(), 0.0
        for (lo, hi), share in BANDS:
            acc += share
            if r <= acc:
                break
        n = rnd.randint(lo, hi)
        if len(sents) < n:
            n = len(sents)
            if n < lo:
                continue
        start = rnd.randint(0, len(sents) - n)
        w = " ".join(sents[start:start + n])
        if good_window(w) and re.search(r"[.!?][\"']?$", w):
            out.append((w, n))
    return out


def units(row: dict, kind: str):
    """yield (text, label_struct) units of one dataset row"""
    if kind == "text":
        t = row["triage"]
        if t and passes_labels(t):
            yield row["text"], t
        return
    msgs = row["messages"]
    for t in row["triage_turns"] or []:
        if not passes_labels(t):
            continue
        i = t.get("message_index")
        if i is None or i >= len(msgs):
            continue
        yield msgs[i]["content"], t                       # the assistant reply
        if i > 0 and msgs[i - 1]["role"] == "user":
            yield msgs[i - 1]["content"], t               # the request before it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--exclude-sha1"); ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only")
    a = ap.parse_args()
    excl = set(open(a.exclude_sha1).read().split()) if a.exclude_sha1 else set()
    api = HfApi(token=TOK); rnd = random.Random(a.seed)
    seen, n_out = set(), 0
    stats = {}
    with open(a.out, "w", encoding="utf-8") as fo:
        for repo, kind, reg, max_shards, cap in SOURCES:
            if a.only and a.only not in repo:
                continue
            cap = int(cap * a.scale); got = 0; scanned = 0
            files = sorted(f for f in api.list_repo_files(f"{ORG}/{repo}", repo_type="dataset") if f.endswith(".parquet"))[:max_shards]
            for fn in files:
                if got >= cap:
                    break
                p = hf_hub_download(f"{ORG}/{repo}", fn, repo_type="dataset", token=TOK, cache_dir=CACHE)
                t = pq.read_table(p, columns=["messages", "triage_turns"] if kind == "chat" else ["text", "triage"])
                for row in t.to_pylist():
                    scanned += 1
                    for text, lab in units(row, kind):
                        # long texts give a few windows, short chat turns one
                        for w, nsent in windows(text, rnd, 2 if len(text) > 1500 else 1):
                            key = hashlib.sha1(w.lower().encode()).hexdigest()
                            head = w.lower()[:90]
                            if key in excl or key in seen or head in seen:
                                continue
                            seen.add(key); seen.add(head)
                            fo.write(json.dumps({"id": f"{repo.removesuffix(SUFFIX)}:{key[:10]}", "english": w, "n_sent": nsent, "source": repo.replace(SUFFIX, ""),
                                                 "register": reg,
                                                 "domain": lab.get("domain"), "difficulty": lab.get("translation_difficulty")}, ensure_ascii=False) + "\n")
                            got += 1; n_out += 1
                            if got >= cap:
                                break
                        if got >= cap:
                            break
                    if got >= cap:
                        break
                shutil.rmtree(os.path.join(CACHE, f"datasets--{ORG}--{repo}"), ignore_errors=True)
            stats[repo] = {"scanned_rows": scanned, "windows": got}
            print(repo, stats[repo], flush=True)
    print(json.dumps({"total": n_out, "by_source": stats}))


if __name__ == "__main__":
    main()
