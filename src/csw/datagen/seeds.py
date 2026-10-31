"""Seed pools. Streams rows through the HF datasets-server /rows API (nothing is downloaded locally).

A seed is {seed_id, lang, track, genre, text, src, license}. Track B seeds are NATIVE-language text (the
LLM derives eng from them); Track C seeds are ENGLISH text to be localised; Track A has no seed (anchors).

NC seeds (NaijaSenti, ghana-chat, youth-conversations) are in NC_SEEDS and are ON by default (seed-only use,
by default); pass --no-nc to drop them.
"""
from __future__ import annotations

from csw.paths import DATA_ROOT, env_value
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = DATA_ROOT
SEED_DIR = HERE / "seeds"
API = "https://datasets-server.huggingface.co"


def _tok() -> str:
    return env_value("HF_TOKEN") or ""


def _get(path: str, **q) -> dict:
    url = f"{API}/{path}?{urllib.parse.urlencode(q)}"
    h = {"Authorization": f"Bearer {_tok()}"} if _tok() else {}
    for i in range(8):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60))
        except Exception as e:
            if i == 7:
                raise RuntimeError(f"{url}: {e}")
            time.sleep(min(60, 3 * 2 ** i))


def fetch_rows(dataset: str, n: int, config: str | None = None, split: str | None = None, seed: int = 0) -> list[dict]:
    """n rows spread over the split (random offsets, 100 per page)."""
    sp = _get("splits", **({"dataset": dataset, "config": config} if config else {"dataset": dataset}))["splits"]
    pick = next((s for s in sp if (config is None or s["config"] == config) and (split is None or s["split"] == split)), sp[0])
    cfg, spl = pick["config"], pick["split"]
    total = _get("size", dataset=dataset, config=cfg)["size"]["config"]["num_rows"]
    rnd = random.Random(seed)
    pages = max(1, -(-n // 100))
    offs = sorted({rnd.randrange(0, max(1, total - 100)) for _ in range(pages)}) if total > 100 * pages else [0]
    out = []
    for o in offs:
        r = _get("rows", dataset=dataset, config=cfg, split=spl, offset=o, length=min(100, max(1, total - o)))
        out += [x["row"] for x in r["rows"]]
    rnd.shuffle(out)
    return out[:n]


def _chat_text(row, role="user"):
    ms = row.get("messages") or row.get("conversations") or []
    return "\n".join(m["content"] for m in ms if m.get("role") == role)[:1200]


# (dataset, lang, track, genre, extractor, license-note)
SEEDS = [
    ("mxronga/yoruba-proverbs-parallel-corpora", "yor", "B", "tradition_law", lambda r: r["yoruba"], "apache-2.0"),
    ("belindatmz/swahiliproverbs", "swh", "B", "tradition_law", lambda r: r["swahili_proverb"], "cc-by-3.0"),
    ("yacdev/somali-ai-friend-conversations", "som", "B", "chat",
     lambda r: "\n".join(m["content"] for m in r["messages"])[:1500], "apache-2.0"),
    ("Zyroxx66/somali-tinystories", "som", "B", "story", lambda r: r["story"][:1500], "unspecified"),
    ("ik/twi_everyday_conversation", "twi", "B", "chat", lambda r: f"{r['question']}\n{r['answer']}", "mit"),
    ("kkenna/oromiffa_convo_data", "orm", "B", "chat",
     lambda r: "\n".join(m["content"] for m in r["messages"])[:1200], "unspecified"),
    ("Ephraimmm/pidgin_clean", "pcm", "B", "chat",
     lambda r: "\n".join(m["content"] for m in r["conversations"])[:1200], "apache-2.0"),
    ("michsethowusu/somali-sentiments-corpus", "som", "B", "opinion", lambda r: r["Somali"], "mit"),
    # English-side story seeds to localise (TinyStories English column shipped inside these repos)
    ("ccibeekeoc42/TinyStories_igbo", "ibo", "C", "story", lambda r: r["English"][:1500], "apache-2.0"),
    ("ccibeekeoc42/TinyStories_yoruba", "yor", "C", "story", lambda r: r["English"][:1500], "apache-2.0"),
    ("roneneldan/TinyStories", "*", "C", "story", lambda r: r["text"][:1500], "cdla-sharing-1.0"),
    # Niger-Delta / Cross-River languages: Bible/web/parallel text is all there is -> religious or formal register seeds;
    # the prompt asks for a NEW everyday situation, so this only supplies vocabulary and voice.
    ("HuggingFaceFW/fineweb-2", "efi", "B", "explain_teach", lambda r: r["text"][:1200], "odc-by", "efi_Latn"),
    ("HuggingFaceFW/fineweb-2", "urh", "B", "explain_teach", lambda r: r["text"][:1200], "odc-by", "urh_Latn"),
    ("michsethowusu/english-efik_sentence-pairs_mt560", "efi", "B", "faith", lambda r: r["efi"], "cc-by-4.0"),
    ("michsethowusu/english-urhobo_sentence-pairs_mt560", "urh", "B", "faith", lambda r: r["urh"], "cc-by-4.0"),
    ("AfriSpeech/madlad-91-african", "ibb", "B", "faith", lambda r: r["local"], "odc-by", "ibb_clean"),
]
NC_SEEDS = [   # non-commercial; ON by default now (seed-only use). AfriSenti/NaijaSenti are CC-BY-NC(-SA).
    *[("HausaNLP/NaijaSenti-Twitter", l, "B", "chat", lambda r: r["tweet"], "cc-by-nc-sa-4.0", l) for l in ("hau", "ibo", "pcm", "yor")],
    ("michsethowusu/ghana-chat-corpus-ak", "twi", "B", "explain_teach", lambda r: r["text_ak"][:1200], "cc-by-nc-4.0"),
    ("michsethowusu/ghana-chat-corpus-ee", "ewe", "B", "explain_teach", lambda r: r["text_ee"][:1200], "cc-by-nc-4.0"),
    ("tech-cyril/twi-afrisenti-cleaned", "twi", "B", "chat", lambda r: r["tweet"], "unknown (AfriSenti = NC)"),
    ("ghananlpcommunity/youth-conversations-tw", "twi", "B", "chat", lambda r: r["text"], "cc-by-nc-4.0"),
    ("ghananlpcommunity/youth-conversations-ee", "ewe", "B", "chat", lambda r: r["text"], "cc-by-nc-4.0"),
]
FEWSHOT = ("gimmy256/african-codeswitching", "cc-by-4.0")   # real mixed utterances: lug swh twi xho yor zul

_PII = re.compile(r"(@\w+|https?://\S+|\+?\d[\d\s\-()]{7,}\d|\b[\w.]+@[\w.]+\b)")


def clean(t: str) -> str:
    t = _PII.sub("", t or "")
    return re.sub(r"\s+", " ", t).strip()


def build(lang: str, per_source: int = 150, allow_nc: bool = True, seed: int = 0) -> list[dict]:
    SEED_DIR.mkdir(exist_ok=True)
    out = []
    for e in SEEDS + (NC_SEEDS if allow_nc else []):
        ds, l, track, genre, ex, lic = e[:6]
        cfg = e[6] if len(e) > 6 else None
        if l != lang and not (lang == "_en" and l == "*"):
            continue
        try:
            rows = fetch_rows(ds, per_source, config=cfg, seed=seed)
        except Exception as e:
            print(f"  [seed] {ds}: {e}", file=sys.stderr)
            continue
        for r in rows:
            try:
                t = clean(ex(r))
            except Exception:
                continue
            if len(t) < 25:
                continue
            out.append({"seed_id": hashlib.md5((ds + t).encode()).hexdigest()[:12], "lang": lang, "track": track,
                        "genre": genre, "text": t, "src": ds, "license": lic})
    (SEED_DIR / f"{lang}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out))
    return out


def load(lang: str) -> list[dict]:
    """Hub-sourced seeds (<lang>.jsonl) + the corpus seeds built on the box (<lang>.vast.jsonl)."""
    out = []
    for name in (f"{lang}.jsonl", f"{lang}.vast.jsonl"):
        p = SEED_DIR / name
        if p.exists():
            out += [json.loads(x) for x in p.read_text().split("\n") if x.strip()]
    return out


def fewshots(lang_name: str, k: int = 3) -> list[dict]:
    """Real mixed utterances for this language if gimmy256/african-codeswitching has it."""
    p = SEED_DIR / "_fewshot.jsonl"
    if not p.exists():
        rows = fetch_rows(FEWSHOT[0], 400, seed=1)
        p.parent.mkdir(exist_ok=True)
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    rows = [json.loads(x) for x in p.read_text().split("\n")]
    hit = [r for r in rows if r.get("matrix_language", "").lower().startswith(lang_name.split()[0].lower())]
    return random.sample(hit, min(k, len(hit)))


if __name__ == "__main__":
    for lg in sys.argv[1:]:
        r = build(lg)
        print(lg, len(r), {s: sum(1 for x in r if x["src"] == s) for s in {x["src"] for x in r}})
