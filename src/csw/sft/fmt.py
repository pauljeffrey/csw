"""Prompt / completion formats for the code-switcher (eng -> m_text | hm | ood) on top of gemma-3-270m-translate-rl2b.

m_text is EXACTLY the rl2b format (model card, 2026-10): user turn  "Translate the text to {Name}:  {english}"  (two spaces after the colon),
model turn  "<iso> <translation>"  (plain iso word, one space).  Nothing about it changes, so the RL'd translation behaviour is kept.

hm / ood are new tasks with short task tags and a "plan first" completion:
    hm   prompt:  "<hm> {Base}+{Switch}:  {english}"        completion: "<yor><eng><dev>kept one | kept two</dev> {code-switched text}"
    ood  prompt:  "<ood> {Base}:  {english}"                completion: "<yor><dev>idiom one | idiom two</dev> {informal native text}"
The language tags come first (the model commits to base + switch language), then the plan (the `devices`), then the text.  Every constant that
decides the surface form lives here, so changing a tag style is a one-line edit.
"""
from __future__ import annotations

import re

# language name used in the instruction -> iso tag. Names are the ones in the rl2b table ("Pidgin", "Sepedi", "Kirundi", ...).
NAMES = {
    "yor": "Yoruba", "hau": "Hausa", "ibo": "Igbo", "pcm": "Pidgin", "twi": "Twi", "ewe": "Ewe", "fon": "Fon", "wol": "Wolof", "swh": "Swahili",
    "kin": "Kinyarwanda", "som": "Somali", "orm": "Oromo", "amh": "Amharic", "tir": "Tigrinya", "lug": "Luganda", "lin": "Lingala", "zul": "Zulu",
    "xho": "Xhosa", "afr": "Afrikaans", "sna": "Shona", "nya": "Chichewa", "sot": "Sesotho", "tsn": "Setswana", "mlg": "Malagasy", "tum": "Tumbuka",
    "run": "Kirundi", "tso": "Tsonga", "nso": "Sepedi", "ssw": "Swati", "nde": "North Ndebele", "kik": "Kikuyu", "luo": "Luo", "bam": "Bambara",
    "kon": "Kongo", "ful": "Fulah", "wes": "Cameroon Pidgin",
    "gpe": "Ghanaian Pidgin",      # NOT in the rl2b table: the translate checkpoint never saw this tag, SFT has to teach it
}
# switch-language names used by profiles.P[lang].switch -> tag. English/French/Arabic/Italian are not in NAMES (they are never a target).
SWITCH_TAG = {"English": "eng", "French": "fra", "Arabic": "ara", "Italian": "ita", "Afrikaans": "afr", "Swahili": "swh", "Amharic": "amh", "Zulu": "zul",
              "Nigerian Pidgin": "pcm", "Ghanaian Pidgin": "gpe", "Yoruba": "yor", "Igbo": "ibo", "Hausa": "hau", "Twi": "twi", "Ewe": "ewe", "Wolof": "wol",
              "Tumbuka": "tum", "Chichewa": "nya", "Shona": "sna", "Lingala": "lin", "Ndebele": "nde", "Kituba": "ktu", "Ga": "gaa", "Ewondo": "ewo", "Duala": "dua",
              "Sheng": "sheng"}      # Sheng has no ISO 639-3 code: the tag is a project-local one
SWITCH_NAME = {**{v: k for k, v in NAMES.items()}, }
TAG_NAME = {t: n for n, t in SWITCH_TAG.items()}          # tag -> display name used in the prompt
for _c, _n in NAMES.items():
    TAG_NAME.setdefault(_c, _n)
TAG_NAME["pcm"] = "Nigerian Pidgin"      # as a SWITCH language the project calls it Nigerian Pidgin

HM_TAG, OOD_TAG = "<hm>", "<ood>"
DEV_OPEN, DEV_CLOSE = "<dev>", "</dev>"
DEV_SEP = " | "

TASKS = ("m_text", "hm", "ood")
_LANG_TAG = re.compile(r"^((?:<(?!dev>)[a-z]{3,5}>)+)")
_DEV = re.compile(re.escape(DEV_OPEN) + r"(.*?)" + re.escape(DEV_CLOSE), re.S)


def tag(code: str) -> str:
    return f"<{code}>"


def prompt_m_text(base: str, english: str) -> str:
    return f"Translate the text to {NAMES[base]}:  {english}"


def prompt_hm(base: str, switch_tag: str, english: str) -> str:
    return f"{HM_TAG} {NAMES[base]}+{TAG_NAME[switch_tag]}:  {english}"


def prompt_ood(base: str, english: str) -> str:
    return f"{OOD_TAG} {NAMES[base]}:  {english}"


def completion_m_text(base: str, text: str) -> str:
    return f"{base} {text}"


def completion_hm(base: str, switch_tag: str, devices: list[str], text: str) -> str:
    return f"{tag(base)}{tag(switch_tag)}{DEV_OPEN}{DEV_SEP.join(devices)}{DEV_CLOSE} {text}"


def completion_ood(base: str, devices: list[str], text: str, switch_tag: str | None = None) -> str:
    return f"{tag(base)}{tag(switch_tag) if switch_tag else ''}{DEV_OPEN}{DEV_SEP.join(devices)}{DEV_CLOSE} {text}"


def gemma_text(prompt: str, completion: str | None = None) -> str:
    """Gemma-3 chat template as the rl2b card shows it (training example when `completion` is given, generation prompt otherwise)."""
    s = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    return s if completion is None else s + completion + "<end_of_turn>\n"


def parse(task: str, base: str, completion: str) -> dict:
    """Split a model completion into its parts. ok=False says the structure is wrong (rewards use that). Never raises."""
    c = completion.strip()
    out = {"ok": False, "tags": [], "devices": [], "text": "", "raw": completion}
    if task == "m_text":
        m = re.match(r"^([a-z]{3,5}) (.*)$", c, re.S)
        if m:
            out.update(ok=True, tags=[m.group(1)], text=m.group(2).strip())
        return out
    t = _LANG_TAG.match(c)
    if not t:
        return out
    out["tags"] = re.findall(r"<(?!dev>)([a-z]{3,5})>", t.group(1))
    rest = c[t.end():].lstrip()
    d = _DEV.match(rest)
    if not d:
        return out
    out["devices"] = [x.strip() for x in d.group(1).split(DEV_SEP.strip()) if x.strip()]
    out["text"] = rest[d.end():].strip()
    out["ok"] = bool(out["text"])
    return out
