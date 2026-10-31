"""GlotLID (fastText) language-identification guard.  Sentence-level (>= MIN_WORDS words), batched.  `Lid.label(sentences)` -> GlotLID codes (script suffix dropped).
The model (cis-lmu/glotlid model_v3.bin, public) is downloaded once; needs `fasttext` (box: PYTHONPATH tail work/pylib)."""
from __future__ import annotations
import os, re

ALIAS = {"pcm": {"pcm", "wes"}, "wes": {"pcm", "wes"}, "efi": {"efi", "ibb"}, "ibb": {"efi", "ibb"}, "twi": {"twi", "aka"}, "orm": {"orm", "gaz", "hae", "gax"}, "ful": {"ful", "fub", "fuv", "fuf", "fuc", "fue", "fuh", "fui", "fuq"},
         "mlg": {"mlg", "plt", "xmv"}, "swh": {"swh", "swc", "swa"}, "nde": {"nde", "nbl"}, "kon": {"kon", "kng", "ktu"}, "tum": {"tum"}, "afr": {"afr"}, "gpe": {"gpe", "pcm", "wes"}, "urh": {"urh"}, "eng": {"eng"},
         "fra": {"fra"}, "sheng": {"swh", "eng"}, "ara": {"ara", "arb", "arz", "apc", "acm", "ary"}, "amh": {"amh"}, "tir": {"tir"}, "lin": {"lin"}}
MIN_WORDS = 4
CONF_MIN, CONF_FULL = 0.30, 0.80
SENT = re.compile(r"(?<=[.!?;:])\s+|\n+")


def allowed(lang: str) -> set[str]: return set(ALIAS.get(lang, {lang}))


class Lid:
    def __init__(self, path: str | None = None):
        import fasttext
        if path is None:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("cis-lmu/glotlid", "model_v3.bin")
        self.m = fasttext.load_model(path)

    def label_conf(self, sents: list[str]) -> list[tuple[str, float]]:
        """(label, probability) per sentence; the probability is GlotLID's confidence in the top label."""
        if not sents: return []
        labs, probs = self.m.predict([x.replace("\n", " ") for x in sents], k=1)
        return [((l[0] if l else "__label__und").replace("__label__", "").split("_")[0], float(p[0]) if len(p) else 0.0) for l, p in zip(labs, probs)]

    def label(self, sents: list[str]) -> list[str]:
        if not sents: return []
        raw = self.m.f.multilinePredict([s.replace("\n", " ") for s in sents], 1, 0.0, "replace"); raw = raw[0] if isinstance(raw, tuple) and len(raw) == 2 and len(raw[0]) == len(sents) else raw; out = []
        for r in raw:
            lab = r[0] if r else None
            while isinstance(lab, (list, tuple)): lab = lab[0] if lab else None
            if isinstance(lab, float): lab = r[1] if len(r) > 1 and isinstance(r[1], str) else None
            out.append(str(lab).replace("__label__", "").split("_")[0] if lab else "und")
        return out


def sentences(text: str) -> list[str]: return [s.strip() for s in SENT.split(text) if len(s.split()) >= MIN_WORDS]


def lid_penalty(lid: Lid | None, text: str, base: str, switch, task_target: str, tol: float = 0.15, need_base: dict | None = None) -> tuple[float, dict]:
    """p = max( (bad_share - tol) / 0.5 , (need - base_share) / need ), capped at 1.  bad = sentences labelled neither base nor (hm only) switch language; need = minimal share of base-language sentences (hm 0.4, others 0.7)."""
    if lid is None: return 0.0, {"lid": "off"}
    ss = sentences(text)
    if not ss: return 0.0, {"lid": "no_sentence"}
    lp = lid.label_conf(ss); ok_b = allowed(base); ok_s = set().union(*[allowed(x) for x in ([switch] if isinstance(switch, str) else (switch or []))]) if switch else set(); need = (need_base or {"hm": 0.4}).get(task_target, 0.7)
    wts = [0.0 if c < CONF_MIN else min(1.0, (c - CONF_MIN) / (CONF_FULL - CONF_MIN)) for _, c in lp]; W = sum(wts)              # a verdict counts in proportion to GlotLID's confidence; below CONF_MIN it does not count
    if W < 1e-6: return 0.0, {"lid": "low_confidence"}
    base_share = sum(w for (l, _), w in zip(lp, wts) if l in ok_b) / W; bad = sum(w for (l, _), w in zip(lp, wts) if l not in ok_b and l not in ok_s) / W
    p = max(max(0.0, bad - tol) / 0.5, max(0.0, need - base_share) / need)
    return min(1.0, p), {"lid_n": len(ss), "lid_bad": round(bad, 3), "lid_base": round(base_share, 3)}


def window_share(lid: Lid | None, text: str, switch: str, win: int = 4) -> float | None:
    """hm: share of non-overlapping `win`-word windows whose GlotLID label is the switch language (a text of fewer than 3 windows says nothing -> None)."""
    if lid is None: return None
    w = text.split(); ch = [" ".join(w[i:i + win]) for i in range(0, len(w) - win + 1, win)]
    if len(ch) < 3: return None
    labs = lid.label(ch); ok = allowed(switch); return sum(x in ok for x in labs) / len(labs)


def switch_lid_penalty(share: float | None, lo: float = 0.15, hi: float = 0.40, slack: float = 0.10) -> tuple[float, dict]:
    """The switch language must be present at 15-40% of the text by language ID (windows are noisy, so `slack` widens the band on both sides); 0 share = the switch language is absent = full penalty."""
    if share is None: return 0.0, {"lid_switch": "short"}
    lo2, hi2 = max(0.0, lo - slack), hi + slack
    p = 0.0 if lo2 <= share <= hi2 else min(1.0, (lo2 - share) / max(lo2, 1e-6) if share < lo2 else (share - hi2) / (1 - hi2))
    return p, {"lid_switch_share": round(share, 3)}
