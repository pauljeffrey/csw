"""Deterministic, all-or-nothing applier for the judge's span edits (rubric_v1.md section 5.3).

    new_text, report = apply_edits(text, edits)        # edits: [{"find","ctx","rep",...}] for ONE field
An edit applies only if  ctx + find  (ctx = up to 4 words immediately before; an empty ctx means find alone) occurs EXACTLY ONCE in the NFC-normalised text, spans do not overlap, and the result
passes the post-checks.  Any failure returns the original text and a report with the reason, so the caller falls back to a full-rewrite request.  Edits are applied from the end of the text.
"""
from __future__ import annotations

import re
import unicodedata

MAX_CHANGE_RATIO = 0.25


def nfc(t: str) -> str:
    return unicodedata.normalize("NFC", t)


def locate(text: str, find: str, ctx: str) -> list[int]:
    """start offsets of `find` that are preceded by ctx (whitespace-insensitive between ctx and find)"""
    find, ctx = nfc(find), nfc(ctx).strip()
    if not find:
        return []
    if not ctx:
        return [m.start() for m in re.finditer(re.escape(find), text)]
    pat = re.escape(ctx) + r"\s*" + re.escape(find)
    return [m.end() - len(find) for m in re.finditer(pat, text)]


def apply_edits(text: str, edits: list[dict], digits_must_match: bool = True) -> tuple[str, dict]:
    t = nfc(text); spans = []
    for i, e in enumerate(edits):
        pos = locate(t, e.get("find", ""), e.get("ctx", ""))
        if len(pos) != 1:
            return text, {"ok": False, "why": f"edit {i}: find occurs {len(pos)} times with its context"}
        spans.append((pos[0], pos[0] + len(nfc(e["find"])), nfc(e.get("rep", ""))))
    spans.sort()
    for (a0, a1, _), (b0, b1, _) in zip(spans, spans[1:]):
        if b0 < a1:
            return text, {"ok": False, "why": "overlapping edits"}
    out = t
    for a0, a1, rep in reversed(spans):
        out = out[:a0] + rep + out[a1:]
    changed = sum(a1 - a0 for a0, a1, _ in spans) + sum(len(r) for _, _, r in spans)
    if changed > MAX_CHANGE_RATIO * 2 * max(1, len(t)):
        return text, {"ok": False, "why": "change ratio too large"}
    if digits_must_match and sorted(re.findall(r"\d+", t)) != sorted(re.findall(r"\d+", out)):
        return text, {"ok": False, "why": "digits changed"}
    if not out.strip():
        return text, {"ok": False, "why": "empty result"}
    return out, {"ok": True, "n_edits": len(spans), "chars_before": len(t), "chars_after": len(out)}
