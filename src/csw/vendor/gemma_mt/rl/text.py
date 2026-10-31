# SPDX-License-Identifier: AGPL-3.0-or-later
"""Turning sampled token ids into the text the reward is allowed to see.

The owner's rule (DECISIONS.md D-12): the reward only ever sees the TRANSLATION. Before scoring we
  1. cut the id sequence at the FIRST stop id (`<end_of_turn>` / `<eos>`): nothing after it exists for the reward,
  2. decode with special tokens removed, and additionally remove any special-token STRINGS left in the text
     (`<start_of_turn>`, `<bos>`, `<unusedNN>` ... typed out as plain characters),
  3. normalise the text to Unicode NFC (D-29: Yoruba / Fon / Ewe diacritics may be sampled precomposed or decomposed; every length and
     lexical rule of the reward counts characters, so the reward must see ONE form; `nfc` switch, default on),
  4. strip the leading language tag (`csw.vendor.gemma_mt.format.strip_iso_prefix`).
Doing 1-4 on the model's raw output closes the obvious ways of earning QE reward with markup instead of language.

EOS discipline (D-26, owner 2026-10-17: the model must end firmly with its end token and output nothing after it) adds two
OBSERVATIONS here, which `reward.py` turns into failure classes:
  * `post_stop_tokens`  how many tokens that are neither padding nor another stop id follow the first stop id in the ids we were
                        given (the cut in step 1 drops them from the text, but they are counted)
  * `dangling`          the text BEFORE the stop id contains a turn / role marker (`<start_of_turn>`, `<end_of_turn>`, `<|assistant|>`
                        ...): the model opened (or closed) a turn inside its own answer
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from csw.vendor.gemma_mt.format import StripResult, strip_iso_prefix

_SPECIAL_STR = re.compile(
    r"<(?:start_of_turn|end_of_turn|bos|eos|pad|unk|mask|start_of_image|end_of_image|image_soft_token|unused\d+|"
    r"\[multimodal\])>")


# turn / role markers that must never appear INSIDE an answer (before its end token)
_TURN_MARKER = re.compile(r"<(?:start_of_turn|end_of_turn)>|<\|(?:assistant|user|system|im_start|im_end)\|>")


@dataclass
class Decoded:
    text: str                 # translation text + (maybe) language tag, specials removed, stripped
    finished: bool            # a stop id was produced inside the budget
    n_tokens: int             # tokens kept (stop id included when finished)
    special_ids_stripped: bool    # special token ids occurred before the stop id
    special_text_stripped: bool   # special-token strings occurred in the decoded text
    post_stop_tokens: int = 0     # non-padding, non-stop tokens after the first stop id (dropped from `text`)
    dangling: bool = False        # a turn / role marker occurs before the stop id


def cut_at_stop(ids: Sequence[int], stop_ids: Sequence[int]):
    """(ids up to and INCLUDING the first stop id, finished)."""
    stops = set(int(s) for s in stop_ids)
    for j, t in enumerate(ids):
        if int(t) in stops:
            return list(ids[: j + 1]), True
    return list(ids), False


def nfc(text: str) -> str:
    """Unicode NFC (canonical composition). Idempotent; a no-op on text without combining marks."""
    return unicodedata.normalize("NFC", text)


def decode_completion(tokenizer: Any, ids: Sequence[int], stop_ids: Sequence[int],
                      special_ids: Optional[Sequence[int]] = None, pad_ids: Sequence[int] = (), normalise: bool = True) -> Decoded:
    """`pad_ids`: ids that are padding (benign after the first stop id, together with the stop ids themselves).
    `normalise`: NFC the decoded text (D-29)."""
    kept, finished = cut_at_stop(ids, stop_ids)
    body = kept[:-1] if finished else kept
    raw = tokenizer.decode(body, skip_special_tokens=False)
    clean = tokenizer.decode(body, skip_special_tokens=True)
    ids_stripped = raw != clean
    text = _SPECIAL_STR.sub("", clean)
    text_stripped = text != clean
    benign = set(int(s) for s in stop_ids) | set(int(p) for p in pad_ids)
    post = sum(1 for t in ids[len(kept):] if int(t) not in benign) if finished else 0
    text = nfc(text) if normalise else text
    return Decoded(text.strip(), finished, len(kept), ids_stripped, text_stripped,
                   post_stop_tokens=post, dangling=bool(_TURN_MARKER.search(raw)))


def translation_from(decoded: Decoded, iso: str) -> StripResult:
    return strip_iso_prefix(decoded.text, iso)
