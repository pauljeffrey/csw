# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single source of truth for the input/output format of the translation model.

    user turn   :  Translate the text to {african_language}:  {english_text}
    model turn  :  {iso} {translation}<end_of_turn>

Reverse direction (African -> English) uses the SAME template with the pivot language: the user turn is
`Translate the text to English:  {african_text}` and the model turn `eng {english}<end_of_turn>`.

wrapped by the model's OWN chat template (Gemma 3 IT):

    <bos><start_of_turn>user
    Translate the text to Yoruba:  Good morning<end_of_turn>
    <start_of_turn>model
    yor E kaaro<end_of_turn>

Rules this file enforces (every one is covered by tests/test_format.py):

  * The template string is LITERAL: there are TWO spaces after the colon (`INSTRUCTION_TEMPLATE`).
  * The rendering always goes through `tokenizer.apply_chat_template`; `reference_render` below is only a
    cross-check that the tokenizer's template is the Gemma 3 one (it fails fast at start-up if it is not).
  * The template already contains `<bos>`; the rendered string is tokenized with `add_special_tokens=False`, so
    there is exactly one BOS.
  * The prompt and the target are tokenized as two separate segments: tokens(prompt) + tokens(target). That is
    exactly what happens at inference (the model continues after the tokenized prompt), so training and inference
    see identical token ids at the boundary. `check_segment_consistency` verifies, on a real tokenizer, that this
    equals tokenizing the joint string (it should; if it does not, the separate-segment form is still the right one).
  * Loss covers ONLY `{iso} {translation}<end_of_turn>`: the prompt tokens and the trailing "\\n" the template
    emits after `<end_of_turn>` carry label -100.
  * The token that must stop generation is `<end_of_turn>` (id 106 in Gemma 3). The tokenizer's `eos_token` of the
    IT checkpoints is `<eos>` (id 1); `generation_config.eos_token_id` lists both. `resolve_stop_tokens` verifies
    this against the actual tokenizer / generation config and `ensure_generation_config` repairs it before saving.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from csw.vendor.gemma_mt.languages import ALL_ISOS, PIVOT_ISO, canonical_iso, display_name, is_pivot, target_tag_and_name

# --------------------------------------------------------------------------------------------------- constants
# LITERAL. Two spaces after the colon, exactly as the project owner types the prompt. Do not "fix" it.
INSTRUCTION_TEMPLATE = "Translate the text to {african_language}:  {english_text}"
END_OF_TURN = "<end_of_turn>"
START_OF_TURN = "<start_of_turn>"
IGNORE_INDEX = -100
FORMAT_VERSION = 1

# What a rendered conversation must look like (used by the tests and the model card).
GEMMA3_TEMPLATE_EXAMPLE = (
    "<bos><start_of_turn>user\nTranslate the text to Yoruba:  Good morning<end_of_turn>\n"
    "<start_of_turn>model\nyor E kaaro<end_of_turn>\n"
)

PREFIX_POLICIES = ("require", "forbid", "optional")

# Translation directions. The format is IDENTICAL in both: only the target language changes.
#   en2x   `Translate the text to Yoruba:  Good morning`   ->  `yor E kaaro`
#   x2en   `Translate the text to English:  E kaaro`        ->  `eng Good morning`
EN2X, X2EN = "en2x", "x2en"
DIRECTIONS = (EN2X, X2EN)


def directed_pair(english: str, target: str, iso: str, direction: str = EN2X) -> Tuple[str, str, str]:
    """(source_text, target_tag, target_text) of one parallel pair in the requested direction."""
    if direction == EN2X:
        return english, iso, target
    if direction == X2EN:
        return target, PIVOT_ISO, english
    raise ValueError(f"direction must be one of {DIRECTIONS}")


# --------------------------------------------------------------------------------------------------- messages
def build_user_content(english: str, iso: str) -> str:
    """`Translate the text to {name}:  {text}`. `english` is the SOURCE text (English for en2x; the African-language
    text for the reverse direction, where `iso` is "eng"). The source is stripped (the chat template trims the
    message ends anyway; stripping first makes the result independent of the template's `| trim`)."""
    english = (english or "").strip()
    if not english:
        raise ValueError("empty source text")
    return INSTRUCTION_TEMPLATE.format(african_language=target_tag_and_name(iso)[1], english_text=english)


def build_model_content(iso: str, translation: str) -> str:
    """`{iso} {translation}` with the canonical tag (aliases are normalised)."""
    tag = target_tag_and_name(iso)[0]
    translation = (translation or "").strip()
    if not translation:
        raise ValueError("empty translation")
    return f"{tag} {translation}"


def build_messages(english: str, iso: str, translation: Optional[str] = None) -> List[Dict[str, str]]:
    msgs = [{"role": "user", "content": build_user_content(english, iso)}]
    if translation is not None:
        msgs.append({"role": "assistant", "content": build_model_content(iso, translation)})
    return msgs


# --------------------------------------------------------------------------------------------------- rendering
def reference_render(messages: Sequence[Dict[str, str]], add_generation_prompt: bool = False, bos: str = "<bos>") -> str:
    """Pure-python re-statement of the Gemma 3 IT chat template for plain-string messages (no system turn needed
    here, but a leading system message is folded into the first user turn exactly like the template does).

    NOT used to build training data. It exists (a) so offline tests and the stub tokenizer have a template and
    (b) so `assert_template_matches_reference` can refuse a tokenizer whose template differs."""
    msgs = list(messages)
    prefix = ""
    if msgs and msgs[0]["role"] == "system":
        prefix = msgs[0]["content"] + "\n\n"
        msgs = msgs[1:]
    out = [bos]
    for i, m in enumerate(msgs):
        if (m["role"] == "user") != (i % 2 == 0):
            raise ValueError("Conversation roles must alternate user/assistant/user/assistant/...")
        role = "model" if m["role"] == "assistant" else m["role"]
        out.append(f"{START_OF_TURN}{role}\n" + (prefix if i == 0 else "") + m["content"].strip() + f"{END_OF_TURN}\n")
    if add_generation_prompt:
        out.append(f"{START_OF_TURN}model\n")
    return "".join(out)


def render_chat(tokenizer: Any, messages: Sequence[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Render with the tokenizer's own chat template (never hand-rolled)."""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("the tokenizer has no chat_template; load the `-it` tokenizer (google/gemma-3-270m-it)")
    return tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=add_generation_prompt)


def assert_template_matches_reference(tokenizer: Any) -> None:
    """Fail fast when the tokenizer's template is not the Gemma 3 one (different role names, extra system text...)."""
    probe_u = "Translate the text to Yoruba:  Good morning"
    probe = [{"role": "user", "content": probe_u}, {"role": "assistant", "content": "yor E kaaro"}]
    got_a = render_chat(tokenizer, probe[:1], add_generation_prompt=True)
    got_b = render_chat(tokenizer, probe, add_generation_prompt=False)
    bos = getattr(tokenizer, "bos_token", None) or "<bos>"
    want_a = reference_render(probe[:1], True, bos=bos)
    want_b = reference_render(probe, False, bos=bos)
    if got_a != want_a or got_b != want_b:
        raise ValueError(
            "the tokenizer's chat template does not render the Gemma 3 format this module is built on.\n"
            f"  prompt   got : {got_a!r}\n  prompt   want: {want_a!r}\n"
            f"  full     got : {got_b!r}\n  full     want: {want_b!r}")


def _encode(tokenizer: Any, text: str) -> List[int]:
    out = tokenizer(text, add_special_tokens=False)
    ids = out["input_ids"] if not isinstance(out, list) else out
    return [int(i) for i in ids]


def prompt_text(tokenizer: Any, english: str, iso: str) -> str:
    """The prompt string: user turn + generation prompt (`<start_of_turn>model\\n`)."""
    return render_chat(tokenizer, build_messages(english, iso), add_generation_prompt=True)


def build_prompt(english: str, iso: str, tokenizer: Any) -> List[int]:
    """Token ids to feed `generate()`: ends with the generation prompt, so the model's first token is the `iso` tag."""
    ids = _encode(tokenizer, prompt_text(tokenizer, english, iso))
    _check_single_bos(tokenizer, ids)
    return ids


def _check_single_bos(tokenizer: Any, ids: Sequence[int]) -> None:
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None or not ids:
        return
    if ids[0] != bos:
        raise ValueError("the rendered prompt does not start with <bos>")
    if len(ids) > 1 and ids[1] == bos:
        raise ValueError("double <bos>: tokenize the chat-template output with add_special_tokens=False")


@dataclass
class Example:
    input_ids: List[int]
    labels: List[int]            # same length as input_ids; IGNORE_INDEX everywhere except the target tokens
    prompt_len: int              # number of prompt tokens (all masked)
    target_len: int              # number of tokens that carry loss: `{iso} {translation}<end_of_turn>`
    tail_len: int                # tokens after <end_of_turn> (the template's "\n"), masked
    text: str = field(default="", repr=False)

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)


def build_example(english: str, iso: str, translation: str, tokenizer: Any) -> Example:
    """Training example. Loss ONLY on `{iso} {translation}<end_of_turn>`.

    The target string is taken from the tokenizer's own rendering of the full conversation (everything after the
    prompt), so nothing about the turn markers is hand-written here."""
    msgs = build_messages(english, iso, translation)
    prompt = render_chat(tokenizer, msgs[:1], add_generation_prompt=True)
    full = render_chat(tokenizer, msgs, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise ValueError("chat template: the full conversation does not start with the generation prompt "
                         f"(prompt={prompt!r}, full={full!r})")
    suffix = full[len(prompt):]                           # "{iso} {translation}<end_of_turn>\n"
    cut = suffix.rfind(END_OF_TURN)
    if cut < 0:
        raise ValueError(f"no {END_OF_TURN} in the rendered model turn: {suffix!r}")
    target_str = suffix[: cut + len(END_OF_TURN)]         # up to and including <end_of_turn>
    tail_str = suffix[cut + len(END_OF_TURN):]            # the "\n" (may be empty with another template)
    p_ids = _encode(tokenizer, prompt)
    t_ids = _encode(tokenizer, target_str)
    tail_ids = _encode(tokenizer, tail_str) if tail_str else []
    _check_single_bos(tokenizer, p_ids)
    eot = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    if not t_ids or t_ids[-1] != eot:
        raise ValueError(f"the target does not end with the {END_OF_TURN} token id ({eot}): {t_ids[-3:]}")
    ids = p_ids + t_ids + tail_ids
    labels = [IGNORE_INDEX] * len(p_ids) + list(t_ids) + [IGNORE_INDEX] * len(tail_ids)
    return Example(ids, labels, len(p_ids), len(t_ids), len(tail_ids), text=full)


def check_segment_consistency(tokenizer: Any, probes: Optional[Sequence[Tuple[str, str, str]]] = None) -> List[str]:
    """Compare tokens(prompt)+tokens(target) with tokens(prompt+target) on probe sentences.

    Returns a list of human-readable mismatch descriptions (empty = consistent). Intended for a REAL tokenizer
    (the integration test and the data-preparation script call it); training uses the separate-segment ids either
    way, so a mismatch is a warning about "joint tokenization differs", not a correctness bug."""
    probes = probes or [
        ("Good morning, how are you?", "yor", "E kaaro, bawo ni o se wa?"),
        ("The meeting starts at 9:30.\nPlease come early.", "hau", "Taron zai fara da karfe 9:30."),
        ("I love you.", "ibo", "Ihunanya m gi."),
    ]
    bad = []
    for en, iso, tr in probes:
        ex = build_example(en, iso, tr, tokenizer)
        joint = _encode(tokenizer, ex.text)
        if joint != ex.input_ids:
            bad.append(f"{iso}: separate={len(ex.input_ids)} tokens, joint={len(joint)} tokens")
    return bad


# --------------------------------------------------------------------------------------------------- stop tokens
@dataclass
class StopTokens:
    end_of_turn_id: int
    eos_id: Optional[int]
    ids: List[int]               # what to pass as `eos_token_id` to generate()
    warnings: List[str] = field(default_factory=list)


def resolve_stop_tokens(tokenizer: Any, generation_config: Any = None) -> StopTokens:
    """Verify at RUNTIME which ids end a model turn. `<end_of_turn>` must be a real token; it is the id the training
    target ends with and the id generation has to stop on. `<eos>` is kept as a second stop id (it is what the
    base checkpoints emit)."""
    eot = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    unk = getattr(tokenizer, "unk_token_id", None)
    if eot is None or (unk is not None and eot == unk):
        raise ValueError(f"{END_OF_TURN} is not in the vocabulary of this tokenizer")
    eos = getattr(tokenizer, "eos_token_id", None)
    warnings: List[str] = []
    if generation_config is not None:
        cfg = getattr(generation_config, "eos_token_id", None)
        cfg_ids = [] if cfg is None else (list(cfg) if isinstance(cfg, (list, tuple)) else [cfg])
        if eot not in cfg_ids:
            warnings.append(f"generation_config.eos_token_id={cfg_ids} does not contain {END_OF_TURN} ({eot}); "
                            "generate() would not stop on it. ensure_generation_config() fixes this.")
    ids = [eot] + ([eos] if eos is not None and eos != eot else [])
    return StopTokens(eot, eos, ids, warnings)


def ensure_generation_config(generation_config: Any, stop: StopTokens, pad_id: Optional[int] = None) -> Any:
    """Make `generate()` stop on <end_of_turn> (and <eos>). Mutates and returns the generation config."""
    generation_config.eos_token_id = list(stop.ids)
    if pad_id is not None:
        generation_config.pad_token_id = pad_id
    return generation_config


# --------------------------------------------------------------------------------------------------- outputs
@dataclass
class StripResult:
    text: str          # the translation without the tag
    status: str        # "ok" | "missing" | "wrong" | "empty"
    found: Optional[str] = None   # the tag that was found (when status is ok/wrong)

    @property
    def has_prefix(self) -> bool:
        return self.status in ("ok", "wrong")


_PREFIX_RE = re.compile(r"^([A-Za-z]{3})(?:\s+|$)")
_KNOWN_TAGS = set(ALL_ISOS) | {"eng"}


def strip_iso_prefix(completion: str, iso: str) -> StripResult:
    """Split a generated completion into (tag, translation).

    status:
      ok       starts with the expected tag followed by a space (or is just the tag)
      wrong    starts with a DIFFERENT known language tag; `text` is what follows it
      missing  does not start with any known tag; `text` is the whole completion
      empty    nothing (left) after stripping
    The completion must already be decoded with special tokens removed. Only whitespace is normalised at the
    ends; the interior of the translation is never touched."""
    expected = target_tag_and_name(iso)[0]
    s = (completion or "").strip()
    if not s:
        return StripResult("", "empty")
    m = _PREFIX_RE.match(s)
    if m and m.group(1) in _KNOWN_TAGS:
        tag = m.group(1)
        rest = s[m.end():].strip()
        if tag == expected:
            return StripResult(rest, "ok", tag)
        return StripResult(rest, "wrong", tag)
    return StripResult(s, "missing")


def prefix_violation(status: str, policy: str = "require") -> bool:
    """True when the prefix status breaks the configured policy (-> the format penalty applies in RL)."""
    if policy not in PREFIX_POLICIES:
        raise ValueError(f"prefix policy must be one of {PREFIX_POLICIES}")
    if policy == "optional":
        return False
    if policy == "require":
        return status != "ok"
    return status in ("ok", "wrong")      # forbid: any leading tag is a violation


def write_greedy_generation_config(model_dir: str, stop_ids=None, pad_id: Optional[int] = 0, max_new_tokens: int = 320) -> str:
    """D-67 (R12): the generation_config.json shipped with the Gemma checkpoints says do_sample true, top_k 64, top_p 0.95; anyone calling `generate()` without arguments loses ~1.4 chrF++.
    Rewrite it GREEDY: do_sample false, no top_k / top_p / min_p / temperature keys, max_new_tokens 320, eos_token_id = the stop ids (default [1, 106] = <eos>, <end_of_turn>). Keeps
    `cache_implementation` (hybrid sliding-window cache of Gemma 3) and the transformers version of the existing file. RL rollouts do not read this file (explicit sampling parameters)."""
    import json
    import os

    path = os.path.join(model_dir, "generation_config.json")
    old = {}
    if os.path.exists(path):
        try:
            old = json.load(open(path, encoding="utf-8"))
        except Exception:
            old = {}
    ids = [int(x) for x in (stop_ids if stop_ids is not None else (1, 106))]
    ids = sorted(set(ids)) if stop_ids is None else list(dict.fromkeys(ids))
    new = {k: v for k, v in old.items() if k in ("_from_model_config", "bos_token_id", "cache_implementation", "transformers_version")}
    new.update({"do_sample": False, "eos_token_id": ids, "pad_token_id": int(pad_id) if pad_id is not None else old.get("pad_token_id", 0), "max_new_tokens": int(max_new_tokens)})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(new, f, indent=2)
        f.write("\n")
    return path
