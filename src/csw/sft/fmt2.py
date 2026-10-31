"""Prompt / completion formats of the code-switcher csw (gemma-3-270m-translate-rl2c + 5 new tasks).

    eng->hm   <hm> <yor>+<eng>:  {english}
    m->hm     <hm> <yor>+<eng>:  {m_text}
    eng->ood  <ood> <yor>:  {english}
    m->ood    <ood> <yor>:  {m_text}
    hm->ood   <ood> <yor>+<eng>:  {hm}
    eng->m    Translate the text to Yoruba:  {english}                (replay of the original translation task; completion "yor {text}")
completion of hm / ood:  <yor><eng><dev>phrase one | phrase two</dev> {text}
Every tag (<hm> <ood> <dev> </dev> and the language tags) is ONE vocabulary token: tokenizer slots that were `<unusedN>` are renamed (csw.sft.make_base), so the model learns them from the first step."""
from __future__ import annotations
import sys
from pathlib import Path
from csw.sft import fmt

HM, OOD, DEV_O, DEV_C = fmt.HM_TAG, fmt.OOD_TAG, fmt.DEV_OPEN, fmt.DEV_CLOSE
DRAFT_O, DRAFT_C = "<draft>", "</draft>"
TASKS = ("eng_hm", "m_hm", "eng_ood", "m_ood", "hm_ood", "eng_m")


def special_tokens() -> list[str]:
    tags = sorted({f"<{c}>" for c in fmt.NAMES} | {f"<{c}>" for c in fmt.SWITCH_TAG.values()})
    return [HM, OOD, DEV_O, DEV_C] + tags + [DRAFT_O, DRAFT_C]      # draft tags are appended last: every id of the earlier tags stays the same


def src_desc(base: str, switch: str | None) -> str:
    return f"<{base}>" if not switch else f"<{base}>+<{switch}>"


def prompt(task: str, base: str, switch: str | None, text: str) -> str:
    """ISO tags, base language first; the language of the input text (English / native / hm) tells the model which source it is, the completion is the same for all of them."""
    if task in ("eng_hm", "m_hm"): return f"{HM} {src_desc(base, switch)}:  {text}"
    if task in ("eng_ood", "m_ood"): return f"{OOD} {src_desc(base, None)}:  {text}"
    if task == "hm_ood":  return f"{OOD} {src_desc(base, switch)}:  {text}"
    if task == "eng_m":   return fmt.prompt_m_text(base, text)
    raise ValueError(task)
