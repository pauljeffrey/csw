"""Added tokens for the code-switcher (<hm>, <ood>, <dev> become real tokens).

Added: the task tags <hm> <ood>, the plan brackets <dev> </dev>, and one token per language tag that appears in hm / ood completions (<yor> ... <eng> <fra> <ara> ...).
They are added as NORMAL (non-special) added tokens on purpose: with special=True `decode(skip_special_tokens=True)` would delete them from the model output
and csw.sft.fmt.parse (and every reward) would see no tags.  m_text keeps the plain "yor " tag of the rl2b checkpoint, so the plain iso words are NOT touched.

    from csw.sft import tokens
    n_added = tokens.add_to_tokenizer(tok)            # transformers tokenizer
    tokens.resize_and_init(model, tok, old_tok)        # resize embeddings, new rows = mean of the old sub-token embeddings (not random)
"""
from __future__ import annotations

from . import fmt

STRUCTURAL = [fmt.HM_TAG, fmt.OOD_TAG, fmt.DEV_OPEN, fmt.DEV_CLOSE]


def language_tags() -> list[str]:
    codes = sorted(set(fmt.NAMES) | set(fmt.SWITCH_TAG.values()))
    return [fmt.tag(c) for c in codes]


def all_new_tokens() -> list[str]:
    return STRUCTURAL + language_tags()


def add_to_tokenizer(tok) -> int:
    """Works with a transformers tokenizer or a tokenizers.Tokenizer; returns how many were new."""
    from tokenizers import AddedToken
    toks = [AddedToken(t, special=False, normalized=False, lstrip=False, rstrip=False) for t in all_new_tokens()]
    return tok.add_tokens(toks)


def resize_and_init(model, tok, old_tok):
    """Grow the (tied) embedding matrix and initialise each new row as the mean of the old sub-token rows of its text, e.g. <hm> = mean('<', 'hm', '>')."""
    import torch
    old_n = len(old_tok)
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        for t in all_new_tokens():
            new_id = tok.convert_tokens_to_ids(t)
            if new_id < old_n:
                continue
            pieces = old_tok(t, add_special_tokens=False)["input_ids"]
            emb[new_id] = emb[pieces].mean(0)
    if model.get_output_embeddings() is not None and model.get_output_embeddings().weight.data_ptr() != emb.data_ptr():
        out = model.get_output_embeddings().weight
        with torch.no_grad():
            for t in all_new_tokens():
                new_id = tok.convert_tokens_to_ids(t)
                if new_id >= old_n:
                    out[new_id] = out[old_tok(t, add_special_tokens=False)["input_ids"]].mean(0)
