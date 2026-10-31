"""Quality estimator: one mmBERT encoder, 27 graded dimensions, masked multi-task regression.

INPUT   [CLS] task | language | pin  source  [SEP]  target  [SEP]  devices  [SEP]
  task  eng_m | eng_hm | m_hm | eng_ood | m_ood | hm_ood (source -> target); devices = the plan phrases (hm) or idioms (ood) the generator committed to, dropped from 25% of the training examples
  so the same model also scores text whose devices are unknown.
OUTPUT  27 numbers in [1,5] (judge scale): M1-M6 | H1-H9 | O1-O7 | D1 D2 | X1 | PD (plan device score) | IQ (idiom score).  Only the dimensions that belong to the target field are scored:
  m  -> M1-M6          hm -> H1-H9 D1 X1 PD          ood -> O1-O7 D2 X1 IQ          (overall = mean of those)
"""
from __future__ import annotations
import json, random
from pathlib import Path

DIMS = [f"M{i}" for i in range(1, 7)] + [f"H{i}" for i in range(1, 10)] + [f"O{i}" for i in range(1, 8)] + ["D1", "D2", "X1", "PD", "IQ"]
FIELD_DIMS = {"m": list(range(0, 6)), "hm": list(range(6, 15)) + [22, 24, 25], "ood": list(range(15, 22)) + [23, 24, 26]}
TASKS = {"eng_m": ("eng", "m"), "eng_hm": ("eng", "hm"), "m_hm": ("m", "hm"), "eng_ood": ("eng", "ood"), "m_ood": ("m", "ood"), "hm_ood": ("hm", "ood")}
SRC_CHOICES = {"m": ["eng"], "hm": ["eng", "m"], "ood": ["eng", "m", "hm"]}
MISMATCH_DIMS = {"m": [0], "hm": [6], "ood": [15]}          # a target paired with another row's source: only fidelity of that field is labelled (1), plus X1
NOISY = {"O1": .5, "O7": .5, "X1": .5, "D2": .5, "O4": .5}
LANG = None


def lang_name(c):
    global LANG
    if LANG is None:
        from csw.sft import fmt; LANG = fmt.NAMES
    return LANG.get(c, c)


def devices_text(rec, tgt):
    if tgt == "hm": return " | ".join(rec.get("plan") or [])
    if tgt == "ood": return " | ".join(rec.get("idioms") or [])
    return ""


class Enc:
    def __init__(self, tok, max_src=220, max_tgt=220, max_dev=80):
        self.tok, self.ms, self.mt, self.md = tok, max_src, max_tgt, max_dev
        self.grader, self.default_grader = False, "std"                  # grader tag (continuous fine-tuning): old lenient labels = std, strict labels = strict; inference uses strict
        self.cls = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
        self.sep = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
        self.pad = tok.pad_token_id
        self.groups = []                                                                                                              # group / per-dimension tokens present in this tokenizer -> appended to every input
        for gs in (GROUPS, DIM_GROUPS):
            gi = [tok.convert_tokens_to_ids(g[0]) for g in gs]
            if all(isinstance(i, int) and i != tok.unk_token_id for i in gi): self.groups = gi; break

    def ids(self, t, n):
        return self.tok(t, add_special_tokens=False, truncation=True, max_length=n)["input_ids"]

    def encode(self, task, rec, source_text, target_text, dev, draft=""):
        head = self.ids(f"{task} | {lang_name(rec['lang'])} | {rec.get('pin') or '-'}" + (f" | grader: {rec.get('grader') or self.default_grader}" if self.grader else ""), 28)
        x = [self.cls] + head + self.ids(source_text, self.ms) + [self.sep] + self.ids(target_text, self.mt) + [self.sep]
        if dev: x += self.ids("devices: " + dev, self.md) + [self.sep]
        if draft: x += self.ids("draft: " + draft, self.ms) + [self.sep]
        return x + self.groups


GROUPS = [("[M]", 0, 6), ("[H]", 6, 15), ("[O]", 15, 22), ("[D]", 22, 24), ("[X]", 24, 25), ("[P]", 25, 27)]       # group tag -> its slice of the 27 outputs (M1-M6 | H1-H9 | O1-O7 | D1 D2 | X1 | PD IQ)


DIM_GROUPS = [(f"[{d}]", i, i + 1) for i, d in enumerate(["M1", "M2", "M3", "M4", "M5", "M6"] + [f"H{i}" for i in range(1, 10)] + [f"O{i}" for i in range(1, 8)] + ["D1", "D2", "X1", "PD", "IQ"])]       # variant: one token per output dimension


def add_group_tokens(tok, groups=None):
    """Six new special tokens appended to every input; the final hidden state of each one feeds its own, completely separate classifier head."""
    groups = groups or GROUPS
    tok.add_special_tokens({"additional_special_tokens": [g[0] for g in groups]})
    return [tok.convert_tokens_to_ids(g[0]) for g in groups]


DEVICE_DIMS = (22, 23, 25, 26)                                   # D1, D2, PD, IQ: kept as labels even when the devices are hidden (design decision: scoring them without seeing the devices teaches the encoder to find the devices in the text)


def make_example(enc, rec, tgt, rnd, train, drop_dev=0.25, mismatch=0.04, pool=None, task=None, use_draft=None, draft_p=0.5, fpool=None, p_wrong_dev=0.04, p_inject=0.04):
    """-> (input_ids, labels[27] with None = unlabelled).  use_draft None: random in training (ood targets only), off in evaluation; True / False forces it."""
    src = task.split("_")[0] if task else rnd.choice([s for s in SRC_CHOICES[tgt] if rec.get(s)])
    if task is None: task = f"{src}_{tgt}"
    y = rec["y"]; lab = [None] * 27
    s_text, t_text = rec[src], rec[tgt]
    if rec.get("h") is not None:                                          # human-rated translation: one scalar that trains the MEAN of M1..M6 (see collate)
        lab[0] = ("H", rec["h"]); return enc.encode(task, rec, s_text, t_text, ""), lab
    if train and pool and rnd.random() < mismatch and tgt != "m" or (train and pool and tgt == "m" and rnd.random() < mismatch):
        other = rnd.choice(pool.get(rec["lang"], [rec]))
        if other["id"] != rec["id"] and other["eng"] != rec["eng"]:
            s_text = other[src]
            for i in MISMATCH_DIMS[tgt]: lab[i] = 1.0
            lab[24] = 1.0
            dev = "" if rnd.random() < drop_dev else devices_text(rec, tgt)
            return enc.encode(task, rec, s_text, t_text, dev), lab
    for i in FIELD_DIMS[tgt]: lab[i] = y[i]
    hide = bool(train and rnd.random() < drop_dev); dev = "" if hide else devices_text(rec, tgt)
    draft = ""
    if train and pool and tgt in ("hm", "ood") and not hide and rnd.random() < p_wrong_dev:      # negative: another row's devices -> the device dimensions fail, the text dimensions keep their labels
        other = rnd.choice(pool.get(rec["lang"], [rec])); wd = devices_text(other, tgt)
        if wd and wd != dev:
            for i in ((22, 25) if tgt == "hm" else (23, 26)): lab[i] = 1.0
            dev = wd
    elif train and fpool and rnd.random() < p_inject:                                      # negative: words of a language that is neither the base nor the switch language (English words for ood) injected into the target
        words = t_text.split()
        if len(words) >= 6:
            foreign = [w for l, ws in fpool.items() if l != rec["lang"] and l not in (rec.get("pin") or "") for w in ws[:1]] if tgt != "ood" else (rec.get("eng") or "").split()
            if foreign:
                k = max(2, int(0.25 * len(words))); pos = rnd.sample(range(len(words)), k)
                for q in pos: words[q] = rnd.choice(foreign)
                t_text = " ".join(words); lab = [None] * 27; lab[{"m": 3, "hm": 9, "ood": 19}[tgt]] = 1.0       # M4 / H4 / O5: only the damaged dimension is labelled
    draft = ""
    if tgt == "ood" and rec.get("draft") and (use_draft if use_draft is not None else (train and rnd.random() < draft_p)): draft = rec["draft"]
    return enc.encode(task, rec, s_text, t_text, dev, draft), lab


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


HUMAN_WEIGHT = 0.6                                  # weight of a human-rated row (mean-of-M1..M6 loss) relative to a judge-labelled dimension


def collate(batch, pad, dev, human=False):
    import torch
    m = max(len(b[0]) for b in batch); ids = torch.full((len(batch), m), pad); att = torch.zeros((len(batch), m), dtype=torch.long)
    y = torch.zeros((len(batch), 27)); w = torch.zeros((len(batch), 27)); hy = torch.zeros(len(batch)); hw = torch.zeros(len(batch))
    for k, b in enumerate(batch):
        x, lab = b[0], b[1]; mult = b[2] if len(b) > 2 else 1.0
        ids[k, :len(x)] = torch.tensor(x); att[k, :len(x)] = 1
        for i, v in enumerate(lab):
            if isinstance(v, tuple): hy[k] = (v[1] - 1) / 4; hw[k] = HUMAN_WEIGHT * mult
            elif v is not None: y[k, i] = (v - 1) / 4; w[k, i] = NOISY.get(DIMS[i], 1.0) * mult
    if human: return ids.to(dev), att.to(dev), y.to(dev), w.to(dev), hy.to(dev), hw.to(dev)
    return ids.to(dev), att.to(dev), y.to(dev), w.to(dev)


def build_model(name, dropout=0.1, meta=None, vocab=None, gids=None):
    """Legacy: one head on [CLS] + mean pool.  Group architecture (meta["arch"] == "group", or vocab + gids given): one separate head per group tag, on that tag's own final hidden state."""
    import torch
    layout = GROUPS
    if meta and meta.get("arch") in ("group", "dim"): vocab, gids = meta["vocab"], meta["group_ids"]; layout = DIM_GROUPS if meta["arch"] == "dim" else GROUPS
    if vocab is not None and gids is not None and len(gids) == len(DIM_GROUPS): layout = DIM_GROUPS
    if vocab is not None:
        from transformers import AutoModel

        class QEG(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = AutoModel.from_pretrained(name); self.enc.resize_token_embeddings(vocab); h = self.enc.config.hidden_size; self.gids = list(gids)
                self.heads = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(h, e - b)) for _, b, e in layout])

            def forward(self, ids, att, return_hidden=False, **kw):
                hs = self.enc(input_ids=ids, attention_mask=att, **kw).last_hidden_state; ar = torch.arange(ids.shape[0], device=ids.device)
                outs = [head(hs[ar, (ids == g).float().argmax(1)]) for g, head in zip(self.gids, self.heads)]
                return (torch.cat(outs, -1), hs) if return_hidden else torch.cat(outs, -1)
        return QEG()
    from transformers import AutoModel

    class QE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(name); h = self.enc.config.hidden_size
            self.head = torch.nn.Sequential(torch.nn.Linear(2 * h, h), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(h, 27))

        def forward(self, ids, att):
            hs = self.enc(input_ids=ids, attention_mask=att).last_hidden_state; m = att.unsqueeze(-1).to(hs.dtype)
            pooled = torch.cat([hs[:, 0], (hs * m).sum(1) / m.sum(1)], -1)
            return self.head(pooled)
    return QE()


def predict(model, enc, items, dev, bs=64):
    """items: list of (ids) -> np.array [n,27] on the 1-5 scale"""
    import torch, numpy as np
    order = sorted(range(len(items)), key=lambda i: len(items[i])); out = np.zeros((len(items), 27), dtype=np.float32); model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(order), bs):
            b = order[s:s + bs]; m = max(len(items[i]) for i in b); ids = torch.full((len(b), m), enc.pad); att = torch.zeros((len(b), m), dtype=torch.long)
            for k, i in enumerate(b): ids[k, :len(items[i])] = torch.tensor(items[i]); att[k, :len(items[i])] = 1
            p = model(ids.to(dev), att.to(dev)).float().cpu().numpy() * 4 + 1
            for k, i in enumerate(b): out[i] = np.clip(p[k], 1, 5)
    return out
