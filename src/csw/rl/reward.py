"""Reward = evaluator score (equal-weight mean of the task's dimensions, scaled to 0..1) minus capped deterministic rule penalties; any format failure -> format_floor.
`score_fn(items) -> list[float]` scores [(task, lang, pin, source, final_text, devices)] on the evaluator's 1-5 scale (mean of the task's dimensions); tests pass a fake one."""
from __future__ import annotations
import re
from . import rules as R, diacritics as DI, lid as LD
from .config import RewardConfig

ALLOW_EN = {"hm"}                                        # the switch language of hm is English: English words are expected there


class Reward:
    def __init__(self, score_fn, cfg: RewardConfig | None = None, funcwords=None, number_words: dict | None = None, lid=None, group: int = 0, qe_fn=None, base_floors: dict | None = None, lang_windows: dict | None = None, allowed_switch: dict | None = None):
        self.score_fn, self.cfg, self.fw, self.nw, self.lid, self.group, self.qe_fn, self.floors = score_fn, cfg or RewardConfig(), funcwords, number_words or {}, lid, group, qe_fn, base_floors or {}; self.lw = lang_windows or {}; self.asw = allowed_switch or {}

    def __call__(self, items: list[dict], completions: list[str]) -> tuple[list[float], list[dict]]:
        """items[i]: dict(task, lang, pin, source (text shown in the prompt), eng, switch_lang). Returns (rewards, details)."""
        c = self.cfg; parsed = [R.parse(x) for x in completions]
        for it, p in zip(items, parsed):
            if it["task"] == "eng_m": p["final"] = re.sub(r"^[a-z]{3}\s+", "", p["final"])             # replay translation: completion is "<lang code> <text>"
        rewards = [0.0] * len(items); details = [dict() for _ in items]; todo = []
        for i, (it, p) in enumerate(zip(items, parsed)):
            pen, d = R.format_penalty(p, it["task"]); details[i].update(d)
            if pen: rewards[i] = c.format_floor; details[i]["gate"] = True; continue
            tgt = it["task"].split("_")[1]; lo, hi = self.lw.get(f'{it["task"]}|{it["lang"]}') or c.length_windows[it["task"]]; pens = {}
            pens["length"], dl = R.length_penalty(p["final"], it["source"], lo, hi); details[i].update(dl)
            pens["loop"], dp = R.loop_penalty(p["final"]); details[i].update(dp)
            pens["number"], dn = R.number_penalty(p["final"], it["source"], R.EN_NUM | set(self.nw.get(it["lang"], ()))); details[i].update(dn)
            sw0 = it.get("switch_lang", "eng"); hm_non_en = tgt == "hm" and sw0 != "eng"; pens["english"], de = R.english_leak(p["final"], (tgt in ALLOW_EN and not hm_non_en) or it["lang"] in c.english_ok_langs or (tgt == "ood" and "eng" in self.asw.get(it["lang"], [])), cap=(0.12 if hm_non_en else c.english_cap)); details[i].update(de)
            pens["draft"], dd = R.draft_penalty(p["draft"], it["source"], c.draft_cap); details[i].update(dd)
            sw = it.get("switch_lang", "eng"); base_txt = R.strip_devices(p["final"], p["devices"]) if tgt == "hm" else p["final"]
            pens["diacritic"], dd2 = DI.penalty(base_txt, it["lang"]); details[i].update(dd2)
            en_ok = it["lang"] in c.english_ok_langs
            if en_ok: lid_extra = {"eng"}                                                          # English is (close to) native in the Pidgins / Afrikaans
            elif it["task"] in ("eng_hm", "m_hm"): lid_extra = {sw}                                          # the switch language given in the prompt
            else: lid_extra = set(self.asw.get(it["lang"], [])) | ({sw} if it["task"] == "hm_ood" else set())     # ood: every language the base language is allowed to switch with is acceptable (plus the prompt's switch language for hm_ood)
            pens["lid"], dli = LD.lid_penalty(self.lid, p["final"], it["lang"], lid_extra, "hm" if (en_ok or tgt == "hm") else tgt, tol=c.lid_tol_hm if tgt == "hm" else c.lid_tol); details[i].update(dli)
            pens["name"], dnm = R.name_penalty(p["final"], it["source"]) if it["task"].startswith("eng_") else (0.0, {}); details[i].update(dnm)
            pens["device_use"], ddu = R.device_use(p["final"], p["devices"]) if it["task"] != "eng_m" else (0.0, {}); details[i].update(ddu)
            if tgt == "hm" and not en_ok:
                pens["lid_switch"], dls = LD.switch_lid_penalty(LD.window_share(self.lid, p["final"], sw), *c.switch_share); details[i].update(dls)
            else: pens["lid_switch"] = 0.0
            pens["base_presence"], dbp = R.base_presence(base_txt, it["lang"], self.fw, self.floors) if it["task"] != "eng_m" else (0.0, {}); details[i].update(dbp)
            if tgt == "hm": pens["base_leak"], dbl = R.base_leak(base_txt, sw, self.fw, cap=c.base_leak_cap); details[i].update(dbl)
            else: pens["base_leak"] = 0.0
            if tgt == "hm": pens["switch"], ds = R.switch_penalty(p["final"], p["devices"], *c.switch_share, fw=self.fw, switch_lang=it.get("switch_lang", "eng")); details[i].update(ds)
            else: pens["switch"] = 0.0
            if self.group and i % self.group == self.group - 1:                                  # greedy member: its own failures count more
                for k in c.greedy_pen_rules: pens[k] = min(1.0, pens[k] * c.greedy_pen_scale)
                details[i]["greedy"] = True
            ded = min(c.rule_cap, sum(c.rule_weights[k] * v for k, v in pens.items())); details[i].update({"rule_deduction": round(ded, 4), "pens": {k: round(v, 3) for k, v in pens.items()}}); rewards[i] = -ded; rewards[i] = max(rewards[i], c.format_floor) if False else rewards[i]
            todo.append(i)
        if todo:
            sc = self.score_fn([(items[i]["task"], items[i]["lang"], items[i].get("pin", ""), items[i]["source"], parsed[i]["final"], parsed[i]["devices"]) for i in todo])
            qe = {}
            if self.qe_fn is not None:
                qi = [i for i in todo if items[i]["task"] == "eng_m" and items[i]["lang"] in c.qe_langs]
                if qi:
                    for i, q in zip(qi, self.qe_fn([(items[i]["source"], parsed[i]["final"]) for i in qi])): qe[i] = min(1.0, max(0.0, float(q)))
            for i, s in zip(todo, sc):
                base = (float(s) - 1.0) / 4.0; details[i]["evaluator"] = round(float(s), 4)
                if i in qe: details[i]["qe"] = round(qe[i], 4); base = (1.0 - c.qe_weight) * base + c.qe_weight * qe[i]
                rewards[i] = max(c.format_floor, rewards[i] + c.evaluator_weight * base)
        return rewards, details
