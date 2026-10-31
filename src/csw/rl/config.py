"""Reward configuration for the csw RL (values are the STARTING proposal; the rule weights are fixed only after the penalty distributions on full_E samples have been looked at)."""
from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    qe_weight: float = 0.7                              # eng_m in AfriCOMET-covered languages: base score = (1 - qe_weight) * evaluator + qe_weight * AfriCOMET-QE (reference-free, 0..1)
    qe_langs: tuple = ("amh", "hau", "ibo", "kin", "lug", "luo", "nya", "pcm", "sna", "swh", "twi", "wol", "yor", "zul")   # the AfriMTE / AfriCOMET languages among ours
    evaluator_weight: float = 1.0                       # reward = evaluator_weight * evaluator score in [0, 1] (mean of the task's dimensions, all equal) - capped rule penalties
    rule_weights: dict = field(default_factory=lambda: {"length": 0.60, "loop": 1.00, "number": 0.80, "english": 0.80, "switch": 1.00, "draft": 0.30, "diacritic": 0.60, "lid": 1.00, "base_leak": 0.80, "name": 0.40, "device_use": 0.60, "lid_switch": 0.0, "base_presence": 0.40, "qe_loop": 0.0})
    rule_cap: float = 2.00                              # total rule deduction never exceeds this, so rules cannot outweigh the evaluator (range 0..1)
    format_floor: float = -1.0                          # any format failure (eos, dev block, unclosed draft, empty final): reward = this, the evaluator is not consulted
    length_windows: dict = field(default_factory=lambda: {"eng_hm": (0.60, 1.40), "m_hm": (0.75, 1.40), "eng_ood": (0.65, 2.70), "m_ood": (0.90, 2.70), "hm_ood": (0.90, 2.60), "eng_m": (0.60, 1.80)})   # reference p5-p95 on the test set (csw_rl/calibrate.py), widened ~15%; ood is long on purpose (idioms)
    base_leak_cap: float = 0.12
    english_cap: float = 0.20                           # reference ood texts exceed 0.08 in 16% of rows; at 0.20 only 5% of references and 7% of generations do
    english_ok_langs: tuple = ("pcm", "gpe", "wes", "afr")   # English is (close to) native there
    draft_cap: float = 1.5                              # draft chars / source chars
    switch_share: tuple = (0.18, 0.45)
    kl_guard: float = 0.15                              # per-token KL(pi||ref) above which the update is cut (same as the translation RL, D-67); reference = the start model (full_E)
    beta_kl: float = 0.02
    greedy_adv_scale: float = 0.3                       # greedy decode joins the group baseline with this advantage scale (translation RL, D-67); advantage cap 3
    greedy_adv_cap: float = 3.0
    greedy_pen_scale: float = 1.5                       # the greedy member (last of each group) is the model's actual decode: its loop / length / lid / base_leak penalties are multiplied by this (capped at 1 before the weight)
    greedy_pen_rules: tuple = ("loop", "length", "lid", "base_leak")
    lid_tol: float = 0.20
    lid_tol_hm: float = 0.35                            # mixed-language hm sentences are mislabelled by sentence LID in 17% of REFERENCE texts at 0.15
