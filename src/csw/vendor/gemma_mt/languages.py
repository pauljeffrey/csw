# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ONE canonical language table of this module.

Every other file asks this table for (a) the tag that is written in front of the translation, (b) the English
name that is written in the instruction, and (c) whether the AfriCOMET encoder covers the language, so that the
RL stage only ever trains on pairs the reward can judge.

Naming rule (fixed by the owner, see DECISIONS.md D-02): the SHORT English name, WITHOUT a Bantu noun-class
prefix and WITHOUT the autonym: "Pidgin" (not "Nigerian Pidgin"), "Zulu" (not "isiZulu"), "Xhosa", "Chichewa",
"Sesotho", "Setswana", "Efik", "Fon". One name per language, used in the instruction, in the model card and in
reports. Never change a name after a model has been trained: the name is
part of the prompt the model learned.

AfriCOMET support follows the repo files this module was built from:
  * rl/bin_data.py                              AFRICOMET_CODE
  * data-gen/africomet_translation/sources.py   AFRICOMET_SUPPORTED   (Ibibio is NOT supported)
Languages that appear in neither file are marked "unverified": they are valid SFT targets, but they are kept out of
`SUPPORTED_RL` until someone checks the AfriCOMET model card (OPEN QUESTION Q-05 in DECISIONS.md).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# status values
VERIFIED = "verified"        # listed in the repo's AFRICOMET_SUPPORTED / AFRICOMET_CODE
UNVERIFIED = "unverified"    # African language we want as an SFT target; AfriCOMET coverage not confirmed from the repo
UNSUPPORTED = "unsupported"  # confirmed NOT covered by the AfriCOMET encoder
LISTED = "listed"            # D-37: named on the AfriCOMET-QE-1.1 model card, but NOT validated against human references (untested, or measured below the usability bar):
                             # RL-trained with the QE blended with the consensus signal at a low advantage weight (calibration `content` block)


@dataclass(frozen=True)
class Language:
    iso: str                           # tag written in front of the translation (our ISO-639-3 style tag)
    name: str                          # English display name written in the instruction (fixed)
    africomet_code: Optional[str]      # code AfriCOMET / the repo uses (None = not covered or unknown)
    africomet_status: str              # VERIFIED | UNVERIFIED | UNSUPPORTED
    glotlid: Tuple[str, ...] = ()      # GlotLID labels accepted as "this language" (empty = no LID guard possible)
    flores: Optional[str] = None       # FLORES+ config / language code (openlanguagedata/flores_plus), e.g. "yor_Latn"
    ntrex: Optional[str] = None        # NTREX-128 file code (newstest2019-ref.<code>.txt)
    mafand: Optional[str] = None       # MAFAND-MT target code (config "en-<code>")
    rl_consensus: bool = False         # D-37: not covered by AfriCOMET at all, but an SFT target that is RL-trained with the QE-free consensus signal only
    smol_sent: Optional[str] = None    # D-42: google/smol SmolSent file code (smolsent/en_<code>.jsonl); SMOL has its own codes (ISO-639-1 where one exists, `ber-Latn` for Kabyle ...)
    smol_doc: Optional[str] = None     # D-42: google/smol SmolDoc file code (smoldoc/en_<code>.jsonl)
    gatitos: Optional[str] = None      # D-42: google/smol GATITOS file code (gatitos/en_<code>.jsonl), the 4,000-entry word-level lexicon
    ibom_mt: Optional[str] = None      # D-42: howard-nlp/ibom-mt config `eng-<code>` (test split, FLORES+ devtest English)
    flores_moz: Optional[str] = None   # D-42: MOZNLP/FLORES_MOZ config (devtest), e.g. "seh_Latn_MZ"
    glot_story: Optional[str] = None   # D-42: cis-lmu/GlotStoryBook `ISO639-3` column value (African Storybook Project; its `Language` column has the ASP codes hz / kj / nr)
    salt: Optional[str] = None         # D-42: Sunbird/salt `text-all` test column prefix (`<code>_text`)

    @property
    def qe_covered(self) -> bool:
        """AfriCOMET-QE may score this language (named on the card): VERIFIED (validated against human references, 42 languages) or LISTED (D-37)."""
        return self.africomet_status in (VERIFIED, LISTED) and self.africomet_code is not None

    @property
    def rl_eligible(self) -> bool:
        """Trained in RL: QE-covered, or an SFT target trained with the consensus signal only (D-37)."""
        return self.qe_covered or self.rl_consensus


# Order = display order in reports. GlotLID label sets are copied from data-gen/africomet_translation/lid.py (EXPECTED).
# Benchmark codes: FLORES+ names are FLORES-200 style; NTREX-128 and MAFAND-MT codes are ISO-639-3 with `swa` for
# Swahili. Everything not in the benchmark's own language list is simply absent at load time (the adapters report
# "no data" per language); the codes below are best effort and are re-checked on the GPU box with
# `python -m gemma_mt.eval.benchmarks --list ...` (OPEN QUESTION Q-07).
_LANGUAGES = (
    # --- West Africa
    Language("yor", "Yoruba", "yo", VERIFIED, ("yor_Latn",), "yor_Latn", "yor", "yor"),
    Language("hau", "Hausa", "ha", VERIFIED, ("hau_Latn",), "hau_Latn", "hau", "hau"),
    Language("ibo", "Igbo", "ig", VERIFIED, ("ibo_Latn",), "ibo_Latn", "ibo", "ibo"),
    Language("pcm", "Pidgin", "pcm", VERIFIED, ("pcm_Latn", "wes_Latn"), None, None, "pcm"),
    Language("efi", "Efik", "efi", VERIFIED, ("efi_Latn", "ibb_Latn")),
    Language("twi", "Twi", "tw", VERIFIED, ("twi_Latn", "aka_Latn"), "twi_Latn_akua1239", None, "twi"),
    Language("ewe", "Ewe", "ee", VERIFIED, ("ewe_Latn",), "ewe_Latn", "ewe", None),
    Language("fon", "Fon", "fon", VERIFIED, ("fon_Latn",), "fon_Latn", None, None),
    Language("wol", "Wolof", "wo", VERIFIED, ("wol_Latn",), "wol_Latn", "wol", None),
    # --- East / Central Africa
    Language("swh", "Swahili", "sw", VERIFIED, ("swh_Latn",), "swh_Latn", "swa", "swa"),
    Language("kin", "Kinyarwanda", "rw", VERIFIED, ("kin_Latn",), "kin_Latn", "kin", "kin"),
    Language("som", "Somali", "so", VERIFIED, ("som_Latn",), "som_Latn", "som", None),
    Language("orm", "Oromo", "om", VERIFIED, ("gaz_Latn", "orm_Latn"), "gaz_Latn", "orm", None),
    Language("amh", "Amharic", "amh", VERIFIED, ("amh_Ethi",), "amh_Ethi", "amh", "amh"),
    Language("tir", "Tigrinya", "tir", VERIFIED, ("tir_Ethi",), "tir_Ethi", "tir", None),
    Language("lug", "Luganda", "lug", VERIFIED, ("lug_Latn",), "lug_Latn", "lug", "lug"),
    Language("lin", "Lingala", "lin", VERIFIED, ("lin_Latn",), "lin_Latn", "lin", None),
    # --- Southern Africa
    Language("zul", "Zulu", "zu", VERIFIED, ("zul_Latn",), "zul_Latn", "zul", "zul"),
    Language("xho", "Xhosa", "xh", VERIFIED, ("xho_Latn",), "xho_Latn", "xho", "xho"),
    Language("afr", "Afrikaans", "af", VERIFIED, ("afr_Latn",), "afr_Latn", "afr", None),
    Language("sna", "Shona", "sn", VERIFIED, ("sna_Latn",), "sna_Latn", "sna-Latn", "sna"),
    Language("nya", "Chichewa", "ny", VERIFIED, ("nya_Latn",), "nya_Latn", "nya", "nya"),
    Language("sot", "Sesotho", "st", VERIFIED, ("sot_Latn",), "sot_Latn", "sot", None),
    Language("tsn", "Setswana", "tn", VERIFIED, ("tsn_Latn",), "tsn_Latn", "tsn", "tsn"),
    Language("mlg", "Malagasy", "mlg", VERIFIED, ("plt_Latn",), "plt_Latn", "mlg", None),
    # --- valid SFT targets that AfriCOMET does NOT cover (never RL-trained)
    Language("ibb", "Ibibio", None, UNSUPPORTED, ("efi_Latn", "ibb_Latn")),
    Language("urh", "Urhobo", None, UNSUPPORTED, ("urh_Latn",)),
)

# --- added 2026-10-12 (owner: "get all you can"): every other African language for which the English->X pairs of the build have data.
# SFT targets only: AfriCOMET coverage is not confirmed, so none of them is RL-eligible. Names are the short English names, fixed once.
_SFT_ONLY = (
    ("bem", "Bemba"), ("tum", "Tumbuka"), ("run", "Kirundi"), ("ven", "Venda"), ("tso", "Tsonga"), ("nso", "Sepedi"), ("ssw", "Swati"),
    ("nbl", "South Ndebele"), ("nde", "North Ndebele"), ("sag", "Sango"), ("tiv", "Tiv"), ("umb", "Umbundu"), ("kmb", "Kimbundu"),
    ("mos", "Mossi"), ("kik", "Kikuyu"), ("luo", "Luo"), ("kam", "Kamba"), ("bin", "Edo"), ("dag", "Dagbani"), ("oke", "Okpe"),
    ("shi", "Tachelhit"), ("dyu", "Dyula"), ("bam", "Bambara"), ("kbp", "Kabiye"), ("bbj", "Ghomala"), ("kon", "Kongo"), ("kea", "Kabuverdianu"),
    ("fat", "Fante"), ("knc", "Kanuri"), ("gaa", "Ga"), ("kab", "Kabyle"), ("nus", "Nuer"), ("swc", "Congo Swahili"), ("lua", "Luba-Kasai"),
    ("kqn", "Kaonde"), ("lun", "Lunda"), ("iso", "Isoko"), ("loz", "Lozi"), ("wal", "Wolaytta"), ("ach", "Acholi"), ("tsc", "Tswa"),
    ("vmw", "Makhuwa"), ("kua", "Kuanyama"), ("seh", "Sena"), ("nyn", "Nyankore"), ("her", "Herero"), ("yao", "Yao"), ("sid", "Sidamo"),
    ("ndc", "Ndau"), ("men", "Mende"), ("kri", "Krio"), ("luy", "Luyia"), ("ish", "Esan"), ("dua", "Duala"), ("bas", "Basaa"),
    ("wes", "Cameroon Pidgin"), ("tzm", "Tamazight"), ("ful", "Fulah"),
)
# --- promoted to RL-eligible 2026-10-17 (owner decision 3/D-32): the 17 languages of the SSA-COMET-vs-AfriCOMET validity study with human-vs-mismatched
# AUC >= 0.80 plus the 5 "marginal" ones (0.70-0.80), see reports/ssa_comet_vs_africomet_2026-10-15.md (section 4). Justification: the AfriCOMET-QE-1.1
# model card lists the same 76 languages as the encoder (Davlan/afro-xlmr-large-76L), and COMET QE takes TEXT ONLY (no language code), so "can AfriCOMET
# score X" is a validity policy of this table, not a model limit; the measured AUC IS that validity test. africomet_code = our tag.
# value: (GlotLID labels accepted, FLORES+ config). GlotLID v3 labels were checked against `cis-lmu/glotlid` model_v3.bin (2,102 labels):
# Ethiopic-script languages are amh_Ethi / tir_Ethi, Malagasy is plt_Latn (no mlg_Latn label), Kikuyu kik_Latn, Nuer nus_Latn ...
# NOT promoted: dyu sag nqo (AUC < 0.70) and fuv knc taq cjk dik (no SFT coverage).
_RL_PROMOTED_2026_10_17 = {
    "run": (("run_Latn",), "run_Latn"), "bem": (("bem_Latn",), "bem_Latn"), "tso": (("tso_Latn",), "tso_Latn"),
    "nso": (("nso_Latn",), "nso_Latn"), "tum": (("tum_Latn",), "tum_Latn"), "mos": (("mos_Latn",), "mos_Latn"),
    "kik": (("kik_Latn",), "kik_Latn"), "luo": (("luo_Latn",), "luo_Latn"), "bam": (("bam_Latn",), "bam_Latn"),
    "kmb": (("kmb_Latn",), "kmb_Latn"), "ssw": (("ssw_Latn",), "ssw_Latn"), "lua": (("lua_Latn",), "lua_Latn"),
    "umb": (("umb_Latn",), "umb_Latn"), "kab": (("kab_Latn",), "kab_Latn"), "kam": (("kam_Latn",), "kam_Latn"),
    "kbp": (("kbp_Latn",), "kbp_Latn"), "nus": (("nus_Latn",), "nus_Latn"),
}
# (the other five of the 22 -- amh tir lin lug mlg -- are rows of the main table above)


# Benchmark codes of SFT-only / promoted languages, checked 2026-10-18 against the real benchmark listings (FLORES+ devtest file list of
# openlanguagedata/flores_plus, GitHub MicrosoftTranslator/NTREX NTREX-128, masakhane-io/lafand-mt data/json_files): (flores, ntrex, mafand).
# Not covered by any of the three: tum, tiv, bin, dag, oke, bbj, fat, gaa, swc, kqn, lun, iso, loz, wal, ach, tsc, kua, seh, nyn, her, yao, sid, ndc, men, kri,
# luy, ish, dua, bas, wes, tzm (FLORES+ has zgh_Tfng = Standard Moroccan Tamazight, a different variety), ful (FLORES+ fuv_Latn / NTREX fuc are specific
# varieties), nbl, kon (no FLORES+ file in the Hub listing).
_BENCH_CODES_SFT_ONLY = {
    "bem": (None, "bem", None), "nde": (None, "nde", None), "nso": (None, "nso", None), "shi": (None, "shi", None), "ssw": (None, "ssw", None),
    "ven": (None, "ven", None), "luo": (None, None, "luo"), "kea": ("kea_Latn", None, None), "sag": ("sag_Latn", None, None),
    "dyu": ("dyu_Latn", None, None), "knc": ("knc_Latn", None, None), "vmw": ("vmw_Latn", None, None),
}


# --- D-37 (owner 2026-10-19): every SFT target that the AfriCOMET-QE-1.1 model card NAMES and that was not yet an RL language joins RL as a QE-BLENDED language
# (low advantage weight; `content` block of the calibration file). The card (fetched 2026-10-19; 75 languages are printed although the text says 76 / the metadata 77)
# names: loz ven iso tiv kqn wal lun kua kon dyu sag (and gaa knc tzm fuv cjk dik taq nqo, which are not among the 77 trained SFT targets). `swc` (Congo Swahili) is not
# on the card but IS Swahili (swa), which the card names and which is validated (AUC 0.96): it is LISTED through swh (anchors of swh, no own calibration).
_RL_LISTED_2026_10_19 = ("loz", "ven", "iso", "tiv", "kqn", "wal", "lun", "kua", "kon", "dyu", "sag", "swc")
# --- D-37: the 23 SFT targets that the card does not name at all are RL-trained with the consensus signal only (no QE call). Of the 85 table rows, nde fat knc gaa men luy tzm ful
# have no SFT training data (the SFT run has 77 target languages) and are not RL languages.
_RL_CONSENSUS_ONLY = ("ibb", "urh", "nbl", "bin", "dag", "oke", "shi", "bbj", "kea", "ach", "tsc", "vmw", "seh", "nyn", "her", "yao", "sid", "ndc", "kri", "ish",
                      "dua", "bas", "wes")
_RL_GLOTLID_ALT = {"swc": ("swc_Latn", "swh_Latn"), "ibb": ("ibb_Latn", "efi_Latn"), "wes": ("wes_Latn", "pcm_Latn")}


def _sft_only_row(i: str, n: str) -> Language:
    b_flores, b_ntrex, b_mafand = _BENCH_CODES_SFT_ONLY.get(i, (None, None, None))
    if i in _RL_PROMOTED_2026_10_17:
        glot, flores = _RL_PROMOTED_2026_10_17[i]
        return Language(i, n, i, VERIFIED, glot, flores, b_ntrex, b_mafand)
    if i in _RL_LISTED_2026_10_19:
        return Language(i, n, i, LISTED, _RL_GLOTLID_ALT.get(i, (f"{i}_Latn",)), b_flores, b_ntrex, b_mafand)
    return Language(i, n, None, UNVERIFIED, (f"{i}_Latn",), b_flores, b_ntrex, b_mafand, rl_consensus=i in _RL_CONSENSUS_ONLY)


_LANGUAGES = _LANGUAGES + tuple(_sft_only_row(i, n) for i, n in _SFT_ONLY)      # same order as before: language ids (iso_to_id) are unchanged
# the main-table rows ibb and urh (AfriCOMET UNSUPPORTED) are RL-trained with the consensus signal only as well (D-37)
_LANGUAGES = tuple(dataclasses.replace(l, rl_consensus=True) if l.iso in _RL_CONSENSUS_ONLY and not l.rl_consensus else l for l in _LANGUAGES)

# --- D-42 (owner 2026-10-19): Google SMOL / GATITOS test data (arXiv 2502.12301; Hub `google/smol`, CC-BY-4.0, pinned in eval/benchmarks.py).
# (SmolSent file code, SmolDoc file code, GATITOS file code) of every SFT target the three resources contain; None = the file does not exist in the
# pinned revision (fdaff3a). Every mapping was checked on the SENTENCES, not only on the code: GlotLID top-1 of the 863 SmolSent references and the token
# coverage of those references in the training targets of each of our languages (reports/gemma_rare_language_benchmarks_2026-10-19.md, section 2).
# Not mapped on purpose: `lu` (Luba-KATANGA, GlotLID lub; our `lua` is Luba-Kasai, which only GATITOS has as `lua`), `ktu` (Kituba; `kg` = Kikongo is `kon`),
# `ndc-ZW` (byte-identical to `ndc`), `ber` (Tifinagh Central Atlas Tamazight; `tzm` has no SFT data), `nd` / `kr` / `ff` / `fuc` / `gaa` (nde knc ful gaa have no SFT data).
# Caveats: `nyn` <- SMOL `cgg` (Rukiga/Chiga, the closest language of the Runyankore-Rukiga cluster: GlotLID says 69% nyn, token coverage in our nyn targets 0.47
# vs 0.28 for the next language) is a PROXY; `twi` <- `ak` (Akan: GlotLID twi 100%, coverage 0.96); `ndc` <- `ndc` is Shona-like text (GlotLID sna 62%).
_BENCH_CODES_SMOL = {
    "ach": ("ach", "ach", "ach"), "afr": ("af", "af", None), "amh": ("am", "am", None), "bam": ("bm", "bm", "bm"), "bem": ("bem", "bem", "bem"),
    "dyu": ("dyu", "dyu", "dyu"), "efi": ("efi", "efi", "efi"), "ewe": ("ee", "ee", "ee"), "fon": ("fon", "fon", "fon"), "hau": ("ha", "ha", None),
    "ibo": ("ig", "ig", None), "iso": (None, None, "iso"), "kab": ("ber-Latn", "ber-Latn", "ber-Latn"), "kik": ("ki", "ki", "ki"), "kin": ("rw", "rw", None),
    "kon": ("kg", "kg", "kg"), "kri": ("kri", "kri", "kri"), "lin": ("ln", "ln", "ln"), "lua": (None, None, "lua"), "lug": ("lg", "lg", "lg"),
    "luo": ("luo", "luo", "luo"), "mlg": ("mg", "mg", None), "mos": ("mos", "mos", "mos"), "nbl": ("nr", "nr", "nr"), "ndc": ("ndc", "ndc", "ndc"),
    "nso": ("nso", "nso", "nso"), "nus": (None, None, "nus"), "nya": ("ny", "ny", None), "nyn": ("cgg", "cgg", "cgg"), "orm": ("om", "om", "om"),
    "pcm": ("pcm", "pcm", "pcm"), "run": ("rn", "rn", "rn"), "sag": (None, None, "sg"), "sna": ("sn", "sn", None), "som": ("so", "so", None),
    "sot": ("st", "st", None), "ssw": ("ss", "ss", "ss"), "swh": ("sw", "sw", None), "tir": ("ti", "ti", "ti"), "tiv": ("tiv", "tiv", "tiv"),
    "tsn": ("tn", "tn", "tn"), "tso": ("ts", "ts", "ts"), "tum": ("tum", "tum", "tum"), "twi": ("ak", "ak", "ak"), "ven": ("ve", "ve", "ve"),
    "wol": ("wo", "wo", "wo"), "xho": ("xh", "xh", None), "yor": ("yo", "yo", None), "zul": ("zu", "zu", None),
}
_LANGUAGES = tuple(dataclasses.replace(l, smol_sent=_BENCH_CODES_SMOL[l.iso][0], smol_doc=_BENCH_CODES_SMOL[l.iso][1], gatitos=_BENCH_CODES_SMOL[l.iso][2])
                   if l.iso in _BENCH_CODES_SMOL else l for l in _LANGUAGES)

# --- D-42: four more small public test sets found for languages that had none (reports/gemma_rare_language_benchmarks_2026-10-19.md, section 5). All English -> X.
#   ibom_mt     IbomMT (CC-BY-NC-4.0): Ibibio, Efik; 1,012 sentences, the FLORES+ devtest English; translated by paid linguists.
#   flores_moz  FLORES-MOZ (CC-BY-SA-4.0): Sena (Mozambique); the FLORES+ devtest English (1,012), professional translators.
#   glot_story  GlotStoryBook (per-row CC-BY / CC-BY-NC): African Storybook Project children's stories aligned to English by (story, sentence number).
#   salt        Sunbird SALT `text-all` test (CC-BY-SA-4.0, gated auto-accept): 500 agricultural sentences in Runyankore, Acholi, Luganda, Igbo, Swahili.
_BENCH_CODES_OTHER = {          # iso: {benchmark: code}
    "ibb": {"ibom_mt": "ibb"}, "efi": {"ibom_mt": "efi"}, "seh": {"flores_moz": "seh_Latn_MZ"},
    "kqn": {"glot_story": "kqn"}, "lun": {"glot_story": "lun"}, "loz": {"glot_story": "loz"}, "kua": {"glot_story": "kua"}, "her": {"glot_story": "her"},
    "dag": {"glot_story": "dag"}, "tsc": {"glot_story": "tsc"}, "nbl": {"glot_story": "nbl"},
    "nyn": {"salt": "nyn"}, "ach": {"salt": "ach"}, "lug": {"salt": "lug"}, "ibo": {"salt": "ibo"}, "swh": {"salt": "swa"},
}
_LANGUAGES = tuple(dataclasses.replace(l, **_BENCH_CODES_OTHER[l.iso]) if l.iso in _BENCH_CODES_OTHER else l for l in _LANGUAGES)

ENGLISH_NAME = "English"
SOURCE_AFRICOMET_CODE = "en"

# English is the PIVOT of the reverse (African -> English) direction. It is deliberately not a row of the table:
# it must never get a language id, an RL slot or a benchmark code. It only has a tag and a name in the instruction.
PIVOT_ISO = "eng"
PIVOT_NAME = ENGLISH_NAME
_PIVOT_SPELLINGS = {"eng", "en", "english"}

LANGUAGES: Dict[str, Language] = {l.iso: l for l in _LANGUAGES}
ALL_ISOS: Tuple[str, ...] = tuple(LANGUAGES)

# Accepted spellings -> canonical tag. Only used when READING data; the model is always trained with the canonical tag.
ALIASES: Dict[str, str] = {
    "swa": "swh", "sw": "swh", "ti": "tir", "mg": "mlg", "yo": "yor", "ha": "hau", "ig": "ibo", "zu": "zul",
    "xh": "xho", "af": "afr", "am": "amh", "so": "som", "rw": "kin", "sn": "sna", "ny": "nya", "st": "sot", "bm": "bam", "ff": "ful",
    "tn": "tsn", "om": "orm", "wo": "wol", "ee": "ewe", "tw": "twi", "aka": "twi", "ln": "lin", "lg": "lug",
}

# Languages trained in RL (D-37: 77 = the SFT targets with data): 42 VALIDATED for QE (data + repo-verified / measured AfriCOMET code, D-29/D-32), 12 LISTED on the
# AfriCOMET card but unvalidated or weak (QE blended with consensus), 23 outside AfriCOMET (consensus only). Before D-37 this was the 42 validated ones.
SUPPORTED_RL: Tuple[str, ...] = tuple(l.iso for l in _LANGUAGES if l.rl_eligible)
QE_VALIDATED: Tuple[str, ...] = tuple(l.iso for l in _LANGUAGES if l.africomet_status == VERIFIED and l.africomet_code is not None)     # the 42 of D-29 / D-32
QE_LANGUAGES: Tuple[str, ...] = tuple(l.iso for l in _LANGUAGES if l.qe_covered)       # languages AfriCOMET-QE may score (42 validated + 12 listed)
CONSENSUS_ONLY: Tuple[str, ...] = tuple(l.iso for l in _LANGUAGES if l.rl_consensus)   # RL languages without any QE call
AFRICOMET_CODE: Dict[str, str] = {l.iso: l.africomet_code for l in _LANGUAGES if l.africomet_code}


def canonical_iso(tag: str) -> Optional[str]:
    """Canonical tag for a tag spelled in any accepted way; None when we do not know the language."""
    if tag is None:
        return None
    t = str(tag).strip().lower().replace("_", "-")
    if t in LANGUAGES:
        return t
    t = t.split("-")[0]
    if t in LANGUAGES:
        return t
    return ALIASES.get(t)


def is_pivot(tag: str) -> bool:
    return tag is not None and str(tag).strip().lower() in _PIVOT_SPELLINGS


def target_tag_and_name(tag: str) -> Tuple[str, str]:
    """(tag written in front of the model's answer, name written in the instruction) for a TARGET language: an African
    language of the table, or the English pivot (reverse direction)."""
    if is_pivot(tag):
        return PIVOT_ISO, PIVOT_NAME
    lang = get_language(tag)
    return lang.iso, lang.name


def get_language(tag: str) -> Language:
    iso = canonical_iso(tag)
    if iso is None:
        raise KeyError(f"unknown language tag {tag!r}; known tags: {', '.join(ALL_ISOS)}")
    return LANGUAGES[iso]


def display_name(tag: str) -> str:
    """The English name written in the instruction for this language (fixed, one per language)."""
    return get_language(tag).name


def iso_to_id(tag: str) -> int:
    """Stable small integer id of a language (position in ALL_ISOS); stored in packed-data indexes."""
    return ALL_ISOS.index(get_language(tag).iso)


def id_to_iso(i: int) -> str:
    return ALL_ISOS[int(i)]


def is_rl_supported(tag: str) -> bool:
    iso = canonical_iso(tag)
    return iso is not None and LANGUAGES[iso].rl_eligible


def table_markdown() -> str:
    """The table as Markdown (used by the model card and the README)."""
    rows = ["| tag | name in the instruction | AfriCOMET | RL |", "|---|---|---|---|"]
    for l in _LANGUAGES:
        rows.append(f"| `{l.iso}` | {l.name} | {l.africomet_code or '-'} ({l.africomet_status}) | "
                    f"{'yes' if l.rl_eligible else 'no'} |")
    return "\n".join(rows)
