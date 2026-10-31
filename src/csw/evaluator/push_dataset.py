"""Stage and push every training / label / test file of the csw + evaluator work to the private dataset repo.
python -m csw.evaluator.push_dataset --repo AletheiaAING/African-Code-Switch --stage work/hub_stage [--no-push]"""
from csw.paths import env_value
import argparse, os, shutil
from pathlib import Path

W = Path("work")
TRAIN = {  # destination under stage -> source
    "csw_sft/final_v5/train.jsonl": W / "final_v5/C/train.jsonl", "csw_sft/final_v5/val.jsonl": W / "final_v5/C/val.jsonl",
    "csw_sft/corrections/csw_edits_train.jsonl": W / "csw_edits/train.jsonl", "csw_sft/corrections/csw_edits_ft_train.jsonl": W / "csw_edits_ft/train.jsonl", "csw_sft/corrections/csw_edits_ft_val.jsonl": W / "csw_edits_ft/val.jsonl",
    "evaluator/stage_v3/train.jsonl": W / "qe_data_v3/train.jsonl", "evaluator/unlabelled_generated_rows.jsonl": W / "cft4/unlabelled.jsonl",
}
for s in ("cft1", "cft2a", "cft2b", "cft3", "cft4"):
    TRAIN[f"evaluator/stage_{s}/train.jsonl"] = W / s / "train.jsonl"; TRAIN[f"evaluator/stage_{s}/val.jsonl"] = W / s / "val.jsonl"
    TRAIN[f"test/evaluator/stage_{s}/test.jsonl"] = W / s / "test.jsonl"; TRAIN[f"test/evaluator/stage_{s}/heldout_chains.jsonl"] = W / s / "heldout_chains.jsonl"
TRAIN["test/evaluator/stage_v3/test.jsonl"] = W / "qe_data_v3/test.jsonl"
TRAIN["test/csw_sft/final_v5/test_off.jsonl"] = W / "final_v5/C/test_off.jsonl"; TRAIN["test/csw_sft/final_v5/test_on.jsonl"] = W / "final_v5/C/test_on.jsonl"
TRAIN["test/csw_sft/test_confirmed_roots.json"] = W / "test_confirmed_roots.json"; TRAIN["test/csw_sft/test_tier2_roots.json"] = W / "test_tier2_roots.json"
TRAIN["test/gold_299/pilot_labels.json"] = W / "pilot_labels.json"
LABEL_DIRS = ["stage2_both", "label_nb", "chains_tr", "chains_tr_rules", "chains_hm", "chains_ood", "replay_flash", "label_more", "regrade_argon", "regrade_flash_strict"]
KEEP = ("labels.jsonl", "labels_old.jsonl", "pairs.jsonl", "records.jsonl", "meta.json", "estimate.json", "analysis.txt")
README = """# African-Code-Switch (private working copy)

Training data, judge labels, corrections and gold test sets for the csw code-switch / idiom model and its evaluator (mmBERT-base, 27 graded outputs).
Not included: the human-rated SSA-MTE / AfriMTE rows (CC-BY-SA-4.0 / CC-BY-4.0, McGill-NLP/SSA-MTE and masakhane/AfriMTE-WMT2024). The evaluator's human-rating training and dev rows
are rebuilt from the public dataset with `qe/build_human.py` (SSA-MTE train split, NLLB-system rows excluded, no English sentence of the test or dev split).

- `csw_sft/final_v5/`           csw SFT train / val (852,941 training examples)
- `csw_sft/corrections/`        examples rebuilt after the strict-Flash edits (csw_edits*), used for the low-learning-rate fine-tune
- `evaluator/stage_*/`          evaluator training sets by stage (v3, cft1 strict hm/ood, cft2a natural-bad translations, cft2b corruption chains, cft3 regraded chains, cft4 = all data plus human ratings; train/val only)
- `labels/*`                    every judge output: `labels.jsonl` (parsed grades), `pairs.jsonl` / `records.jsonl` (the graded items), raw batch results where kept
- `test/`                       gold and held-out sets, never trained on: evaluator test rows (Flash + luna labels), held-out corruption chains, the 299 gold rows (flash old/strict, luna, argon), csw test splits and the confirmed-root lists
Judges: Gemini 3.8 Flash (Batch API), GPT-5.6 luna, Gemini 4 Argon (299 rows). Labels are 1-5; null = not graded.
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); ap.add_argument("--stage", default="work/hub_stage"); ap.add_argument("--no-push", action="store_true"); a = ap.parse_args()
    st = Path(a.stage); shutil.rmtree(st, ignore_errors=True)
    def put(dst, src):
        src = Path(src)
        if not src.exists(): print("MISSING", src); return
        (st / dst).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, st / dst)
    for d, s in TRAIN.items(): put(d, s)
    for d in LABEL_DIRS:
        for f in KEEP: put(f"labels/{d}/{f}", W / d / f)
        for f in sorted((W / d).glob("results_*.jsonl")): put(f"labels/{d}/{f.name}", f)
        for f in sorted((W / d / "results").glob("*.raw.jsonl")) if (W / d / "results").exists() else []: put(f"labels/{d}/results/{f.name}", f)
    put("labels/label_all.jsonl", W / "label_all.jsonl"); put("test/gold_299/argon_regrade_analysis.txt", W / "regrade_argon/analysis.txt"); put("test/gold_299/flash_strict_analysis.txt", W / "regrade_flash_strict/analysis.txt")
    (st / "README.md").write_text(README); n = sum(1 for _ in st.rglob("*") if _.is_file()); gb = sum(f.stat().st_size for f in st.rglob("*") if f.is_file()) / 1e9; print(f"staged {n} files, {gb:.2f} GB")
    if a.no_push: return
    from huggingface_hub import HfApi
    tok = env_value("HF_WRITE_TOKEN")
    api = HfApi(token=tok); api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(folder_path=str(st), repo_id=a.repo, repo_type="dataset", num_workers=4); print("PUSHED")


if __name__ == "__main__":
    main()
