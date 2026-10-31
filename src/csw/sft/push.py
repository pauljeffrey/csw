"""Push a trained csw checkpoint to the Hub: private repo first; a model card only if it stays private.
python -m csw.sft.push --src work/runs/sft1/best   (repository from CSW_MODEL_REPO)"""
import argparse, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--repo", default=os.environ.get("CSW_MODEL_REPO")); ap.add_argument("--card", default=""); a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_WRITE_TOKEN"]); private = True
    try:
        api.create_repo(a.repo, private=True, exist_ok=True)
    except Exception as e:
        print("private repo refused:", str(e)[:200], "-> public, no card"); private = False; api.create_repo(a.repo, private=False, exist_ok=True)
    if private and a.card and Path(a.card).exists(): api.upload_file(path_or_fileobj=a.card, path_in_repo="README.md", repo_id=a.repo)
    api.upload_folder(folder_path=a.src, repo_id=a.repo, ignore_patterns=["*.pt", "state.pt", "last/*"])
    print("pushed", a.repo, "private" if private else "PUBLIC")


if __name__ == "__main__":
    main()
