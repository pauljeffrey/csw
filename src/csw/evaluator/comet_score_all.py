"""Score (src, mt, ref) segments with SSA-COMET / AfriCOMET checkpoints.  python -m csw.evaluator.comet_score_all --segments work/ssa_full.jsonl --out work/comet_ssa_full.json"""
import argparse, json
from huggingface_hub import hf_hub_download, snapshot_download
from comet import load_from_checkpoint

MODELS = {"ssa_mtl_ref": ("McGill-NLP/ssa-comet-mtl", True), "ssa_mtl_qe": ("McGill-NLP/ssa-comet-mtl", False), "ssa_qe": ("McGill-NLP/ssa-comet-qe", False),
          "africomet_stl11_ref": ("masakhane/africomet-stl-1.1", True), "africomet_mtl_ref": ("masakhane/africomet-mtl", True), "africomet_qe_stl11": ("masakhane/africomet-qe-stl-1.1", False)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--segments", required=True); ap.add_argument("--out", required=True); ap.add_argument("--bs", type=int, default=64); a = ap.parse_args()
    segs = [json.loads(l) for l in open(a.segments, encoding="utf-8")]; res = {"ids": [s["id"] for s in segs]}
    loaded = {}; prev = json.load(open(a.out)) if __import__("os").path.exists(a.out) else {}
    for k, (repo, ref) in MODELS.items():
        if prev.get(k): res[k] = prev[k]; continue
        try:
            if repo not in loaded: loaded[repo] = load_from_checkpoint(snapshot_download(repo) + "/checkpoints/model.ckpt")
            data = [{"src": s["eng"], "mt": s["mt"], **({"ref": s["ref"]} if ref else {})} for s in segs]
            res[k] = [float(x) for x in loaded[repo].predict(data, batch_size=a.bs, gpus=1, progress_bar=False).scores]; print(k, "ok", flush=True)
        except Exception as e:
            res[k] = None; print(k, "FAILED", repr(e)[:300], flush=True)
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
