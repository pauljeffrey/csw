"""AfriCOMET-QE scoring worker (runs in the COMET environment: python -m csw.rl.qe_worker).  stdin: one JSON per line {"id": n, "pairs": [[src, mt], ...]}; stdout: {"id": n, "scores": [...]}.
Model: masakhane/africomet-qe-stl-1.1 (reference-free).  Used only for the eng_m task, in the AfriMTE / AfriCOMET languages."""
import json, sys

def main():
    from huggingface_hub import snapshot_download
    from comet import load_from_checkpoint
    m = load_from_checkpoint(snapshot_download("masakhane/africomet-qe-stl-1.1") + "/checkpoints/model.ckpt"); print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        r = json.loads(line); data = [{"src": a, "mt": b if b.strip() else "."} for a, b in r["pairs"]]
        sc = [float(x) for x in m.predict(data, batch_size=64, gpus=1, progress_bar=False).scores]
        print(json.dumps({"id": r["id"], "scores": sc}), flush=True)

if __name__ == "__main__":
    main()
