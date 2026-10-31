"""The trained group-token evaluator as a reward scorer: score_fn(items) -> mean (1-5) of the task's dimensions (equal weights)."""
import json, sys
from pathlib import Path
from csw.evaluator import qe_lib as Q


class EvaluatorScorer:
    def __init__(self, run_dir, device="cuda", grader="strict", bs=96):
        import torch
        from transformers import AutoTokenizer
        d = Path(run_dir); meta = json.load(open(d / "meta.json")); self.tok = AutoTokenizer.from_pretrained(d); self.enc = Q.Enc(self.tok); self.enc.grader = bool(meta.get("grader_tag")); self.enc.default_grader = grader
        self.model = Q.build_model(meta["model"], meta=meta); self.model.load_state_dict(torch.load(d / "best.pt", map_location="cpu")); self.model.to(device).eval(); self.device, self.bs = device, bs
        for p in self.model.parameters(): p.requires_grad_(False)

    def predict_all(self, items):
        """items: [(task, lang, pin, source, final, devices)] -> np.array [n, 27] (1-5)"""
        ids = [self.enc.encode(t, {"lang": lang, "pin": pin or ""}, src, fin, " | ".join(devs) if devs else "") for t, lang, pin, src, fin, devs in items]
        return Q.predict(self.model, self.enc, ids, self.device, bs=self.bs)

    def __call__(self, items):
        P = self.predict_all(items); out = []
        for (t, *_), row in zip(items, P): out.append(float(sum(row[i] for i in Q.FIELD_DIMS[Q.TASKS[t][1]]) / len(Q.FIELD_DIMS[Q.TASKS[t][1]])))
        return out
