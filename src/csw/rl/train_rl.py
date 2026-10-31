"""GRPO for the csw code-switcher.  Policy = full_E (Gemma-3 based, 24 layers), reference = the start model (frozen), reward = csw.rl.reward (group-token evaluator + capped rules, equal-weight dimensions).
Group = (G-1) sampled completions + the prompt's GREEDY completion appended last (greedy joins the baseline only through the sampled statistics, advantage scale 0.3, cap 3 -- D-67 of the translation RL).
KL guard: if the mean per-token KL(pi||ref) of a step exceeds cfg.kl_guard (0.15) the optimizer step is skipped.  A fixed monitor set is decoded greedily every --monitor-every steps (evaluator score vs KL = the overoptimisation curve).
python -m csw.rl.train_rl --model work/runs/full_E/best --evaluator work/runs/ev_final --prompts work/rl_prompts --out work/rl_run1 --steps 400"""
import argparse, collections, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from csw.rl.config import RewardConfig
from csw.rl.evaluator import EvaluatorScorer
from csw.rl.reward import Reward
from csw.rl.lid import Lid
from csw.rl.qe_client import QEClient
from csw.rl import rules as R
from csw.rl.rules import FuncWords
from csw.vendor.gemma_mt.rl import grpo
from csw.vendor.gemma_mt.rl.sampling import merge_greedy_into_groups, assemble_batch
from csw.vendor.gemma_mt.rl.vllm_sampler import VLLMSampler
from csw.vendor.gemma_mt.rl.vllm_engine import EngineConfig, VLLMEngine
from csw.vendor.gemma_mt.losses import completion_logprobs


def chat(tok, prompt): return tok(f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n", add_special_tokens=False)["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    for k, v in dict(model=None, evaluator=None, prompts=None, out=None).items(): ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--prompts-per-step", type=int, default=32); ap.add_argument("--group", type=int, default=8); ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-6); ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--micro", type=int, default=32); ap.add_argument("--monitor-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=50); ap.add_argument("--gpu-util", type=float, default=0.30); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--resume", default=""); ap.add_argument("--kl-target", type=float, default=0.12); ap.add_argument("--kl-guard", type=float, default=0.35); ap.add_argument("--beta0", type=float, default=0.02); ap.add_argument("--monitor-n", type=int, default=720); ap.add_argument("--no-lid", action="store_true"); ap.add_argument("--task-weights", default="eng_hm=0.35,eng_ood=0.15,m_hm=0.15,m_ood=0.10,hm_ood=0.10,eng_m=0.15"); ap.add_argument("--no-qe", action="store_true"); ap.add_argument("--gen-log-every", type=int, default=2); ap.add_argument("--chrf-slack", type=float, default=1.0); ap.add_argument("--chrf-base", type=float, default=47.49); ap.add_argument("--chrf-task-slack", type=float, default=2.0)
    a = ap.parse_args(); cfg = RewardConfig(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rnd = random.Random(a.seed); torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model); pad = tok.pad_token_id; stop = [tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id]; dev = "cuda"
    train = [json.loads(l) for l in open(Path(a.prompts) / "train.jsonl", encoding="utf-8")]; monitor = [json.loads(l) for l in open(Path(a.prompts) / "monitor.jsonl", encoding="utf-8")][: a.monitor_n]
    refmon_p = Path(a.prompts) / "refmon.jsonl"; refmon = [json.loads(l) for l in open(refmon_p, encoding="utf-8")] if refmon_p.exists() else []; base_chrf = [a.chrf_base]; best_chrf = [-1.0]; TASK_BASE = {"m_ood": 47.61, "hm_ood": 42.36, "eng_ood": 35.35, "m_hm": 59.27, "eng_hm": 52.85}       # SFT model (full_E) on the 300 held-out prompts
    policy = AutoModelForCausalLM.from_pretrained(a.resume or a.model, torch_dtype=torch.float32).to(dev); policy.config.use_cache = False; policy.gradient_checkpointing_enable()
    ref = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval(); [p.requires_grad_(False) for p in ref.parameters()]
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scorer = EvaluatorScorer(a.evaluator, device=dev); here = Path(__file__).parent / "data"; fw = FuncWords(here / "funcwords.json"); nwf = here / "number_words.json"; nw = {k: set(v) for k, v in json.load(open(nwf, encoding="utf-8")).items()} if nwf.exists() else {}
    reward = Reward(scorer, cfg, funcwords=fw, number_words=nw, lid=(None if a.no_lid else Lid()), group=a.group, qe_fn=(None if a.no_qe else QEClient()), base_floors=json.load(open(here / "base_fw_floor.json")), lang_windows=(json.load(open(here / "length_windows.json")) if (here / "length_windows.json").exists() else {}), allowed_switch=json.load(open(here / "allowed_switch.json")))
    fw.t.setdefault("eng", R.EN_FUNC); fw.t.setdefault("fra", R.FR_FUNC)
    sampler = VLLMSampler(VLLMEngine(EngineConfig(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=2048, max_num_seqs=256, seed=a.seed)), seed=a.seed)
    G = a.group; beta = a.beta0; log = open(out / "train_log.jsonl", "a"); gen_log = open(out / "gens_log.jsonl", "a"); mon_gen = open(out / "monitor_gens.jsonl", "a"); mon_log = open(out / "monitor_log.jsonl", "a"); best = -1e9; t0 = time.time()

    def save(path):
        sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in policy.state_dict().items()}; policy.save_pretrained(path, state_dict=sd); tok.save_pretrained(path)       # the fp32 master weights stay untouched

    def score_rollout(items, ro, greedy_all=False):
        texts = [tok.decode(c, skip_special_tokens=False) for c in ro.completions]; rep = [dict(it, switch_lang=it.get("switch") or "eng", pin=it.get("switch") or "") for it in items for _ in range(len(ro.completions) // len(items))]
        reward.group = 1 if greedy_all else G; rew, det = reward(rep, [t if f else t.replace("<end_of_turn>", "") for t, f in zip(texts, ro.finished)]); return rew, det, texts

    def run_refmon(step):
        """Reference chrF++ of the greedy FINAL text on held-out SFT-validation prompts (not in the RL prompt pool): the independent check the reward cannot be hacked on."""
        if not refmon: return None
        import sacrebleu
        ro = sampler.generate([chat(tok, it["prompt"]) for it in refmon], group_size=1, max_new_tokens=a.max_new, temperature=0.0, top_p=1.0, top_k=0, stop_ids=stop, pad_id=pad, greedy=True, key=-step - 7); by = collections.defaultdict(lambda: ([], []))
        for it, c, f in zip(refmon, ro.completions, ro.finished):
            t = tok.decode(c, skip_special_tokens=False); t = t if f else t.replace("<end_of_turn>", ""); by[it["task"]][0].append(R.parse(t)["final"]); by[it["task"]][1].append(R.parse(it["completion"] + " </s>")["final"])
        per = {k: sacrebleu.corpus_chrf(h, [r], word_order=2).score for k, (h, r) in by.items()}; return {"chrf": float(np.mean(list(per.values()))), "by_task": per}

    def run_monitor(step):
        nonlocal best
        policy.eval(); sampler.sync(policy, step * 1000 + 1); prompts = [chat(tok, it["prompt"]) for it in monitor]
        ro = sampler.generate(prompts, group_size=1, max_new_tokens=a.max_new, temperature=0.0, top_p=1.0, top_k=0, stop_ids=stop, pad_id=pad, greedy=True, key=-step)
        rew, det, mtexts = score_rollout(monitor, ro, greedy_all=True)
        for j in range(0, len(monitor), max(1, len(monitor) // 48)): mon_gen.write(json.dumps({"step": step, "task": monitor[j]["task"], "lang": monitor[j]["lang"], "source": monitor[j]["source"][:600], "completion": mtexts[j].replace("<pad>", "").strip(), "reward": round(float(rew[j]), 3), "evaluator": det[j].get("evaluator"), "pens": {k: v for k, v in (det[j].get("pens") or {}).items() if v}}, ensure_ascii=False) + "\n")
        mon_gen.flush(); ev = [d["evaluator"] for d in det if "evaluator" in d]; by = collections.defaultdict(list)
        for it, r in zip(monitor, rew): by[it["task"]].append(r)
        m = {"step": step, "monitor_reward": float(np.mean(rew)), "monitor_evaluator": float(np.mean(ev)) if ev else None, "gate_rate": float(np.mean([bool(d.get("gate")) for d in det])), "by_task": {k: float(np.mean(v)) for k, v in by.items()}}
        rm = run_refmon(step)
        if rm:
            m["ref_chrf"] = rm["chrf"]; m["ref_chrf_by_task"] = rm["by_task"]
            if base_chrf[0] is None: base_chrf[0] = rm["chrf"]
        fails = [t for t, b in TASK_BASE.items() if rm and rm["by_task"].get(t, 1e9) < b - a.chrf_task_slack]                      # per-task guard: EVERY task must stay within chrf_task_slack of its own SFT baseline
        ok = (not rm) or (rm["chrf"] >= base_chrf[0] - a.chrf_slack and not fails)
        if rm: m["guard_ok"] = bool(ok); m["guard_fail"] = fails; print("GUARD", "ok" if ok else f"FAIL {fails}", flush=True)
        mon_log.write(json.dumps(m) + "\n"); mon_log.flush(); print("MONITOR", json.dumps(m), flush=True); policy.train()
        if rm and rm["chrf"] > best_chrf[0]: best_chrf[0] = rm["chrf"]; save(out / "best_chrf")
        if m["monitor_reward"] > best and ok: best = m["monitor_reward"]; save(out / "best")
        return m

    tw = {k: float(v) for k, v in (x.split("=") for x in a.task_weights.split(","))} if a.task_weights else {}
    queues = {}

    def next_batch():
        """prompts are drawn per task by weight (eng_hm is up-weighted); inside a task no prompt repeats until its pool is used up"""
        if not tw: return rnd.sample(train, a.prompts_per_step)
        b = []
        for t in rnd.choices(list(tw), weights=list(tw.values()), k=a.prompts_per_step):
            if not queues.get(t): queues[t] = rnd.sample([x for x in train if x["task"] == t], sum(x["task"] == t for x in train))
            b.append(queues[t].pop())
        return b

    run_monitor(0)
    for step in range(1, a.steps + 1):
        ts = time.time(); batch = next_batch(); prompts = [chat(tok, it["prompt"]) for it in batch]
        sampler.sync(policy, step); policy.eval()
        smp = sampler.generate(prompts, group_size=G - 1, max_new_tokens=a.max_new, temperature=a.temperature, top_p=1.0, top_k=0, stop_ids=stop, pad_id=pad, key=step)
        grd = sampler.generate(prompts, group_size=1, max_new_tokens=a.max_new, temperature=0.0, top_p=1.0, top_k=0, stop_ids=stop, pad_id=pad, greedy=True, key=step)
        ro = merge_greedy_into_groups(smp, grd, G - 1); rew, det, texts = score_rollout(batch, ro); t_gen = time.time() - ts
        if step % a.gen_log_every == 0:                                                       # readable samples: 6 prompts, the greedy decode and one sampled decode of each, with reward and rule penalties
            for g in range(6):
                for kind, j in (("greedy", g * G + G - 1), ("sample", g * G)):
                    it = batch[g]; gen_log.write(json.dumps({"step": step, "kind": kind, "task": it["task"], "lang": it["lang"], "switch": it.get("switch"), "source": it["source"][:600], "completion": texts[j].replace("<pad>", "").strip(), "reward": round(float(rew[j]), 3), "evaluator": det[j].get("evaluator"), "pens": {k: v for k, v in (det[j].get("pens") or {}).items() if v}, "gate": det[j].get("gate", False)}, ensure_ascii=False) + "\n")
            gen_log.flush()
        r = torch.tensor(rew, dtype=torch.float32); adv, flat = grpo.group_advantages_sampled_baseline(r, G, greedy_cap=cfg.greedy_adv_cap); adv = grpo.scale_last_member(adv, G, cfg.greedy_adv_scale).to(dev)
        policy.train(); opt.zero_grad(set_to_none=True); N = len(ro.completions); total_tokens = float(sum(len(c) for c in ro.completions)); acc = collections.Counter()
        for s in range(0, N, a.micro):
            idx = list(range(s, min(N, s + a.micro))); b = assemble_batch([ro.prompts[i] for i in idx], [ro.completions[i] for i in idx], pad, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = completion_logprobs(policy, b["input_ids"], b["attention_mask"], b["prompt_width"], b["completion_mask"], temperature=a.temperature)
                with torch.no_grad(): rlp = completion_logprobs(ref, b["input_ids"], b["attention_mask"], b["prompt_width"], b["completion_mask"], temperature=a.temperature)
            o = grpo.grpo_token_loss(lp, adv[idx], b["completion_mask"], ref_logp=rlp, kl_coef=beta, kl_estimator="k3c", loss_agg="token", normaliser=total_tokens); o["loss"].backward()
            for k in ("kl_sum", "pg_sum", "n_tokens"): acc[k] += float(o[k])
        kl = acc["kl_sum"] / max(acc["n_tokens"], 1.0); gn = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)); cut = kl > a.kl_guard
        if cut or not math.isfinite(gn): opt.zero_grad(set_to_none=True)
        else: opt.step()
        used_beta = beta; ratio = kl / a.kl_target                                        # adaptive KL coefficient: pulled up when the policy drifts past the target, relaxed when it is well inside
        if ratio > 1.5: beta = min(2.0, beta * 1.5)
        elif ratio < 0.67: beta = max(0.005, beta / 1.5)
        ev = [d["evaluator"] for d in det if "evaluator" in d]; pens = collections.defaultdict(list)
        for d in det:
            for k, v in (d.get("pens") or {}).items(): pens[k].append(v)
        rec = {"step": step, "reward": float(np.mean(rew)), "evaluator": float(np.mean(ev)) if ev else None, "gate_rate": float(np.mean([bool(d.get("gate")) for d in det])), "kl": kl, "kl_cut": bool(cut), "beta": used_beta, "grad_norm": gn, "flat_groups": flat,
               "mean_tokens": total_tokens / N, "finished": float(np.mean(ro.finished)), "pens": {k: float(np.mean(v)) for k, v in pens.items()}, "greedy_reward": float(np.mean(rew[G - 1::G])), "sec": time.time() - ts, "sec_gen": t_gen}
        log.write(json.dumps(rec) + "\n"); log.flush(); print(f"step {step} reward {rec['reward']:.3f} ev {rec['evaluator']} gate {rec['gate_rate']:.3f} kl {kl:.4f} beta {used_beta:.3f}{' CUT' if cut else ''} gn {gn:.2f} tok {rec['mean_tokens']:.0f} greedy {rec['greedy_reward']:.3f} {rec['sec']:.0f}s", flush=True)
        if step % a.monitor_every == 0: run_monitor(step)
        if step % a.save_every == 0:
            save(out / "last")
    run_monitor(a.steps); print("DONE", flush=True)


if __name__ == "__main__":
    main()
