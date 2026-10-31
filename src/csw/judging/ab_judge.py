"""gpt-4o judge on the A/B test rows (A = production prompt, B = rules first + 5 shots), same cards.  python -m csw.judging.ab_judge build|submit|poll|report"""
from csw.paths import DATA_ROOT
import json, sys, collections, statistics as st
from pathlib import Path
HERE = DATA_ROOT
from csw.judging import oai_judge as oj
OUT = HERE / "oai_ab"; OUT.mkdir(exist_ok=True); oj.OUT = OUT
def build():
    reqs = []; meta = {}
    for tag in ("A", "B"):
        for l in (HERE / "out_ab" / f"{tag}.jsonl").read_text(encoding="utf-8").split("\n"):
            if not l.strip(): continue
            o = json.loads(l)
            if not o.get("ok"): continue
            r = dict(o["row"]); cid = f"ab{tag}-{o['id']}"
            reqs.append({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": oj.SYSTEM}, {"role": "user", "content": oj.card(r)}], "temperature": 0, "max_tokens": 500, "response_format": {"type": "json_object"}}})
            meta[cid] = {"tag": tag, "id": o["id"], "lang": o["lang"], "rule": o["score"]}
    (OUT / "requests.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in reqs) + "\n", encoding="utf-8"); (OUT / "meta.json").write_text(json.dumps(meta))
    print(len(reqs), "requests")
def report():
    meta = json.loads((OUT / "meta.json").read_text()); L = {}
    for l in (OUT / "labels.jsonl").read_text(encoding="utf-8").split("\n"):
        if l.strip(): o = json.loads(l); L[o["id"]] = o["label"]
    by = collections.defaultdict(list); pair = collections.defaultdict(dict)
    for cid, m in meta.items():
        lab = L.get(cid)
        if lab and isinstance(lab.get("scores"), dict):
            by[m["tag"]].append(lab); pair[m["id"]][m["tag"]] = lab
    f = lambda v: round(st.mean(v), 2) if v else "-"
    print(f"{'':3}{'n':>4} {'m_text':>7} {'hm':>6} {'ood':>6} {'eng':>6} | {'keep':>5} {'fix':>5} {'drop':>5} | can_read | scores<=6 (any field)")
    for t in ("A", "B"):
        v = by[t]; sc = lambda k: [x["scores"].get(k) for x in v if isinstance(x["scores"].get(k), (int, float))]
        vc = collections.Counter(x.get("verdict") for x in v)
        low = sum(min(x["scores"].get(k, 10) for k in ("m_text", "hm", "ood")) <= 6 for x in v)
        print(f"{t:3}{len(v):>4} {f(sc('m_text')):>7} {f(sc('hm')):>6} {f(sc('ood')):>6} {f(sc('eng')):>6} | {vc['keep']:>5} {vc['fix']:>5} {vc['drop']:>5} | {sum(bool(x.get('can_read')) for x in v):>3}/{len(v):<4} | {low}")
    both = [(p["A"], p["B"]) for p in pair.values() if "A" in p and "B" in p]
    mean3 = lambda x: st.mean(x["scores"].get(k, 0) for k in ("m_text", "hm", "ood"))
    wins = sum(mean3(b) > mean3(a) + 0.01 for a, b in both); loss = sum(mean3(b) < mean3(a) - 0.01 for a, b in both)
    print(f"paired on {len(both)} identical cards: B better {wins}, A better {loss}, tie {len(both)-wins-loss}; mean diff (B-A) = {st.mean(mean3(b)-mean3(a) for a,b in both):+.2f}")
if __name__ == "__main__":
    c = sys.argv[1]
    {"build": build, "submit": oj.submit, "poll": oj.poll, "report": report}[c]()
