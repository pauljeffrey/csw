"""Diverse culture seed cards from the GPU Gemma-4-31B: for each language, the countries / states / communities / dialect areas where it is spoken, and for each place a set of distinct, factual
cultural aspects (food, rites, music, crafts, kinship, law, markets, beliefs, ...). One card = one document topic for the small CPU writer.
python -m csw.datagen.culture_seeds --langs swh yor ... [--places 8] [--conc 12]     -> culture_seeds/<lang>.jsonl   {id, lang, country, region, community, aspect, title, description, terms[]}"""
from csw.paths import DATA_ROOT
import json, os, sys, re, hashlib, random, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = DATA_ROOT
os.environ.setdefault("CS_OUT", "out_culture")
from csw.datagen import cs_pipeline as pl
from csw.datagen import cs_prompts as pr
from csw.datagen.providers import Request
from csw.datagen.profiles import P
ASPECTS = ["food and cooking", "farming, seasons and land", "fishing, herding and trades", "markets, money and trade networks", "kinship, family roles and respect", "marriage and weddings", "birth, naming and childhood",
           "initiation and coming of age", "funerals, mourning and remembrance", "chieftaincy, councils and customary law", "belief and religion (traditional, Christian, Muslim, and how they mix)", "festivals and the calendar",
           "music and instruments", "dance and masquerade", "oral literature: proverbs, riddles, folktales", "clothing, cloth and adornment", "architecture, houses and compounds", "crafts: pottery, weaving, carving, metalwork, beadwork",
           "medicine, healing and herbs", "games, sport and children's play", "greetings, etiquette and hospitality", "humour, storytelling and public speaking", "history, migration and origin stories", "city life: transport, youth culture, work",
           "education and apprenticeship", "landscape, landmarks and the environment"]
GPU = pl.prov("google/gemma-4-31b-it", int(sys.argv[sys.argv.index("--conc") + 1]) if "--conc" in sys.argv else 12) if False else None
def ask(pv, sysm, user, mt):
    r = pv.complete(Request("cs", [{"role": "system", "content": sysm}, {"role": "user", "content": user}], max_tokens=mt, temperature=0.5, top_p=0.95, response_format={"type": "json_object"}))
    return pr.parse_json(r.text) if r.ok else None
def main():
    a = sys.argv; opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    langs = []; i = a.index("--langs") + 1
    while i < len(a) and not a[i].startswith("--"): langs.append(a[i]); i += 1
    nplaces, conc, naspects = int(opt("--places", 8)), int(opt("--conc", 12)), int(opt("--aspects", 12)); pl.PROVIDER = "local"; pv = pl.prov(opt("--model", "google/gemma-4-31b-it"), conc)      # --model + CS_LOCAL_URL select the server (the CPU llama-server on :8001 keeps the GPU untouched)
    out = HERE / "culture_seeds"; out.mkdir(exist_ok=True); lk = threading.Lock()
    SYS = "You are an anthropologist and cultural journalist with deep, careful knowledge of African peoples. You only state things you are confident are real and typical; you never invent festivals, names or customs; when a detail is uncertain you stay general. You avoid stereotypes and exoticism. Output JSON only."
    def places(lang):
        p = P[lang]
        d = ask(pv, SYS, f"The language {p.name} is spoken in these known areas: {', '.join(p.persona_regions)}; dialect/variety labels used: {', '.join(p.dialects)}.\nList {nplaces} DISTINCT places where {p.name} is a living language of everyday life, spread across different countries, states/provinces and communities, and where dialect or community culture really differs (include rural as well as urban, majority and minority communities). "
                       f'Return {{"places": [{{"country": "...", "region": "state/province/area", "community": "people/community/dialect group", "note": "one factual line on what makes it distinct"}}]}}', 1800)
        return (d or {}).get("places") or []
    def cards(lang, pl_, k):
        p = P[lang]; rnd = random.Random(f"{lang}-{k}"); asp = rnd.sample(ASPECTS, naspects)
        d = ask(pv, SYS, f"Place: {pl_.get('community')} community, {pl_.get('region')}, {pl_.get('country')} (language: {p.name}). Context: {pl_.get('note')}.\nFor EACH of these aspects give ONE specific, factual, typical item of this community's culture (a named dish, a ceremony and what happens, a craft and its materials, a proverb theme, a musical instrument, a kinship rule...), concrete enough to base an article on: {'; '.join(asp)}.\n"
                       f'Return {{"items": [{{"aspect": "...", "title": "short English title", "description": "2-3 factual English sentences with concrete details", "terms": ["local words in {p.name} or the community language, only if you are sure of them"]}}]}}', 3000)
        return (d or {}).get("items") or []
    def one_lang(lang):
        f = out / f"{lang}.jsonl"; seen = set(); n0 = 0
        if f.exists():
            n0 = sum(1 for _ in f.open(encoding="utf-8"))
            if n0 >= nplaces * 10: return
        ps = places(lang); cnt = 0
        for k, pl_ in enumerate(ps[:nplaces]):
            for it in cards(lang, pl_, k):
                key = re.sub(r"\W+", " ", (it.get("title", "") + it.get("aspect", "")).lower()).strip()
                if not it.get("description") or key in seen: continue
                seen.add(key)
                rec = {"id": hashlib.sha1((lang + key + str(pl_.get("community"))).encode()).hexdigest()[:12], "lang": lang, "country": pl_.get("country"), "region": pl_.get("region"), "community": pl_.get("community"),
                       "aspect": it.get("aspect"), "title": it.get("title"), "description": it.get("description"), "terms": it.get("terms") or []}
                with lk, f.open("a", encoding="utf-8") as fo: fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cnt += 1
        print(lang, "places", len(ps), "cards", cnt, flush=True)
    with ThreadPoolExecutor(3) as ex: list(ex.map(one_lang, langs))
if __name__ == "__main__": main()
