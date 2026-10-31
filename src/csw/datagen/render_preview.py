"""Regenerate prompt_preview.md from the LIVE prompt builders + exemplars/gold.jsonl + prompt_changelog.md.
Called automatically at the start of every generate / genall / rehm / mcheck / judge run, so the file never goes stale."""
from csw.paths import DATA_ROOT
import hashlib, sys, time
from pathlib import Path
HERE = DATA_ROOT


def render(path=None):
    from csw.datagen import cs_pipeline as pl
    from csw.datagen import cs_prompts as pr
    gold = pr.load_gold()
    plan = pl.make_plan("yor", 80)
    B = next(c for c in plan if c["track"] == "B" and c["length"] == "medium")
    A = next(c for c in plan if c["track"] == "A" and c["length"] == "short")
    C = next(c for c in plan if c["track"] == "C")
    mB, mA, mC = (pr.gen_prompt(pl.P["yor"], x, gold) for x in (B, A, C))
    system = mB[0]["content"]
    sysA, sysC = mA[0]["content"], mC[0]["content"]   # language-specific system messages (up to 3 rotating example sets per language)
    row = {"eng": "<plain English of the sample>", "m_text": "<official translation>"}
    card = mB[1]["content"].split("\nHOW TO START")[0]
    rehm_user = card + ("\n\nThe plain English and the official translation of this sample are already written:\n"
                        f"ENG: {row['eng']}\nM_TEXT: {row['m_text']}\n\n" 'Write ONLY the hm field (item 3) for this sample: a code-switched variation of M_TEXT. Return exactly {"hm": "..."}')
    sample_row = {"scenario": B and {"dialect": B["dialect"], "switch": B["switch"]}, "eng": "<eng>", "m_text": "<m_text>", "hm": "<hm>", "ood": "<ood>"}
    judge = pr.judge_prompt(pl.P["yor"], sample_row)
    h = hashlib.md5((system + mB[1]["content"] + pl.BT_SYS + pl.MT_SYS + rehm_user).encode()).hexdigest()[:8]
    log = (HERE / "prompt_changelog.md").read_text(encoding="utf-8") if (HERE / "prompt_changelog.md").exists() else ""
    # register this fingerprint in prompt_versions.json (v3.1, v3.2, ... in order of first appearance) so rows can be tagged by version
    import json
    vp = HERE / "prompt_versions.json"
    vers = json.loads(vp.read_text()) if vp.exists() else {}
    if h not in vers:
        vers[h] = {"version": f"v3.{len(vers) + 1}", "first_rendered": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
        vp.write_text(json.dumps(vers, indent=1))
    version = vers[h]["version"]
    tok = lambda s: len(s) // 3
    out = f"""# Code-switching prompts - live preview

**Prompt version {version} - fingerprint `{h}`** - regenerated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} from the code that actually runs (it changes whenever any prompt below changes). {len(gold)} gold rows injected.

## Change log (newest last)

{log}

---

## 1. Generation prompt (every row)

The **system message is language-specific**: the rules are shared, and the GOLD block holds that language's own native-corrected examples (3 high / 4 mid / 5 low; high-resource languages short of rows borrow one row each from other high-resource languages; up to 3 rotating example sets, each a cacheable prefix). The exact rows per language are in `fewshot_map.json`. The **user message** (task card) changes per row. System ~ {len(system):,} chars (~{tok(system):,} tokens).

### 1a. SYSTEM MESSAGE (Yoruba shown)

```text
{system}
```

### 1b. USER MESSAGE - Yoruba, native-seeded row (track B, medium)

```text
{mB[1]['content']}
```

### 1c. USER MESSAGE - Yoruba, anchor-seeded row (track A, short)

```text
{mA[1]['content']}
```

### 1d. USER MESSAGE - Yoruba, English-story-seed row (track C)

```text
{mC[1]['content']}
```

**Where the input text goes:** in the user message after `HOW TO START:` - track B `SEED: \"\"\"<native passage>\"\"\"` (inspiration only), track C `ENGLISH SEED: \"\"\"<English text>\"\"\"` (localised), track A none (anchors only). The system message never contains row-specific text.

---

## 2. hm-only regeneration (`rehm`)

Same system message as 1a. User message:

```text
{rehm_user}
```

---

## 3. m_text accuracy check (`mcheck`)

Back-translation (the model never sees `eng`; chrF of its English vs `eng` is the score):

```text
SYSTEM: {pl.BT_SYS}
USER:   <m_text>
```

Re-translation of rows scoring below the threshold (4 candidates, best by back-translation score wins; old m_text kept as `m_text_rejected`):

```text
SYSTEM: {pl.MT_SYS.format(name='<language>', dialect='<dialect>', orth='<orthography>')}
USER:   <eng>
```

---

## 4. Judge

```text
{judge[0]['content']}

{judge[1]['content']}
```
"""
    p = Path(path) if path else HERE / "prompt_preview.md"
    p.write_text(out, encoding="utf-8")
    return h, len(out)


if __name__ == "__main__":
    print(render())
