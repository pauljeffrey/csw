# csw

*Weights and data are available to reviewers on request.*

I wanted to see whether reinforcement learning could make a small model better at writing the way people in Africa actually write: mixing languages mid-sentence, and using idioms that don't translate word for word. This repository is the record of that attempt, including the parts that didn't work.

The model is a 301M-parameter decoder built on Gemma-3-270M. It covers 37 African languages and six tasks: translation from English, code-switching (from English or from a single-language text), and idiom writing (from English, from a single-language text, or from a code-switched text). I first trained it with supervised fine-tuning, then with GRPO, using a learned evaluator and a set of hand-written rules as the reward.

The short version of what happened: the reward went up and the text did not get better.

- The paper is in [`paper/paper.pdf`](paper/paper.pdf) (source in [`paper/paper.md`](paper/paper.md)).
- The tables, logs and figures behind every number are in [`results/`](results/).
- The code that produced them is in [`src/csw/`](src/csw/). [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) explains how the stages fit together, [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) explains why, and [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) lists what is read from the environment (see `.env.example`).

This repository holds no data or weights, and no instructions for running the model.

## What I found

**A learned reward on its own gets gamed.** In my first runs the reward was mostly the evaluator, plus a few rules. Over 540 steps the reward rose from 0.37 to 0.61, while chrF++ against the reference texts fell from 50.05 to 43.29 on English-to-code-switch and from 55.58 to 48.35 on monolingual-to-code-switch. English words were leaking into the output, and the evaluator didn't mind.

**The evaluator is weaker than its numbers suggested.** It matches the Gemini judges that labelled its training data fairly well (Spearman 0.69 to 0.85 depending on the task). Against 20,204 human-rated translations it reaches 0.25, where SSA-COMET-QE gets 0.49 and AfriCOMET-QE 0.35. When I later checked it where it matters for GRPO, ranking the outputs for one prompt against each other, it agreed with Flash only 54% of the time. Raising its score by 0.2 points should have bought about 0.02 points of judged quality. The judged score fell instead.

**Rules did most of the work.** For the final runs I added checks for language identification, function words, diacritics, names, numbers, length, repetition and device use, a reference-free translation score for English-to-language prompts, a fresh prompt pool that shares nothing with the SFT data, and a held-out chrF++ check applied per task. Splitting up the reward gain, about 74% came from the rules and about 26% from the evaluator.

**The final model is mixed.** On the full test set (6,230 items, 34 languages, greedy decoding), chrF++ improves by 0.8 to 1.3 on the three idiom tasks and drops by 0.7 and 1.0 on the two code-switching tasks. Valid output format goes from about 95% to 99.8%.

**An independent judge is less kind.** Gemini 3.8 Flash graded all 24,898 outputs from the SFT model, two RL checkpoints and the text the training data was generated from, using a rubric that doesn't say which system wrote what. RL comes out 0.09 points below SFT on a 1 to 5 scale (95% interval -0.10 to -0.07). Every task is lower, with monolingual-to-code-switch worst at -0.19, and 18 of 34 languages decline, Nigerian Pidgin most (-0.54). Both models sit about 1.4 points below the generator's own text. A Gemini 4 Argon check on 1,000 items points the same way, though a sample that small can't confirm the size of the drop. I only saw the decline once the full set was graded: two earlier 1,000-item samples showed nothing.

**A few smaller results.**
- Picking the best of eight samples with the evaluator raises the evaluator score but not chrF++. The best possible pick is 3 to 4 points above greedy, so the missing piece is a better selector, not more samples.
- Six extra "group" tokens I added to the evaluator did not specialise. Their attention and hidden states ended up nearly identical.
- In the SFT ablations, starting from a translation-tuned model beat the plain base model, and glossed devices hurt. Adding 6 interleaved layers beat adding 12, but neither beat the 18-layer baseline when trained on 20% of the data. An English draft helped only the English-to-idiom task, within the noise.

## What I could not do

One seed per configuration. The supervision comes from LLM judges, with no human ratings of code-switching quality. A blind candidate pool for a human preference study exists, but I haven't run it. The switch-share rule penalises 29 to 36% of the human references, language ID is unreliable on short mixed spans, and the number-word and diacritic tables are incomplete. The list of allowed switch languages for each base language was supplied by a person, and a native speaker may find some pairs unnatural. I also didn't test a lower evaluator weight, or preference-based training on judge-picked pairs; the numbers suggest both are worth trying.

## The two models

The generator is Gemma-3-270M, translation-tuned, with 6 identity-initialised layers interleaved (24 layers in total) and the task tags added to the vocabulary. The SFT data is 832,421 rows written by a 31B instruction model, filtered by rules and by two LLM judges on about 22% of rows. Validation has 17,399 rows and the curated test set 6,230, with no English source shared with training. For RL I used GRPO with groups of 8 (seven sampled, one greedy), 32 prompts per step, AdamW at 2e-6 and a KL penalty to the SFT model with an adaptive coefficient, on 46,145 fresh prompts. The reward is the evaluator mean minus weighted rule penalties, floored at -1, plus an AfriCOMET-QE term for English-to-language prompts in the 14 languages it covers. I did not use SSA-MTE, AfriMTE, NLLB output or TranslatePsy for training.

The evaluator is mmBERT-base with six group tokens added to its 256,006-token vocabulary and six small heads giving 27 scores from 1 to 5. It was trained once on 167k records from this project's data, labelled mostly by Gemini 3.8 Flash, plus synthetic negatives and graded corruption chains. Best validation Spearman was 0.683. It ranks a better text above a worse one 80% of the time on 534 held-out pairs, and puts corruption chains in the right order 84 to 89% of the time. It ignores the devices block, and scrambling the target lowers its scores by 1.7 to 2.4 points. Because the labels come from commercial LLM judges, it is for research use only.

## Citing

Citation details will be added with the final version of the paper.
