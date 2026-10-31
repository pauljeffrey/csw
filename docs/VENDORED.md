# Vendored code

`src/csw/vendor/gemma_mt/` contains a small subset of a sibling research project (`gemma_mt`, English-to-African-language translation fine-tuning of Gemma 3 270M IT). The GRPO trainer in `csw.rl.train_rl` was written against the interfaces of that project and imports its GRPO building blocks, vLLM rollout sampler and completion log-probability function, so the required files are included here instead of being installed as a dependency.

## Files

| vendored file | purpose in this repository |
|---|---|
| `gemma_mt/rl/grpo.py` | group advantages, KL estimators and the token-level GRPO loss |
| `gemma_mt/rl/sampling.py` | HF `generate` sampling, merging the greedy completion into groups, assembling (prompt, completion) tensors |
| `gemma_mt/rl/vllm_sampler.py` | rollout sampler behind one interface (HF or vLLM, weights pushed after every optimiser step) |
| `gemma_mt/rl/vllm_engine.py` | the vLLM engine, in-process or in a child process |
| `gemma_mt/rl/text.py` | turning sampled token ids into the text a reward may see (stop handling, language-tag stripping) |
| `gemma_mt/losses.py` | `completion_logprobs`, token-level log-probabilities without materialising full-vocabulary logits |
| `gemma_mt/fused_loss.py` | fused LM-head log-probability (optional Triton path used by `losses.py`) |
| `gemma_mt/format.py` | prompt / completion format of the translation model; `rl/text.py` imports `strip_iso_prefix` from it |
| `gemma_mt/languages.py` | language table needed by `format.py` |
| `gemma_mt/modeling.py` | model construction and loading used by the HF-backed engine |
| `gemma_mt/attention.py` | cross-sample attention masking for packed sequences (optional, used by `modeling.py`) |

## What was changed

Only the dotted import paths were rewritten, from `gemma_mt.<module>` to `csw.vendor.gemma_mt.<module>`, including the module path used when the vLLM engine is started as a child process. No other line was changed, and every file keeps its original header.

## Licence

Every vendored file carries the header `SPDX-License-Identifier: AGPL-3.0-or-later`. Those headers are kept untouched, and the vendored files remain under that licence. The copyright notice of the source project says that the licence covers the source code of that module only and does not change the terms of model weights or datasets produced with it. This repository has no top-level licence file yet; the licence of the repository as a whole is to be decided by the owner, and that decision has to take into account that the vendored files are AGPL-3.0-or-later.

## Other code not written for this repository

* `csw.datagen.providers` (OpenRouter, Together and Doubleword clients and a shared base class) is the project's own provider layer, copied from the same research tree as the rest of the code and not a third-party component.
* Third-party libraries (PyTorch, transformers, huggingface_hub, vLLM, sacrebleu, unbabel-comet, fastText with GlotLID, google-genai, openai, scipy, numpy) are runtime dependencies and are not bundled.
