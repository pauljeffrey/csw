# 8-bit GGUF check

779 test prompts (every 8th item of the 6,230-item test set, all five tasks, 34 languages), greedy decoding, chrF++ of the final text against the reference, macro-averaged over languages, then over tasks. Each cell is chrF++ / share of outputs with valid format. "same text" is the share of outputs identical to the vLLM fp32 output, and the last column is the chrF++ of the output against that fp32 output.

```

csw_sft: chrF++ (macro over languages, final text) / format_ok;  agreement with vLLM fp32 output
config                       eng_hm      eng_ood       hm_ood         m_hm        m_ood   mean   same-text  chrF(vs fp32 out)
csw_sft_vllm_fp32        52.52/0.97   37.33/1.00   38.98/1.00   56.77/0.96   37.81/1.00   44.68   100.0%    100.0
csw_sft_vllm_bf16        52.81/0.97   35.61/1.00   37.50/1.00   56.13/0.96   37.03/1.00   43.82     3.6%     66.4
csw_sft_gguf_f32         52.13/0.97   37.36/1.00   38.63/1.00   56.85/0.96   37.74/1.00   44.54    69.4%     92.9
csw_sft_gguf_f16         52.14/0.97   36.86/1.00   37.45/1.00   56.14/0.96   38.05/1.00   44.13    18.5%     77.8
csw_sft_gguf_q8_0        52.80/0.97   36.39/1.00   37.58/1.00   56.69/0.94   38.22/1.00   44.34     6.3%     70.0

csw_rl: chrF++ (macro over languages, final text) / format_ok;  agreement with vLLM fp32 output
config                       eng_hm      eng_ood       hm_ood         m_hm        m_ood   mean   same-text  chrF(vs fp32 out)
csw_rl_vllm_fp32         52.04/1.00   36.38/1.00   37.80/1.00   56.72/1.00   39.11/1.00   44.41   100.0%    100.0
csw_rl_vllm_bf16         50.99/1.00   36.59/1.00   37.75/1.00   56.06/1.00   39.04/1.00   44.09     6.5%     73.8
csw_rl_gguf_f16          51.66/1.00   36.42/1.00   37.82/1.00   56.34/1.00   39.28/1.00   44.30    24.3%     82.2
csw_rl_gguf_q8_0         51.67/1.00   36.91/1.00   38.66/1.00   56.11/1.00   39.01/1.00   44.47     8.1%     75.1
```

Reading: the Q8 file scores within about 0.3 chrF++ of fp32 on both models (SFT -0.34, RL +0.06), which is inside the spread between fp32, bf16 and f16 (up to 0.9 on SFT). Format validity is unchanged. Greedy outputs are rarely identical across numeric formats (bf16 itself matches fp32 text in only 4 to 7% of items), so the comparison is on scores, not on text. The RL f32 GGUF was not run.

Conversion note: llama.cpp's Gemma-3 converter assigns full-attention layers by a fixed every-sixth-layer rule and ignores the model's `layer_types`. This model has 6 extra layers, so its full-attention layers sit at 6, 14 and 22. With the stock converter the GGUF scored about 20 chrF++ against 44; writing the per-layer pattern into the file (a three-line change in the converter) restores parity. The unused `<image_soft_token>` entry (id 262144, beyond the embedding table) also has to be removed from the tokenizer before conversion.
