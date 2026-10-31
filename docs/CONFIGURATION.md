# Configuration

The source tree contains no keys, tokens, hosts or repository identifiers that belong to the project's own data and models.
All of them are read from the process environment (optionally from a `.env` file, see `src/csw/paths.py`). `.env.example` lists every variable.

| group | variables | meaning |
|---|---|---|
| provider keys | `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`, `HF_WRITE_TOKEN` | credentials for the generation, judging and Hub steps |
| locations | `CSW_DATA_ROOT`, `CSW_ENV_FILE`, `CSW_PAPER_DIR`, `CSW_WORK`, `CSW_PYTHON`, `CSW_COMET_PYTHON`, `CSW_QE_LOG`, `CSW_RETOK_DIR`, `CSW_TOKENIZER_*`, `CSW_HF_CACHE` | where data, work directories, interpreters and caches live |
| Hub identifiers | `CSW_MODEL_REPO`, `CSW_DATASET_REPO`, `CSW_DATASET_NAME`, `CSW_BASE_MODEL`, `CSW_REPLAY_*_REPO` | the model, dataset and replay-source repositories |
| English sources | `CSW_ENGLISH_SOURCE_ORG`, `CSW_ENGLISH_SOURCE_SUFFIX`, `CSW_ENGLISH_SOURCES_FILE` | the annotated English source datasets (a JSON list of `[repo, kind, register, max shards, max windows]`) |
| generator models | `CSW_GEN_MODEL`, `CSW_ADJ_MODEL` | models used by the data-generation pipeline |

Stages that need an identifier that is not set stop with a message naming the variable.
