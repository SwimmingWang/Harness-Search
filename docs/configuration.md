# Configuration

Shell entry points load `runtime_paths.env`, then `.env.local`; variables already exported by the caller take precedence. Set `RUNTIME_PATHS` or `CONFIG_ENV` to choose different files (use `/dev/null` for a clean invocation). Configuration files use Bash assignment syntax. `Config` also reads root `.env`/`.env.local` for direct Python calls; shell-derived retrieval/model settings are most conveniently supplied through the launchers.

## Paths and retrieval

| Variable | Default / purpose |
|---|---|
| `PYTHON_BIN` | `.venv/bin/python` if present, otherwise `python3` |
| `VLLM_BIN` | `.venv-serve/bin/vllm` |
| `DATA_ROOT` | `<checkout>/data` |
| `HARNESS_SEARCH_QUERY_DATA_ROOT` | `$DATA_ROOT/queries` |
| `INDEX_ROOT` | `$DATA_ROOT/indexes/<dataset>` |
| `CORPUS` | Released Parquet directory selected per dataset |
| `DATASET_BM25_DIR` | `$INDEX_ROOT/bm25` |
| `DATASET_QDRANT_COLLECTION` | Collection from the README mapping |
| `LOCAL_HYBRID_QDRANT_URL` | `http://127.0.0.1:6333` |
| `LOCAL_HYBRID_EMBEDDING_URL` | `http://127.0.0.1:8012/v1` |
| `LOCAL_HYBRID_EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-8B` |
| `EMBEDDING_DIMENSIONS` | `4096` for the dense builder; must match the served model |
| `VLLM_RERANKER_URL` | `http://127.0.0.1:8011` |
| `INDEX_CONCURRENCY` | `8` embedding batches |
| `QDRANT_IMAGE` | Docker image; pin a tag/digest for reproduction |
| `QDRANT_BIN`, `QDRANT_CONFIG` | Optional externally installed server binary and YAML config |

The shell launcher exports the BrowseComp+ qrels/query/answer paths beneath the query root. `BROWSECOMPPLUS_QRELS_GOLD_PATH`, `BROWSECOMPPLUS_QRELS_EVIDENCE_PATH`, `BROWSECOMPPLUS_QUERIES_PATH`, and `BROWSECOMPPLUS_ANSWERS_PATH` override these individually.

## Models and execution

| Variable | Default / purpose |
|---|---|
| `POLICY_MODEL` | Served name `Qwen/Qwen3.5-27B` |
| `POLICY_MODEL_PATH` | `models/Qwen3.5-27B` checkpoint |
| `EMBEDDING_MODEL`, `RERANKER_MODEL` | Corresponding checkpoint paths under `models/` |
| `POLICY_BASE_URL` | Policy OpenAI-compatible endpoint |
| `JUDGE_BASE_URL` | Intent/judge endpoint; defaults to policy endpoint |
| `INTENT_MODEL_BASE_URL` | Explicit override for the intent/judge endpoint |
| `RELEVANCE_JUDGE_MODEL_NAME` | Defaults to the policy served name; also used by the intent planner |
| `SUMMARY_AUDITOR_MODEL_NAME` | Defaults to the policy served name |
| `LOCAL_MODEL_API_KEY` | `EMPTY` for unauthenticated local models |
| `POLICY_MODEL_API_KEY`, `INTENT_MODEL_API_KEY` | Separate endpoint credentials if needed |
| `POLICY_PROTOCOL` | `qwen_chat`; optionally `vllm_tokens` |
| `TENSOR_PARALLEL_SIZE` | `8` for the supplied local server layout |
| `POLICY_GPU_MEMORY`, `EMBEDDING_GPU_MEMORY`, `RERANKER_GPU_MEMORY` | `0.55`, `0.22`, `0.22` |
| `AUTO_START_MODELS` | `0`; set `1` to start models before evaluation |
| `SERVE_ONLY` | `0`; set `1` to start models and skip evaluation |
| `RERANKER` | `vllm`; `none` disables reranking. LongSealQA always uses `none` |
| `N_QUERIES` | `0` = all selected queries |
| `SPLIT` | `all` for BrowseComp+/LongSealQA, `test` for Web/SEC |
| `QUERY_IDS` | Space-separated explicit query IDs |
| `MAX_TURNS`, `MAX_TOKENS`, `PARALLEL`, `SEED` | `40`, `2048`, `4`, `42` |
| `INTENT_SEARCHES_PER_STAGE` | `5` stage search operations before curation |
| `OUT` | Timestamped output directory under `outputs/` |
| `DRY_RUN` | Print resolved run/build/serve configuration, without execution |

The supplied local server launcher binds fixed loopback ports 8000/8012/8011. For other hosts/ports, start those services externally and set endpoint URLs. The embedding/reranking clients expect unauthenticated local-compatible services. The launcher supports separate authenticated policy and intent/judge endpoints.

Legacy `REFERENCE_JUDGE_*`, `SUMMARY_JUDGE_*`, `INTENT_CURATOR_QUERY_DATA_ROOT`, and `V8D_REVISE_INTENT_TOOL` names remain accepted. Prefer current names for new experiments. Legacy BrowseComp+ `LOCAL_HYBRID_BM25_DIR`/collection overrides remain recognized; per-dataset `DATASET_*` overrides take priority.

## Troubleshooting

- **401 / inaccessible SEC query repository:** use an authorized HF account or local query files; see [data.md](data.md).
- **Missing model/checkpoint:** download it or configure an absolute path. Use `DRY_RUN=1 bash scripts/start_model_services.sh all` to inspect launch commands.
- **vLLM allocation/startup failure:** inspect `tmp/service_logs/*.log`; adjust GPU count, model choice, and memory fractions together.
- **Missing index:** download the matching corpus and run its index builder. Build and evaluation must share `DATA_ROOT`, `INDEX_ROOT`, and collection overrides.
- **Cache configuration changed / no build manifest:** use a new cache path and collection. Legacy caches do not contain sufficient provenance for safe automatic reuse.
- **Rerun with a different corpus or model:** use new index paths and a new collection. Do not reuse an index merely because its row count matches.
- **First offline run fails loading tokenizers:** populate tokenizer/model caches with a connected setup run before setting `HF_HUB_OFFLINE=1`.
- **SQLite version too old for Chroma imports:** use a current Python/SQLite build; Linux users may install `pysqlite3-binary`, which the runtime recognizes when available.
