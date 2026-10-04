#!/usr/bin/env bash
# Shared configuration; source from Bash entry points.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Precedence: exported environment > .env.local > runtime_paths.env > defaults.
# Save exported values before sourcing local, user-owned configuration files.
_hs_original_env="$(export -p)"
for _hs_config in "${RUNTIME_PATHS:-$ROOT/runtime_paths.env}" "${CONFIG_ENV:-$ROOT/.env.local}"; do
  if [[ -f "$_hs_config" ]]; then
    set -a
    source "$_hs_config"
    set +a
  fi
done
eval "$_hs_original_env"
unset _hs_original_env _hs_config
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
PYTHON="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="${PYTHON_BIN:-python3}"
export HARNESS_SEARCH_QUERY_DATA_ROOT="${HARNESS_SEARCH_QUERY_DATA_ROOT:-${INTENT_CURATOR_QUERY_DATA_ROOT:-$DATA_ROOT/queries}}"
export LOCAL_HYBRID_QDRANT_URL="${LOCAL_HYBRID_QDRANT_URL:-http://127.0.0.1:6333}"
export LOCAL_HYBRID_EMBEDDING_URL="${LOCAL_HYBRID_EMBEDDING_URL:-http://127.0.0.1:8012/v1}"
export LOCAL_HYBRID_EMBEDDING_MODEL="${LOCAL_HYBRID_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"
export VLLM_RERANKER_URL="${VLLM_RERANKER_URL:-http://127.0.0.1:8011}"
POLICY_MODEL="${POLICY_MODEL:-${POLICY_MODEL_NAME:-Qwen/Qwen3.5-27B}}"
POLICY_BASE_URL="${POLICY_BASE_URL:-http://127.0.0.1:8000/v1}"
export INTENT_MODEL_BASE_URL="${INTENT_MODEL_BASE_URL:-${JUDGE_BASE_URL:-$POLICY_BASE_URL}}"
export POLICY_MODEL_API_KEY="${POLICY_MODEL_API_KEY:-${LOCAL_MODEL_API_KEY:-EMPTY}}"
export INTENT_MODEL_API_KEY="${INTENT_MODEL_API_KEY:-${LOCAL_MODEL_API_KEY:-EMPTY}}"
export INTENT_MODEL_HEADER_API_KEY="${INTENT_MODEL_HEADER_API_KEY:-$INTENT_MODEL_API_KEY}"
export RELEVANCE_JUDGE_MODEL_NAME="${RELEVANCE_JUDGE_MODEL_NAME:-${REFERENCE_JUDGE_MODEL_NAME:-${INTENT_MODEL_NAME:-$POLICY_MODEL}}}"
export RELEVANCE_JUDGE_FALLBACK_NAME="${RELEVANCE_JUDGE_FALLBACK_NAME:-${REFERENCE_JUDGE_FALLBACK_NAME:-$RELEVANCE_JUDGE_MODEL_NAME}}"
export SUMMARY_AUDITOR_MODEL_NAME="${SUMMARY_AUDITOR_MODEL_NAME:-${SUMMARY_JUDGE_MODEL_NAME:-$POLICY_MODEL}}"
export SUMMARY_AUDITOR_FALLBACK_NAME="${SUMMARY_AUDITOR_FALLBACK_NAME:-${SUMMARY_JUDGE_FALLBACK_NAME:-$SUMMARY_AUDITOR_MODEL_NAME}}"
export BROWSECOMPPLUS_QRELS_GOLD_PATH="${BROWSECOMPPLUS_QRELS_GOLD_PATH:-$HARNESS_SEARCH_QUERY_DATA_ROOT/browsecompplus/qrels_gold.txt}"
export BROWSECOMPPLUS_QRELS_EVIDENCE_PATH="${BROWSECOMPPLUS_QRELS_EVIDENCE_PATH:-$HARNESS_SEARCH_QUERY_DATA_ROOT/browsecompplus/qrels_evidence.txt}"
export BROWSECOMPPLUS_QUERIES_PATH="${BROWSECOMPPLUS_QUERIES_PATH:-$HARNESS_SEARCH_QUERY_DATA_ROOT/browsecompplus/queries.tsv}"
export BROWSECOMPPLUS_ANSWERS_PATH="${BROWSECOMPPLUS_ANSWERS_PATH:-$HARNESS_SEARCH_QUERY_DATA_ROOT/browsecompplus/answers.jsonl}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export ANONYMIZED_TELEMETRY=False POSTHOG_DISABLED=1

dataset_paths() {
  local dataset="$1" corpus_split collection default_index_root
  default_index_root="$DATA_ROOT/indexes/$dataset"
  case "$dataset" in
    browsecompplus) corpus_split=test; EXPECTED_CORPUS_COUNT=1144886; collection=browsecompplus_qwen3_embedding_8b_4096_local ;;
    web) corpus_split=test; EXPECTED_CORPUS_COUNT=54735; collection="${WEB_QDRANT_COLLECTION:-web_test_1_17_qwen3_embedding_8b_4096_full}"; default_index_root="${WEB_INDEX_ROOT:-$default_index_root}" ;;
    sec) corpus_split=train; EXPECTED_CORPUS_COUNT=2115106; collection=sec_1_4_qwen3_embedding_8b_4096_full ;;
    *) echo "No shared corpus for dataset: $dataset" >&2; return 2 ;;
  esac
  CORPUS="${CORPUS:-$DATA_ROOT/corpora/$dataset/corpora/$dataset/$corpus_split}"
  INDEX_ROOT="${INDEX_ROOT:-$default_index_root}"
  BM25_DIR="${DATASET_BM25_DIR:-$INDEX_ROOT/bm25}"
  COLLECTION="${DATASET_QDRANT_COLLECTION:-$collection}"
  if [[ "$dataset" == browsecompplus ]]; then
    BM25_DIR="${DATASET_BM25_DIR:-${LOCAL_HYBRID_BM25_DIR:-$BM25_DIR}}"
    COLLECTION="${DATASET_QDRANT_COLLECTION:-${LOCAL_HYBRID_QDRANT_COLLECTION:-$COLLECTION}}"
  fi
}
