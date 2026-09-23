#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || $# == 0 ]]; then
  echo "Usage: $0 {browsecompplus|web|sec|longsealqa} [runner.evaluate options]"
  echo "N_QUERIES=0 runs the full split; DRY_RUN=1 prints the command; AUTO_START_MODELS=1 starts local models."
  exit 0
fi
DATASET="$1"; shift
case "$DATASET" in browsecompplus|web|sec|longsealqa) ;; *) echo "Unknown dataset: $DATASET" >&2; exit 2;; esac
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ "$DATASET" == browsecompplus || "$DATASET" == longsealqa ]]; then
  SPLIT="${SPLIT:-all}"
else
  SPLIT="${SPLIT:-test}"
fi
RERANKER="${RERANKER:-vllm}"
if [[ "$DATASET" == longsealqa ]]; then
  RERANKER=none
else
  dataset_paths "$DATASET"
  export LOCAL_HYBRID_BM25_DIR="$BM25_DIR"
  export LOCAL_HYBRID_QDRANT_COLLECTION="$COLLECTION"
  unset LOCAL_HYBRID_QDRANT_PATH
fi
export V8D_REDIRECT_TOOL=1 V8D_INTENT_STATE_TRACKING=1 V8D_VERIFY_TOOL=0
export V8D_AUTO_POPULATE_FIRST_SEARCH=0 V3_JUDGES_ENABLED=1
export INTENT_SEARCHES_PER_STAGE="${INTENT_SEARCHES_PER_STAGE:-5}"
OUT="${OUT:-$ROOT/outputs/${DATASET}_$(date +%Y%m%d_%H%M%S)}"
export SAVE_TRAJECTORIES=1 TRAJECTORY_SAVE_PATH="$OUT/trajectories"
args=(--dataset "$DATASET" --split "$SPLIT" --collection-split "${COLLECTION_SPLIT:-test}"
  --n-queries "${N_QUERIES:-0}" --seed "${SEED:-42}" --max-turns "${MAX_TURNS:-40}"
  --max-tokens "${MAX_TOKENS:-2048}" --parallel "${PARALLEL:-4}"
  --base-url "$POLICY_BASE_URL" --model "$POLICY_MODEL" --fallback-model "${FALLBACK_MODEL:-$POLICY_MODEL}"
  --policy-protocol "${POLICY_PROTOCOL:-qwen_chat}" --reranker "$RERANKER"
  --output "$OUT/eval_results.json" --partial-output "$OUT/eval_results.partial.jsonl")
if [[ -n "${QUERY_IDS:-}" ]]; then
  read -r -a query_ids <<< "$QUERY_IDS"
  args+=(--query-ids "${query_ids[@]}")
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "$PYTHON" -m runner.evaluate "${args[@]}" "$@"; printf '\n'
  if [[ "$DATASET" != longsealqa ]]; then printf 'BM25=%s\nQdrant=%s/%s\n' "$BM25_DIR" "$LOCAL_HYBRID_QDRANT_URL" "$COLLECTION"; fi
  exit 0
fi
if [[ "${AUTO_START_MODELS:-0}" == 1 || "${SERVE_ONLY:-0}" == 1 ]]; then
  mode=all; [[ "$DATASET" == longsealqa ]] && mode=policy
  "$ROOT/scripts/start_model_services.sh" "$mode"
fi
[[ "${SERVE_ONLY:-0}" == 1 ]] && exit 0
curl -fsS --max-time 5 -H "Authorization: Bearer $POLICY_MODEL_API_KEY" "$POLICY_BASE_URL/models" >/dev/null
curl -fsS --max-time 5 -H "Authorization: Bearer $INTENT_MODEL_API_KEY" "$INTENT_MODEL_BASE_URL/models" >/dev/null
if [[ "$DATASET" != longsealqa ]]; then
  [[ -s "$BM25_DIR/manifest.json" ]] || { echo "Missing BM25 index: $BM25_DIR; run scripts/build_dataset_indexes.sh $DATASET" >&2; exit 2; }
  curl -fsS --max-time 5 "$LOCAL_HYBRID_EMBEDDING_URL/models" >/dev/null
  curl -fsS --max-time 10 "$LOCAL_HYBRID_QDRANT_URL/collections/$COLLECTION" >/dev/null
  if [[ "$RERANKER" != none ]]; then curl -fsS --max-time 5 "$VLLM_RERANKER_URL/v1/models" >/dev/null; fi
fi
mkdir -p "$OUT"
cd "$ROOT"
"$PYTHON" -m runner.evaluate "${args[@]}" "$@" 2>&1 | tee "$OUT/eval.log"
