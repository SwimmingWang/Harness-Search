#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-all}"
case "$MODE" in all|policy|retrieval) ;; *) echo "Usage: $0 {all|policy|retrieval}"; exit 2;; esac
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
VLLM="${VLLM_BIN:-$ROOT/.venv-serve/bin/vllm}"
SERVICE_LOG_DIR="${SERVICE_LOG_DIR:-$ROOT/tmp/service_logs}"
TP="${TENSOR_PARALLEL_SIZE:-8}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
model_ready() { curl -fsS --max-time 3 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }
launch() {
  local name="$1" port="$2"; shift 2
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "$VLLM" serve "$@" --host 127.0.0.1 --port "$port" --tensor-parallel-size "$TP"; printf '\n'
    return
  fi
  if model_ready "$port"; then echo "$name already ready on port $port"; return; fi
  [[ -x "$VLLM" ]] || { echo "Missing vLLM: $VLLM; install vllm==0.20.2 in .venv-serve" >&2; exit 2; }
  mkdir -p "$SERVICE_LOG_DIR"
  nohup env VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
    "$VLLM" serve "$@" --host 127.0.0.1 --port "$port" --tensor-parallel-size "$TP" \
    >"$SERVICE_LOG_DIR/$name.log" 2>&1 < /dev/null &
  local pid=$! elapsed=0
  echo "$pid" > "$SERVICE_LOG_DIR/$name.pid"
  echo "Starting $name (pid=$pid); log=$SERVICE_LOG_DIR/$name.log"
  until model_ready "$port"; do
    kill -0 "$pid" 2>/dev/null || { tail -n 40 "$SERVICE_LOG_DIR/$name.log" >&2; exit 1; }
    (( elapsed < STARTUP_TIMEOUT )) || { echo "$name startup timed out; inspect $SERVICE_LOG_DIR/$name.log" >&2; exit 1; }
    sleep 5; elapsed=$((elapsed + 5))
  done
}
if [[ "$MODE" != retrieval ]]; then
  launch policy 8000 "${POLICY_MODEL_PATH:-$ROOT/models/Qwen3.5-27B}" \
    --served-model-name "$POLICY_MODEL" --gpu-memory-utilization "${POLICY_GPU_MEMORY:-0.55}" \
    --max-model-len 32768 --max-num-batched-tokens 16384 --trust-remote-code
fi
if [[ "$MODE" != policy ]]; then
  launch embedding 8012 "${EMBEDDING_MODEL:-$ROOT/models/Qwen3-Embedding-8B}" \
    --served-model-name "$LOCAL_HYBRID_EMBEDDING_MODEL" --gpu-memory-utilization "${EMBEDDING_GPU_MEMORY:-0.22}" \
    --max-model-len 8192 --trust-remote-code --runner pooling
  launch reranker 8011 "${RERANKER_MODEL:-$ROOT/models/Qwen3-Reranker-8B}" \
    --served-model-name Qwen/Qwen3-Reranker-8B --gpu-memory-utilization "${RERANKER_GPU_MEMORY:-0.22}" \
    --max-model-len 8192 --trust-remote-code --runner pooling \
    --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
fi
