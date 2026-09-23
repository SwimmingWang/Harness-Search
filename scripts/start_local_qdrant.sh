#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if curl -fsS --max-time 3 "$LOCAL_HYBRID_QDRANT_URL/collections" >/dev/null 2>&1; then
  echo "Qdrant is ready at $LOCAL_HYBRID_QDRANT_URL"; exit 0
fi
case "$LOCAL_HYBRID_QDRANT_URL" in http://127.0.0.1:6333|http://localhost:6333) ;;
  *) echo "Start the configured Qdrant endpoint externally: $LOCAL_HYBRID_QDRANT_URL" >&2; exit 2;; esac
if [[ -n "${QDRANT_BIN:-}" ]]; then
  CONFIG="${QDRANT_CONFIG:-$DATA_ROOT/qdrant_server/config.yaml}"
  [[ -f "$CONFIG" ]] || { echo "Provide QDRANT_CONFIG for the standalone binary: $CONFIG" >&2; exit 2; }
  mkdir -p "$ROOT/tmp/service_logs"
  nohup "$QDRANT_BIN" --config-path "$CONFIG" > "$ROOT/tmp/service_logs/qdrant.log" 2>&1 < /dev/null &
else
  docker compose -f "$ROOT/compose.yaml" up -d qdrant
fi
for ((i=0; i<120; i++)); do
  if curl -fsS --max-time 3 "$LOCAL_HYBRID_QDRANT_URL/collections" >/dev/null 2>&1; then
    echo "Qdrant is ready at $LOCAL_HYBRID_QDRANT_URL"; exit 0
  fi
  sleep 1
done
echo "Qdrant startup timed out; inspect Docker or tmp/service_logs/qdrant.log" >&2
exit 1
