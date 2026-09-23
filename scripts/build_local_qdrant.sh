#!/usr/bin/env bash
# Compatibility entry point for a dense-only build.
set -euo pipefail
if (( $# == 0 )); then echo "Usage: $0 {browsecompplus|web|sec} [builder options]"; exit 2; fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
dataset_paths "$1"; shift
exec "$PYTHON" "$ROOT/scripts/build_dataset_qdrant.py" --corpus "$CORPUS" \
  --url "$LOCAL_HYBRID_QDRANT_URL" --cache "$INDEX_ROOT/embedding_cache" \
  --collection "$COLLECTION" --expected "$EXPECTED_CORPUS_COUNT" \
  --embedding-url "$LOCAL_HYBRID_EMBEDDING_URL" --embedding-model "$LOCAL_HYBRID_EMBEDDING_MODEL" \
  --dimensions "${EMBEDDING_DIMENSIONS:-4096}" --concurrency "${INDEX_CONCURRENCY:-8}" "$@"
