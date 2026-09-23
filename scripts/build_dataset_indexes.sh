#!/usr/bin/env bash
set -euo pipefail
if [[ $# == 0 || "${1:-}" == --help ]]; then
  echo "Usage: $0 {browsecompplus|web|sec}"
  echo "Builds matching BM25 and Qdrant indexes. Requires Qdrant and embedding services."
  exit 0
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
DATASET="$1"
dataset_paths "$DATASET"
if [[ -n "${INDEX_LIMIT:-}" ]]; then
  echo "INDEX_LIMIT is only supported by the low-level Python builders; use separate smoke index paths for partial corpora." >&2
  exit 2
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'corpus=%s\nbm25=%s\ncollection=%s\nexpected=%s\n' "$CORPUS" "$BM25_DIR" "$COLLECTION" "$EXPECTED_CORPUS_COUNT"
  exit 0
fi
[[ -d "$CORPUS" ]] || { echo "Missing corpus: $CORPUS; run scripts/download_data.sh $DATASET" >&2; exit 2; }
curl -fsS --max-time 5 "$LOCAL_HYBRID_EMBEDDING_URL/models" >/dev/null
curl -fsS --max-time 5 "$LOCAL_HYBRID_QDRANT_URL/collections" >/dev/null
if [[ ! -s "$BM25_DIR/manifest.json" ]]; then
  "$PYTHON" "$ROOT/scripts/build_dataset_bm25.py" --corpus "$CORPUS" --output "$BM25_DIR" --expected "$EXPECTED_CORPUS_COUNT"
else
  "$PYTHON" - "$BM25_DIR/manifest.json" "$EXPECTED_CORPUS_COUNT" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    manifest = json.load(f)
if manifest.get("count") != int(sys.argv[2]):
    raise SystemExit("Existing BM25 count does not match this corpus. Use a separate INDEX_ROOT.")
print("Reusing BM25 index:", sys.argv[1])
PY
fi
"$PYTHON" "$ROOT/scripts/build_dataset_qdrant.py" --corpus "$CORPUS" \
  --url "$LOCAL_HYBRID_QDRANT_URL" --cache "$INDEX_ROOT/embedding_cache" \
  --collection "$COLLECTION" --embedding-url "$LOCAL_HYBRID_EMBEDDING_URL" \
  --embedding-model "$LOCAL_HYBRID_EMBEDDING_MODEL" --dimensions "${EMBEDDING_DIMENSIONS:-4096}" \
  --expected "$EXPECTED_CORPUS_COUNT" --concurrency "${INDEX_CONCURRENCY:-8}"
printf 'Ready: dataset=%s bm25=%s qdrant_collection=%s\n' "$DATASET" "$BM25_DIR" "$COLLECTION"
