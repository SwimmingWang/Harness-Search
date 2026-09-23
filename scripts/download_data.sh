#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo "Usage: $0 [browsecompplus web sec longsealqa]"
  echo "Installs bundled Web/SEC test queries, downloads other queries and retrieval corpora; no arguments selects all four."
  exit 0
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if (( $# == 0 )); then set -- browsecompplus web sec longsealqa; fi
# Bundled-query validation and upstream access checks precede corpus downloads.
"$PYTHON" "$ROOT/scripts/download_queries.py" --output "$HARNESS_SEARCH_QUERY_DATA_ROOT" \
  --transfer-output "$DATA_ROOT/transfer" --split test "$@"
exec "$ROOT/scripts/download_released_corpora.sh" "$@"
