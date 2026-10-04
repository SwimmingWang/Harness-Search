#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo "Usage: $0 [browsecompplus web sec longsealqa]"
  echo "Reuses local Web/SEC test queries, installs bundled or downloads missing queries, and downloads retrieval corpora; no arguments selects all four."
  exit 0
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if (( $# == 0 )); then set -- browsecompplus web sec longsealqa; fi
# Prefer validated local queries before bundled snapshots or upstream downloads.
"$PYTHON" "$ROOT/scripts/download_queries.py" --output "$HARNESS_SEARCH_QUERY_DATA_ROOT" \
  --transfer-output "$DATA_ROOT/transfer" --split test "$@"
exec bash "$ROOT/scripts/download_released_corpora.sh" "$@"
