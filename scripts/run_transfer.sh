#!/usr/bin/env bash
# Compatibility entry point; use scripts/run_eval.sh for new runs.
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/run_eval.sh" "$@"
