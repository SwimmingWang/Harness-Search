#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# == 0 )); then set -- browsecompplus web sec longsealqa; fi
for dataset in "$@"; do
  case "$dataset" in browsecompplus|longsealqa) split=all;; web|sec) split=test;; *) echo "Unknown dataset: $dataset" >&2; exit 2;; esac
  N_QUERIES=0 SPLIT="$split" bash "$ROOT/scripts/run_eval.sh" "$dataset"
done
