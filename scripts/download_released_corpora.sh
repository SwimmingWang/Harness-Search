#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if (( $# == 0 )); then set -- browsecompplus web sec; fi
for dataset in "$@"; do
  [[ "$dataset" == longsealqa ]] && continue
  dataset_paths "$dataset"
  "$PYTHON" - "$dataset" "$DATA_ROOT/corpora/$dataset" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download
name, output = sys.argv[1:]
split = "train" if name == "sec" else "test"
snapshot_download(
    repo_id="pat-jj/harness-1-train-data", repo_type="dataset", local_dir=output,
    revision=os.environ.get("CORPUS_REVISION", "main"),
    allow_patterns=[f"corpora/{name}/{split}/*.parquet"],
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None,
    max_workers=int(os.environ.get("DOWNLOAD_WORKERS", "2")),
)
print(f"ready corpus={name} path={output}/corpora/{name}/{split}")
PY
done
