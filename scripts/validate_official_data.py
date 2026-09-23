#!/usr/bin/env python3
"""Fail fast when a paper run is pointed at incomplete or mismatched data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

import pyarrow.parquet as pq


def parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("web", "sec"), required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--bm25-dir", type=Path, required=True)
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--expected-corpus", type=int, required=True)
    args = parser.parse_args()

    query_file = args.query_root / args.dataset / f"{args.split}.parquet"
    if not query_file.is_file():
        raise SystemExit(f"missing official query split: {query_file}")
    query_count = parquet_rows(query_file)
    if query_count != args.expected_queries:
        raise SystemExit(
            f"query count mismatch: expected={args.expected_queries} actual={query_count} file={query_file}"
        )

    manifest_file = args.bm25_dir / "manifest.json"
    if not manifest_file.is_file():
        raise SystemExit(f"missing BM25 manifest: {manifest_file}")
    manifest = json.loads(manifest_file.read_text())
    bm25_count = next((manifest[k] for k in ("count", "num_documents", "num_chunks", "corpus_size") if k in manifest), None)
    if bm25_count is None:
        chunk_ids = args.bm25_dir / "chunk_ids.jsonl"
        bm25_count = sum(1 for _ in chunk_ids.open()) if chunk_ids.is_file() else None
    if bm25_count != args.expected_corpus:
        raise SystemExit(f"BM25 count mismatch: expected={args.expected_corpus} actual={bm25_count}")

    opener = build_opener(ProxyHandler({}))
    with opener.open(f"{args.qdrant_url.rstrip('/')}/collections/{args.collection}", timeout=10) as response:
        result = json.load(response)["result"]
    point_count = result.get("points_count")
    if point_count != args.expected_corpus:
        raise SystemExit(f"Qdrant count mismatch: expected={args.expected_corpus} actual={point_count}")

    print(
        f"official data validated: dataset={args.dataset} queries={query_count} "
        f"bm25={bm25_count} qdrant={point_count}"
    )


if __name__ == "__main__":
    main()
