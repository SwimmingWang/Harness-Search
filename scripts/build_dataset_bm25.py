#!/usr/bin/env python3
"""Build a local bm25s index for an Harness-Search corpus."""

import argparse
import json
import time
from pathlib import Path

import bm25s
import Stemmer
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=0, help="optional expected row count")
    parser.add_argument("--limit", type=int, default=0, help="optional prefix size for smoke indexes")
    args = parser.parse_args()
    paths = sorted(args.corpus.rglob("train-*.parquet")) if args.corpus.is_dir() else []
    jsonl_path = args.corpus if args.corpus.is_file() and args.corpus.suffix == ".jsonl" else None
    if not paths and jsonl_path is None:
        raise RuntimeError(f"no parquet shards or JSONL corpus found at {args.corpus}")

    started = time.time()
    ids, texts = [], []
    if jsonl_path is not None:
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                ids.append(str(row["id"])); texts.append(str(row["contents"]))
                if args.limit and len(ids) >= args.limit:
                    break
        paths = [jsonl_path]
    else:
        for shard, path in enumerate(paths, 1):
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=8192, columns=["chunk_id", "document_text"]
            ):
                data = batch.to_pydict()
                for cid, text in zip(data["chunk_id"], data["document_text"]):
                    ids.append(str(cid)); texts.append(str(text))
                    if args.limit and len(ids) >= args.limit:
                        break
                if args.limit and len(ids) >= args.limit: break
            if args.limit and len(ids) >= args.limit: break
            print(json.dumps({"event": "loaded_shard", "shard": shard, "rows": len(ids)}), flush=True)
    if args.expected and len(ids) != args.expected:
        raise RuntimeError(f"row count {len(ids)} != {args.expected}")

    stemmer = Stemmer.Stemmer("english")
    tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=True)
    retriever = bm25s.BM25(method="lucene")
    retriever.index(tokens, show_progress=True)
    args.output.mkdir(parents=True, exist_ok=True)
    retriever.save(args.output, show_progress=True)
    with (args.output / "chunk_ids.jsonl").open("w") as handle:
        for chunk_id in ids:
            handle.write(json.dumps(chunk_id) + "\n")
    (args.output / "manifest.json").write_text(json.dumps({
        "count": len(ids), "method": "lucene", "stemmer": "english",
        "stopwords": "en", "elapsed_seconds": time.time() - started, "shards": len(paths),
    }, indent=2))
    print(json.dumps({"event": "all_complete", "count": len(ids), "elapsed": time.time() - started}), flush=True)


if __name__ == "__main__":
    main()
