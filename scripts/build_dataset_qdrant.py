#!/usr/bin/env python3
"""Build a resumable Qdrant dense index and local embedding cache."""

from __future__ import annotations

import argparse, json, random, threading, time
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
from openai import OpenAI
from qdrant_client import QdrantClient, models

EXPECTED = 1_144_886
COLLECTION = "browsecompplus_qwen3_embedding_8b_4096_local"


def log(event, **kw):
    print(json.dumps({"event": event, "time": time.time(), **kw}), flush=True)


def rows(corpus: Path):
    if corpus.is_file() and corpus.suffix == ".jsonl":
        with corpus.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line); text = str(row["contents"]); cid = str(row["id"])
                title = text.splitlines()[0].strip('"') if text else ""
                yield cid, text, {"source": cid, "title": title}
        return
    paths = sorted(corpus.rglob("train-*.parquet"))
    if not paths:
        raise RuntimeError(f"no parquet shards or JSONL corpus found at {corpus}")
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2048, columns=["chunk_id", "document_text", "metadata_json"]):
            d = batch.to_pydict()
            for cid, text, raw_meta in zip(d["chunk_id"], d["document_text"], d["metadata_json"]):
                meta = json.loads(raw_meta) if raw_meta else {}
                meta.setdefault("source", str(cid).split("_", 1)[0])
                yield str(cid), str(text), meta


def packed(source, enc, max_items, max_tokens):
    ids, texts, metas, count = [], [], [], 0
    for cid, text, meta in source:
        n = len(enc.encode(text, disallowed_special=()))
        if ids and (len(ids) >= max_items or count + n > max_tokens):
            yield ids, texts, metas, count
            ids, texts, metas, count = [], [], [], 0
        ids.append(cid); texts.append(text); metas.append(meta); count += n
    if ids:
        yield ids, texts, metas, count


def embed(texts, attempts, base_url, model, dimensions):
    client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=180, max_retries=0)
    for attempt in range(1, attempts + 1):
        try:
            result = client.embeddings.create(model=model, input=texts, encoding_format="float")
            vecs = [x.embedding for x in sorted(result.data, key=lambda x: x.index)]
            if len(vecs) != len(texts) or any(len(x) != dimensions for x in vecs):
                raise RuntimeError("embedding shape mismatch")
            return vecs
        except Exception as exc:
            if attempt == attempts: raise
            delay = min(60, 2 ** (attempt - 1) + random.random())
            log("embedding_retry", attempt=attempt, delay=delay, error=str(exc)); time.sleep(delay)


def save(path, ids, vecs, dimensions):
    tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(pa.table({"chunk_id": ids, "embedding": pa.array(vecs, type=pa.list_(pa.float32(), dimensions))}), tmp, compression="zstd")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True); p.add_argument("--qdrant", type=Path)
    p.add_argument("--url", help="Qdrant server URL; mutually exclusive with --qdrant")
    p.add_argument("--cache", type=Path, required=True); p.add_argument("--collection", default=COLLECTION)
    p.add_argument("--embedding-url", default="http://127.0.0.1:8012/v1")
    p.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-8B")
    p.add_argument("--dimensions", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=8); p.add_argument("--max-items", type=int, default=400)
    p.add_argument("--max-tokens", type=int, default=240000); p.add_argument("--attempts", type=int, default=10)
    p.add_argument("--expected", type=int, default=0, help="optional expected row count"); p.add_argument("--limit", type=int)
    a = p.parse_args(); a.cache.mkdir(parents=True, exist_ok=True)
    if bool(a.url) == bool(a.qdrant):
        raise RuntimeError("pass exactly one of --url or --qdrant")
    if a.qdrant:
        a.qdrant.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(url=a.url, timeout=180) if a.url else QdrantClient(path=str(a.qdrant))
    manifest_path = a.cache / "build_config.json"
    files = [a.corpus] if a.corpus.is_file() else sorted(a.corpus.rglob("train-*.parquet"))
    if not files:
        raise RuntimeError(f"No corpus shards at {a.corpus}")
    signature = {
        "corpus": [{"path": str(p.resolve()), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files],
        "model": a.embedding_model, "dimensions": a.dimensions,
        "max_items": a.max_items, "max_tokens": a.max_tokens, "limit": a.limit,
        "collection": a.collection, "qdrant": a.url or str(a.qdrant.resolve()),
    }
    existing = {x.name for x in client.get_collections().collections}
    old_count = client.count(a.collection, exact=True).count if a.collection in existing else 0
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != signature:
            raise RuntimeError("Index/cache configuration changed. Use a new cache directory and collection.")
    elif old_count or any(a.cache.glob("*.parquet")):
        raise RuntimeError("Existing index/cache has no build_config.json; choose a new cache directory and collection.")
    else:
        manifest_path.write_text(json.dumps(signature, indent=2))
    if a.collection not in existing:
        client.create_collection(a.collection, vectors_config=models.VectorParams(size=a.dimensions, distance=models.Distance.COSINE, on_disk=True), hnsw_config=models.HnswConfigDiff(on_disk=True), optimizers_config=models.OptimizersConfigDiff(memmap_threshold=10000))
    completed = client.count(a.collection, exact=True).count
    if a.corpus.is_file() and a.corpus.suffix == ".jsonl":
        with a.corpus.open(encoding="utf-8") as handle:
            available = sum(1 for _ in handle)
    else:
        available = sum(pq.ParquetFile(path).metadata.num_rows for path in sorted(a.corpus.rglob("train-*.parquet")))
    expected = a.expected or available
    if a.expected and a.expected != available:
        raise RuntimeError(f"row count {available} != expected {a.expected}")
    target = min(a.limit or expected, expected)
    log("resume", completed=completed, target=target)
    if completed > target:
        raise RuntimeError("Collection contains more rows than the target; use a separate collection.")
    # Count is not a safe resume cursor: parallel batches can leave holes after a
    # crash. Replay deterministic IDs from the start, using cached embeddings.
    source = islice(rows(a.corpus), target)
    batches = iter(packed(source, tiktoken.get_encoding("cl100k_base"), a.max_items, a.max_tokens))
    completed = 0
    started = time.time(); initial = 0
    write_lock = threading.Lock() if not a.url else None

    def work(offset, batch):
        ids, texts, metas, tokens = batch; end = offset + len(ids); cache = a.cache / f"{offset:07d}_{end:07d}.parquet"
        if cache.exists():
            data = pq.read_table(cache).to_pydict(); vecs = data["embedding"]
            if data["chunk_id"] != ids or len(vecs) != len(ids) or any(len(v) != a.dimensions for v in vecs):
                raise RuntimeError(f"cache mismatch {cache}")
        else:
            vecs = embed(texts, a.attempts, a.embedding_url, a.embedding_model, a.dimensions); save(cache, ids, vecs, a.dimensions)
        points = [models.PointStruct(id=offset+i, vector=v, payload={"chunk_id": cid, "document": text, "metadata": meta}) for i,(cid,text,meta,v) in enumerate(zip(ids,texts,metas,vecs))]
        # Qdrant local mode persists payloads through SQLite, which only supports
        # one writer. Keep embedding requests parallel, but serialize upserts.
        if write_lock:
            with write_lock:
                client.upload_points(a.collection, points, batch_size=100, parallel=1, wait=True)
        else:
            client.upload_points(a.collection, points, batch_size=100, parallel=1, wait=True)
        return len(ids), tokens, cache.name

    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        while completed < target:
            wave=[]; offset=completed
            for _ in range(a.concurrency):
                try: batch=next(batches)
                except StopIteration: break
                remain=target-offset
                if remain <= 0: break
                if len(batch[0]) > remain: batch=tuple([x[:remain] for x in batch[:3]] + [batch[3]])
                wave.append((offset,batch)); offset += len(batch[0])
            if not wave: break
            for f in as_completed([pool.submit(work,o,b) for o,b in wave]):
                n,t,c=f.result(); log("batch_complete",rows=n,tokens=t,cache=c)
            completed=offset
            elapsed=max(time.time()-started,1e-6); rate=(completed-initial)/elapsed
            log("progress",completed=completed,total=target,rate_rows_s=round(rate,3),eta_seconds=round((target-completed)/rate) if rate else None)
    completed=client.count(a.collection,exact=True).count
    if completed != target: raise RuntimeError(f"count {completed}, target {target}")
    client.close()
    log("all_complete",collection=a.collection,count=completed)

if __name__ == "__main__": main()
