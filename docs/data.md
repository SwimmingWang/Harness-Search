# Data sources and local formats

## Sources

| Asset | Source | Local format |
|---|---|---|
| Chunked retrieval corpora | [pat-jj/harness-1-train-data](https://huggingface.co/datasets/pat-jj/harness-1-train-data) | `chunk_id`, `document_text`, `metadata_json` in Parquet |
| BrowseComp+ questions/qrels | [Tevatron/browsecomp-plus](https://huggingface.co/datasets/Tevatron/browsecomp-plus) | TSV, JSONL, and TREC qrels after local de-obfuscation |
| Web test questions | Bundled local snapshot; upstream reference [kellyhongg/web_1_17_test](https://huggingface.co/datasets/kellyhongg/web_1_17_test) | `datasets/queries/web/test.parquet`, 554 rows |
| SEC test questions | Bundled local snapshot, original upstream revision unrecorded | `datasets/queries/sec/test.parquet`, 691 rows |
| Alternative upstream SEC test | [kellyhongg/sec_test_new](https://huggingface.co/datasets/kellyhongg/sec_test_new) | Explicit `--upstream` download, expected 502 rows |
| SEC train questions (optional) | [kellyhongg/1_18_sec_train](https://huggingface.co/datasets/kellyhongg/1_18_sec_train) | Not needed for test evaluation |
| LongSealQA | [vtllms/sealqa](https://huggingface.co/datasets/vtllms/sealqa), `longseal` configuration | `transfer/longseal_test.parquet` |

The two bundled snapshots include questions, answers, and evidence labels, with
SHA-256 checksums and actual row counts in [`datasets/manifest.json`](../datasets/manifest.json).
The 691-row SEC snapshot must not be reported as the 502-row upstream reference
split. No training splits are included. See [`datasets/README.md`](../datasets/README.md)
for the local snapshot provenance and excluded placeholder/duplicate train files.

Upstream access was checked without authentication on 2026-09-21. SEC queries returned an authorization error. The historical Web training path `kellyhongg/1_17_web_train` is not publicly accessible; the similarly named `web_train_1_17` repository is empty. These upstream access requirements do not apply to installing the bundled test snapshots.

The SEC **retrieval corpus** is the released `corpora/sec/train` collection, even when evaluating **test queries**. No released `corpora/sec/test` shard is assumed. BrowseComp+ uses the released chunk IDs, so substituting an independently chunked corpus requires remapping its qrels.

## Install bundled Web/SEC queries

```bash
python scripts/download_queries.py web sec
```

This validates the bundled checksums, schema, and row counts, then copies the
files to `data/queries/{web,sec}/test.parquet`. It makes no Hugging Face requests
for these snapshots. Use `--output /path/to/queries` to install elsewhere.
An existing identical file is reused; a different local file is preserved and
the installer asks for a new output directory.

## Download BrowseComp+, LongSealQA, and corpora

```bash
# BrowseComp+ query/answer/qrels conversion only:
python scripts/download_queries.py browsecompplus
# LongSealQA questions and its per-query documents:
python scripts/download_queries.py longsealqa

# Large corpora only:
bash scripts/download_released_corpora.sh browsecompplus web sec
```

The combined commands `bash scripts/download_data.sh browsecompplus` and
`bash scripts/download_data.sh longsealqa` perform these steps together with any
required shared corpus download. Neither dataset is stored in this repository.

For query revision pinning, pass `--revision <commit>` to the Python downloader and select one dataset. An explicit revision or a `WEB_TEST_QUERY_REPO`/`SEC_TEST_QUERY_REPO` override selects upstream instead of the bundled snapshot. For corpus pinning, export `CORPUS_REVISION=<commit>` before the shell downloader. Record revisions in experiment notes. LongSealQA is maintained upstream, so later downloads can have different query counts.

`HF_TOKEN`, `HUGGINGFACE_TOKEN`, or an existing `hf auth login` credential can authorize downloads. Shell entry points load `.env.local`; direct Python download commands read environment variables. `HF_ENDPOINT` may be set explicitly for a compatible mirror; the default is Hugging Face.

For an explicit upstream Web/SEC query download into a separate directory:

```bash
hf auth login  # When required by the upstream source.
python scripts/download_queries.py --upstream web sec --output data/upstream_queries
```

The upstream path retains its expected row-count checks (Web test: 554; SEC test:
502; SEC train: 3,453). Set `HARNESS_SEARCH_QUERY_DATA_ROOT` to the chosen output
directory when evaluating that alternative split.

## Bring an authorized local query file

Place an existing Web or SEC split at:

```text
$DATA_ROOT/queries/web/test.parquet
$DATA_ROOT/queries/sec/test.parquet
```

The runtime also accepts `.jsonl` and `.json`. Required fields are `query_id`, `query`, `document_ids`, and `answer`. Query IDs are normalized to strings.

- Web `document_ids`: a list of document ID strings. Optional `gold_document_ids` distinguishes final-answer documents; absent values fall back to `document_ids`.
- SEC `document_ids`: a list of fact records with `fact`, `chunk_ids` (a list of chunk ID strings), and `is_final_answer` (boolean). Preserve the released fact-level labels; do not replace them with document IDs.

The loader accepts Python-list strings used by the original tabular exports as well as native Arrow lists. Synthetic examples:

```json
{"query_id":"example-web","query":"A synthetic question","document_ids":["doc42"],"gold_document_ids":["doc42"],"answer":"An example"}
```

```json
{"query_id":"example-sec","query":"A synthetic filing question","document_ids":[{"fact":"A synthetic fact","chunk_ids":["doc42_0"],"is_final_answer":true}],"answer":"An example"}
```

Explicit upstream downloads can replace the selected output file; choose a separate output directory to preserve an existing local split. Download just its corpus with `scripts/download_released_corpora.sh sec`, then build its index and evaluate normally. A custom authorized HF source can be selected with `SEC_TEST_QUERY_REPO` or `WEB_TEST_QUERY_REPO` for the query downloader; row-count/schema validation still applies.

## BrowseComp+ files

The downloader produces:

- `queries.tsv`: query ID and text, quoted TSV with no header.
- `answers.jsonl`: `query_id` and `answer`.
- `qrels_gold.txt`, `qrels_evidence.txt`: `query_id 0 document_id 1`.

Only fields needed by the runtime are decoded and exported. Large obfuscated document bodies are streamed through without being written as a decoded corpus. The shared retrieval corpus is downloaded separately. Benchmark content stays under the ignored data directory.

## Additional adapters

The Python registry also retains `bc_plus`, `2wikimultihopqa`, `hotpotqa`, and `musique` for existing experiments. They are not covered by the four-dataset setup scripts. The latter three require manually prepared `data/<dataset>/dev.jsonl` with `id`, `question`, and `golden_answers`, plus a matching retrieval corpus/index. See `LocalMultiHopQADataset` before using them; this repository does not provide their complete data acquisition pipeline.
