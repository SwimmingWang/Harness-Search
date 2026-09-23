# Bundled evaluation queries

This directory contains two local evaluation snapshots with questions, answers,
and evidence annotations. Full document corpora are downloaded separately.

| File | Rows | Contents |
|---|---:|---|
| [`queries/web/test.parquet`](queries/web/test.parquet) | 554 | Query, answer, relevant document IDs, gold document IDs, distractor IDs |
| [`queries/sec/test.parquet`](queries/sec/test.parquet) | 691 | Query, answer, fact-level evidence and final-answer labels |

File sizes and SHA-256 hashes are recorded in [`manifest.json`](manifest.json).
The release preserves the original local Parquet bytes and schema.

## Install

From the repository root:

```bash
python -m pip install -r requirements-data.txt
python scripts/download_queries.py web sec
# Download matching full corpora when needed:
bash scripts/download_released_corpora.sh web sec
```

The query installer uses these checked-in files without calling Hugging Face.
It verifies their hashes and copies them to `data/queries/`. The combined
`bash scripts/download_data.sh web sec` also downloads the corpora.

## Provenance and split boundaries

The snapshots were exported from the local Harness-Search evaluation setup on
2026-09-23. Their original upstream revision identifiers were not recorded.
The configured Web upstream reference is
[`kellyhongg/web_1_17_test`](https://huggingface.co/datasets/kellyhongg/web_1_17_test).

The bundled SEC snapshot contains **691 distinct query IDs**, whereas the
configured alternative upstream
[`kellyhongg/sec_test_new`](https://huggingface.co/datasets/kellyhongg/sec_test_new)
is expected to contain 502 queries. This local snapshot is not claimed to be
that 502-query release. Report the manifest hash and query count in experiments;
scores from different query sets are not directly comparable.

No training splits are included: the local Web train file contained one
`__offline_train_placeholder__` row, and the local SEC train file was byte-for-byte
identical to its test file. Neither is an independent training set. Keep these
released test questions out of SFT data and training-data selection.

Dataset terms and attribution remain those of the upstream sources; the code's
license does not relicense these snapshots. See [third-party notices](../THIRD_PARTY.md).

## Other datasets

BrowseComp+ and LongSealQA are not bundled. Download them with:

```bash
bash scripts/download_data.sh browsecompplus
bash scripts/download_data.sh longsealqa
```

These scripts use [Tevatron/browsecomp-plus](https://huggingface.co/datasets/Tevatron/browsecomp-plus)
and the `longseal` configuration of [vtllms/sealqa](https://huggingface.co/datasets/vtllms/sealqa).
Full corpus sources and formats are documented in [docs/data.md](../docs/data.md).
