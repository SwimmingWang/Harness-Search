# Development

Install `requirements.txt` into a Python 3.11 environment. Before submitting changes:

```bash
python -m unittest discover -s tests -v
python -m compileall -q harness_search runtime runner datagen scripts
for script in scripts/*.sh dataset_runs/*.sh; do bash -n "$script"; done
```

Keep changes to policy decisions, memory commitment, and audit authority separate. The Memory Operator is the only writer of working memory. Changes to episode transitions should include a synthetic regression test. See [the architecture notes](docs/architecture.md).

Do not commit `.env.local`, `runtime_paths.env`, downloaded benchmarks under `data/`, model weights, indexes, experiment outputs, or archives of old experiments. The only bundled benchmark data are the explicitly released Web/SEC test query snapshots under `datasets/queries/`; keep their checksums, row counts, and provenance in `datasets/manifest.json`. The `.gitignore` also excludes a local `data` symlink. Keep credentials in local configuration or environment variables.

To create a clean source archive from a working research directory:

```bash
python scripts/export_source.py --output dist/Harness-Search-source.tar.gz
```

The exporter uses an explicit source allowlist, excludes backups/caches, rejects symlinks, and does not include Git history. Inspect the generated manifest before publishing. This command does not upload anything. Choose licensing for your original contributions and preserve the upstream notices in [THIRD_PARTY.md](THIRD_PARTY.md).
