"""Portable launch, local data, and source export regression tests (no network)."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.export_source import export, source_files

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def run_script(self, script, *args, **overrides):
        env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"],
               "CONFIG_ENV": "/dev/null", "RUNTIME_PATHS": "/dev/null", "DRY_RUN": "1", **overrides}
        return subprocess.check_output(["bash", str(ROOT / script), *args], env=env, text=True)

    def test_web_builder_and_evaluator_share_custom_index(self):
        overrides = {"DATA_ROOT": "/tmp/synthetic dataset", "INDEX_ROOT": "/tmp/synthetic index"}
        build = self.run_script("scripts/build_dataset_indexes.sh", "web", **overrides)
        run = self.run_script("dataset_runs/run_web.sh", **overrides)
        self.assertIn("bm25=/tmp/synthetic index/bm25", build)
        self.assertIn("BM25=/tmp/synthetic index/bm25", run)
        self.assertIn("web_test_1_17_qwen3_embedding_8b_4096_full", run)
        self.assertIn("--split test", run)
        self.assertIn("--n-queries 0", run)

    def test_longseal_needs_no_index_or_retrieval_models(self):
        run = self.run_script("dataset_runs/run_longsealqa.sh")
        self.assertIn("--split all", run)
        self.assertIn("--reranker none", run)
        self.assertNotIn("BM25=", run)
        serve = self.run_script("scripts/start_model_services.sh", "policy")
        self.assertNotIn("--runner pooling", serve)

    def test_exported_environment_overrides_local_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "settings.env"
            config.write_text("N_QUERIES=999\nPARALLEL=999\n")
            run = self.run_script("dataset_runs/run_web.sh", CONFIG_ENV=str(config), N_QUERIES="3", PARALLEL="1")
            self.assertIn("--n-queries 3", run)
            self.assertIn("--parallel 1", run)

    def test_archive_excludes_research_artifacts(self):
        import tarfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "harness_search").mkdir()
            (root / "harness_search/source.py").write_text("pass\n")
            (root / "harness_search/source.py.orig").write_text("private backup")
            (root / "harness_search/._source.py").write_text("metadata")
            (root / ".env.local").write_text("TOKEN=synthetic\n")
            (root / "runtime_paths.env").write_text("PRIVATE=synthetic\n")
            (root / "data").symlink_to(root / "missing")
            output = root / "export.tar.gz"
            export(root, output)
            with tarfile.open(output) as archive:
                self.assertEqual(archive.getnames(), ["Harness-Search/harness_search/source.py"])
            (root / "harness_search/link.py").symlink_to(root / ".env.local")
            with self.assertRaises(ValueError):
                source_files(root)


class DataSetupTests(unittest.TestCase):
    def make_bundled_split(self, root, name="sec"):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = root / "queries" / name / "test.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([{
            "query_id": "synthetic-id", "query": "Question?",
            "answer": "Answer", "document_ids": ["doc42"],
        }]), path)
        metadata = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": 1}
        (root / "manifest.json").write_text(json.dumps({"files": {f"queries/{name}/test.parquet": metadata}}))
        return path

    def test_bundled_queries_install_offline_and_can_be_reused(self):
        from scripts import download_queries
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_bundled_split(root / "bundled")
            output = root / "installed"
            with patch.object(download_queries, "BUNDLED_ROOT", root / "bundled"), \
                 patch.dict(os.environ, {}, clear=True), \
                 patch.object(sys, "argv", ["download_queries.py", "sec", "--output", str(output)]), \
                 patch.object(download_queries.datasets, "load_dataset", side_effect=AssertionError("Unexpected HF access")):
                download_queries.main()
                download_queries.main()
            self.assertEqual((output / "sec/test.parquet").read_bytes(), source.read_bytes())

    def test_corrupt_bundled_queries_are_rejected_before_copying(self):
        from scripts import download_queries
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_bundled_split(root / "bundled")
            source.write_bytes(source.read_bytes() + b"corrupt")
            output = root / "installed"
            with patch.object(download_queries, "BUNDLED_ROOT", root / "bundled"), \
                 self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                download_queries.install_bundled_split("sec", "test", output)
            self.assertFalse(output.exists())

    def test_bundled_install_preserves_a_different_local_split(self):
        from scripts import download_queries
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_bundled_split(root / "bundled")
            output = root / "installed"
            output.mkdir()
            destination = output / "test.parquet"
            destination.write_bytes(b"existing local split")
            with patch.object(download_queries, "BUNDLED_ROOT", root / "bundled"), \
                 self.assertRaisesRegex(RuntimeError, "Refusing to replace"):
                download_queries.install_bundled_split("sec", "test", output)
            self.assertEqual(destination.read_bytes(), b"existing local split")

    def test_explicit_upstream_selection_bypasses_bundled_queries(self):
        from scripts import download_queries
        data = download_queries.datasets.Dataset.from_list([{
            "query_id": "synthetic-id", "query": "Question?",
            "answer": "Answer", "document_ids": ["doc42"],
        }])
        cases = [(["--upstream"], {}), (["--revision", "pinned-revision"], {}),
                 ([], {"SEC_TEST_QUERY_REPO": "synthetic/custom-sec"})]
        with tempfile.TemporaryDirectory() as temporary:
            for flags, environment in cases:
                with self.subTest(flags=flags, environment=environment), \
                     patch.dict(os.environ, environment, clear=True), \
                     patch.object(sys, "argv", ["download_queries.py", "sec", "--output", temporary, *flags]), \
                     patch.object(download_queries, "EXPECTED_ROWS", {"sec": {"test": 1}}), \
                     patch.object(download_queries, "install_bundled_split", side_effect=AssertionError("Unexpected bundled selection")), \
                     patch.object(download_queries.datasets, "load_dataset", return_value={"test": data}) as load:
                    download_queries.main()
                    self.assertEqual(load.call_args.args[0], environment.get("SEC_TEST_QUERY_REPO", download_queries.SOURCES["sec"]["test"]))
                    self.assertEqual(load.call_args.kwargs["revision"], flags[1] if flags[:1] == ["--revision"] else "main")

    def test_browsecomp_conversion_round_trip(self):
        from scripts.download_queries import BROWSECOMP_CANARY, write_browsecomp
        from datagen.search_dataset import BrowseCompPlusDataset
        key = hashlib.sha256(BROWSECOMP_CANARY.encode()).digest()
        def encrypted(value):
            return base64.b64encode(bytes(v ^ key[i % len(key)] for i, v in enumerate(value.encode()))).decode()
        row = {"query_id": "synthetic-id", "query": encrypted('A\tquestion?'), "answer": encrypted("A synthetic answer"),
               "gold_docs": [{"docid": encrypted("doc42")}], "evidence_docs": [{"docid": encrypted("doc43")}]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(write_browsecomp([row], root), 1)
            self.assertEqual(BrowseCompPlusDataset._load_queries(root / "queries.tsv"), {"synthetic-id": "A\tquestion?"})
            self.assertEqual(BrowseCompPlusDataset._load_qrels(root / "qrels_gold.txt"), {"synthetic-id": {"doc42": 1}})
            self.assertEqual(BrowseCompPlusDataset._load_decrypted_answers(root / "answers.jsonl"), {"synthetic-id": "A synthetic answer"})

    def test_web_test_evaluation_does_not_load_train(self):
        from datagen.search_dataset import get_dataset
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "web"
            folder.mkdir()
            (folder / "test.jsonl").write_text(json.dumps({"query_id": "synthetic-id", "query": "Question?", "answer": "Answer", "document_ids": ["doc42"]}) + "\n")
            with patch.dict(os.environ, {"HARNESS_SEARCH_QUERY_DATA_ROOT": temporary}), \
                 patch("datagen.search_dataset.load_hf_dataset_first_available", side_effect=AssertionError("Unexpected HF access")):
                dataset = get_dataset("web", split="test")
            self.assertEqual(dataset.get_test_query_ids(), ["synthetic-id"])
            self.assertEqual(dataset.get_train_query_ids(), [])
            self.assertEqual(dataset._get_final_answer_document_ids("synthetic-id"), {"doc42"})

    def test_config_does_not_require_unrelated_api_keys(self):
        from harness_search.config import Config
        with patch.dict(os.environ, {}, clear=True):
            config = Config(_env_file=None)
            self.assertEqual(config.openai_api_key.get_secret_value(), "EMPTY")
            self.assertEqual(config.tinker_api_key.get_secret_value(), "")


if __name__ == "__main__":
    unittest.main()
