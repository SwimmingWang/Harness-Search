"""Exercise a parallel-build interruption against a real tiny local Qdrant DB."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qdrant_client import QdrantClient, models
from scripts import build_dataset_qdrant as builder


class TinyEncoding:
    def encode(self, text, **kwargs):
        return list(text.encode())


class IndexResumeTests(unittest.TestCase):
    def test_partial_upload_holes_are_repaired_using_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            corpus.write_text("".join(json.dumps({"id": f"d{i}", "contents": f"text {i}"}) + "\n" for i in range(6)))
            argv = ["build", "--corpus", str(corpus), "--qdrant", str(root / "db"),
                    "--cache", str(root / "cache"), "--collection", "synthetic",
                    "--dimensions", "2", "--max-items", "2", "--concurrency", "2"]
            def embedding(texts, *args):
                return [[float(text.rsplit(' ', 1)[-1]), 1.0] for text in texts]
            with patch("sys.argv", argv), patch.object(builder.tiktoken, "get_encoding", return_value=TinyEncoding()), \
                 patch.object(builder, "embed", side_effect=embedding) as embed:
                builder.main()
                self.assertEqual(embed.call_count, 3)
                # Simulate a partial parallel upload: missing earlier IDs, later IDs present.
                client = QdrantClient(path=str(root / "db"))
                client.delete("synthetic", models.PointIdsList(points=[0, 1]), wait=True)
                self.assertEqual(client.count("synthetic").count, 4)
                client.close()
                embed.reset_mock()
                builder.main()
                embed.assert_not_called()
                client = QdrantClient(path=str(root / "db"))
                records = client.retrieve("synthetic", ids=list(range(6)), with_payload=True)
                self.assertEqual({r.id: r.payload["chunk_id"] for r in records}, {i: f"d{i}" for i in range(6)})
                client.close()


if __name__ == "__main__":
    unittest.main()
