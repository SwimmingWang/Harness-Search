from abc import ABC, abstractmethod
import ast
from collections import defaultdict
from enum import Enum
import os
from pathlib import Path
from typing import List, Literal, Optional, Set, Tuple
import datasets
import csv
import json
import random
from urllib.parse import urlsplit, urlunsplit
import harness_search.config as config
from harness_search.tasks import chunk_ids_to_doc_ids


SPLIT_SEED = 42
TRAIN_RATIO = 0.8
REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFER_DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data")) / "transfer"

# Within the train split, further divide into SFT and RL subsets
SFT_RL_SPLIT_SEED = 123  # Different seed from train/test split for independence
SFT_RATIO = 0.3  # 30% of train queries for SFT, 70% for RL

# Type alias for fact-level document structure
FactItem = dict  # {"fact": str, "chunk_ids": List[str], "is_final_answer": bool}


def normalize_document_id(document_id: str) -> str:
    """Normalize a document ID for evaluation.

    For URL-like IDs, strip the fragment to avoid mismatches between equivalent
    links such as ``/wiki/Foo`` and ``/wiki/Foo#section``.
    """
    if "://" not in document_id:
        return document_id

    parsed = urlsplit(document_id)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def load_local_dataset_split(dataset_name: str, split_name: str):
    """Load a downloaded query split before falling back to Hugging Face."""
    root = os.environ.get(
        "HARNESS_SEARCH_QUERY_DATA_ROOT",
        os.environ.get("INTENT_CURATOR_QUERY_DATA_ROOT", str(Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data")) / "queries")),
    )
    if not root:
        return None
    base = Path(root) / dataset_name
    candidates = [
        base / f"{split_name}.parquet",
        base / f"{split_name}.jsonl",
        base / f"{split_name}.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        loader = "parquet" if path.suffix == ".parquet" else "json"
        loaded = datasets.load_dataset(loader, data_files=str(path), split="train")
        required = {"query_id", "query", "document_ids", "answer"}
        missing = required - set(loaded.column_names)
        if missing:
            raise ValueError(f"{path} is missing query columns: {sorted(missing)}")
        if dataset_name == "web" and "gold_document_ids" not in loaded.column_names:
            loaded = loaded.add_column("gold_document_ids", loaded["document_ids"])
        return loaded
    return None


def load_hf_dataset_first_available(
    hf_path: str,
    *,
    split_preferences: Tuple[str, ...] = ("test", "train", "validation"),
) -> datasets.Dataset:
    """Load a HuggingFace dataset and pick the first available preferred split."""
    cfg = config.get_config()
    token = (cfg.huggingface_token.get_secret_value() or None)
    raw = datasets.load_dataset(hf_path, token=token)

    for split_name in split_preferences:
        if split_name in raw and len(raw[split_name]) > 0:
            return raw[split_name]

    # Fallback to first non-empty split, then first split if all are empty.
    for split_name in raw.keys():
        if len(raw[split_name]) > 0:
            return raw[split_name]

    first_split = next(iter(raw.keys()))
    return raw[first_split]


def _coerce_string_list(value) -> List[str]:
    """Accept Arrow lists and JSON/Python-list strings from dataset previews."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
    return [str(item) for item in value if item]


# ============================================================================
# Backward-compatible enum (used by existing callers)
# ============================================================================


class SearchDatasetName(Enum):
    """Dataset names supported by the standalone evaluator."""
    BROWSECOMPPLUS = "browsecompplus"
    BC_PLUS = "bc_plus"
    WEB = "web"
    SEC = "sec"
    LONGSEALQA = "longsealqa"
    TWOWIKIMULTIHOPQA = "2wikimultihopqa"
    HOTPOTQA = "hotpotqa"
    MUSIQUE = "musique"


class SearchDataset(ABC):
    """
    Abstract base class for search datasets.

    A search dataset is a dataset of search queries and the documents that are required
    to answer the query or that are relevant to the query.

    Subclasses must implement `_load_dataset()` to populate `_search_queries_dataset`
    with a HuggingFace Dataset containing the following columns:
    - query_id: The query id
    - query: The search query
    - document_ids: The documents that are required to answer the query or that are relevant to the query.
                    For document-level evaluation: List[str] of document/chunk IDs.
                    For fact-level evaluation: List[FactItem] where each FactItem has
                    {"fact": str, "chunk_ids": List[str], "is_final_answer": bool}.
    - answer: The answer to the query

    Subclasses can override `evaluation_mode` property to change evaluation behavior:
    - "document": Standard document/chunk-level evaluation (default)
    - "fact": Fact-level evaluation where a fact is found if ANY of its chunk_ids are retrieved

    For final_answer_recall evaluation:
    - Document-level datasets can override `_get_final_answer_document_ids()` to specify
      which document IDs are "final answer" documents (e.g., gold vs evidence in BrowseCompPlus).
    - Fact-level datasets automatically use facts where is_final_answer=True.
    """

    _search_queries_dataset: datasets.Dataset
    _query_index: dict  # Maps query_id -> row dict for O(1) lookups
    _train_query_ids: List[str]  # Query IDs in the train split
    _test_query_ids: List[str]  # Query IDs in the test split

    # Chroma collection configuration - override in subclasses
    # Can be a single collection name or a list for load balancing
    CHROMA_COLLECTIONS: List[str] = []
    # Optional split-specific collections (if not set, falls back to CHROMA_COLLECTIONS)
    CHROMA_COLLECTIONS_TRAIN: Optional[List[str]] = None
    CHROMA_COLLECTIONS_TEST: Optional[List[str]] = None

    def __init__(self) -> None:
        # Subclass loads dataset into self._search_queries_dataset
        self._load_dataset()

        # Build common indices
        self._build_query_index()
        self._create_train_test_split()

    @abstractmethod
    def _load_dataset(self) -> None:
        """Load the dataset into self._search_queries_dataset. Implemented by subclasses."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name identifier for this dataset."""
        pass

    @property
    def evaluation_mode(self) -> Literal["document", "fact"]:
        """Return the evaluation mode for this dataset.

        - "document": Standard document/chunk-level evaluation. document_ids is List[str].
        - "fact": Fact-level evaluation. document_ids is List[FactItem] where each fact
                  has chunk_ids. A fact counts as found if ANY of its chunk_ids are retrieved.

        Override this in subclasses that use fact-level evaluation.
        """
        return "document"

    def get_chroma_collections(
        self, split: Optional[Literal["train", "test"]] = None
    ) -> List[str]:
        """Get the Chroma collection names that back this dataset.

        Args:
            split: If provided, return collections specific to that split.
                   If None, returns the default collections.

        Returns:
            A list of Chroma collection names. Multiple collections can be used
            for load balancing (one is randomly selected per request).

        Raises:
            ValueError: If no collections are configured for the requested split.
        """
        if split == "train" and self.CHROMA_COLLECTIONS_TRAIN is not None:
            collections = self.CHROMA_COLLECTIONS_TRAIN
        elif split == "test" and self.CHROMA_COLLECTIONS_TEST is not None:
            collections = self.CHROMA_COLLECTIONS_TEST
        else:
            collections = self.CHROMA_COLLECTIONS

        if not collections:
            raise ValueError(
                f"No Chroma collections configured for dataset '{self.name}'"
                + (f" (split={split})" if split else "")
            )
        return collections

    def _build_query_index(self) -> None:
        """Build query index for O(1) lookups instead of O(n) filter operations."""
        self._query_index = {}
        for i in range(len(self._search_queries_dataset)):
            row = self._search_queries_dataset[i]
            # Handle document_ids that may be stored as string instead of list
            # TODO: We should fix this in the dataset itself.
            document_ids = row["document_ids"]
            if isinstance(document_ids, str):
                document_ids = ast.literal_eval(document_ids)
            # For document-level evaluation, ensure document_ids are strings
            # (model outputs are strings, so we need consistent types for comparison)
            if self.evaluation_mode == "document":
                document_ids = [
                    normalize_document_id(str(doc_id)) for doc_id in document_ids
                ]
            # Ensure query_id is always a string
            query_id = str(row["query_id"])
            self._query_index[query_id] = {
                "query_id": query_id,
                "query": row["query"],
                "document_ids": document_ids,
                "answer": row["answer"],
            }

    def _create_train_test_split(self) -> None:
        """Create deterministic train/test split (80/20)."""
        all_query_ids = list(self._query_index.keys())
        all_query_ids_sorted = sorted(all_query_ids)  # Sort for determinism
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(all_query_ids_sorted)
        split_idx = int(len(all_query_ids_sorted) * TRAIN_RATIO)
        self._train_query_ids = all_query_ids_sorted[:split_idx]
        self._test_query_ids = all_query_ids_sorted[split_idx:]

    def get_train_query_ids(self) -> List[str]:
        """Return all query ids in the train split (80% of data)."""
        return self._train_query_ids.copy()

    def get_test_query_ids(self) -> List[str]:
        """Return all query ids in the test split (20% of data)."""
        return self._test_query_ids.copy()

    def _create_sft_rl_split(self) -> None:
        """Split train queries into SFT (30%) and RL (70%) subsets.

        This is a deterministic sub-split of the train set. The split is
        performed after the train/test split, so it's independent of it.
        """
        train_ids_sorted = sorted(self._train_query_ids)  # Sort for determinism
        rng = random.Random(SFT_RL_SPLIT_SEED)
        rng.shuffle(train_ids_sorted)
        split_idx = int(len(train_ids_sorted) * SFT_RATIO)
        self._sft_query_ids = train_ids_sorted[:split_idx]
        self._rl_query_ids = train_ids_sorted[split_idx:]

    def get_sft_query_ids(self) -> List[str]:
        """Return query ids for SFT training (30% of train split)."""
        if not hasattr(self, "_sft_query_ids"):
            self._create_sft_rl_split()
        return self._sft_query_ids.copy()

    def get_rl_query_ids(self) -> List[str]:
        """Return query ids for RL training (70% of train split)."""
        if not hasattr(self, "_rl_query_ids"):
            self._create_sft_rl_split()
        return self._rl_query_ids.copy()

    def get_random_query(
        self, split: Optional[Literal["train", "test"]] = None
    ) -> Tuple[str, str]:
        """Get a random query from the search queries dataset.

        Args:
            split: If provided, only sample from the specified split ("train" or "test").
                   If None, sample from all queries.

        Returns the query id and query text.
        """
        if split == "train":
            query_ids = self._train_query_ids
        elif split == "test":
            query_ids = self._test_query_ids
        else:
            query_ids = list(self._query_index.keys())

        query_id = random.choice(query_ids)
        return (query_id, self._query_index[query_id]["query"])

    def get_all_query_ids(
        self, split: Optional[Literal["train", "test", "sft", "rl"]] = None
    ) -> List[str]:
        """Return all query ids contained in the dataset.

        Args:
            split: If provided, only return query ids from the specified split.
                   - "train": All train queries (80% of data)
                   - "test": All test queries (20% of data)
                   - "sft": SFT subset of train queries (30% of train = 24% of total)
                   - "rl": RL subset of train queries (70% of train = 56% of total)
                   - None: All query ids
        """
        if split == "train":
            return self._train_query_ids.copy()
        elif split == "test":
            return self._test_query_ids.copy()
        elif split == "sft":
            return self.get_sft_query_ids()
        elif split == "rl":
            return self.get_rl_query_ids()
        return list(self._query_index.keys())

    def get_expected_document_ids(self, query_id: str) -> List[str]:
        """Get the expected document/chunk ids for a given query id.

        For document-level datasets: returns the document_ids list directly.
        For fact-level datasets: returns a flattened list of all chunk_ids from all facts.

        Returns a list of document/chunk IDs.
        """
        return list(self._get_all_relevant_chunk_ids(query_id))

    def get_expected_facts(self, query_id: str) -> List[FactItem]:
        """Get the expected facts for a given query id.

        Only meaningful for fact-level datasets (evaluation_mode == "fact").
        For document-level datasets, this returns an empty list.

        Returns a list of fact objects, each with keys:
        - "fact": str - description of the fact
        - "chunk_ids": List[str] - chunk IDs containing this fact
        - "is_final_answer": bool - whether this fact is the final answer
        """
        if self.evaluation_mode != "fact":
            raise ValueError(f"Dataset {self.name} is not a fact-level dataset")
        return self._query_index[query_id]["document_ids"]

    def get_expected_answer(self, query_id: str) -> str:
        """Get the expected answer for a given query id.

        Returns the expected answer.
        """
        return self._query_index[query_id]["answer"]

    def get_query_by_id(self, query_id: str) -> Tuple[str, str]:
        """Get a query by id from the search queries dataset.

        Returns the query id and query text.
        """
        row = self._query_index[query_id]
        return (row["query_id"], row["query"])

    def _get_all_relevant_chunk_ids(self, query_id: str) -> Set[str]:
        """Get all relevant chunk IDs for a query, handling both evaluation modes.

        For document-level: returns document_ids directly.
        For fact-level: extracts and flattens all chunk_ids from fact objects.
        """
        document_ids = self._query_index[query_id]["document_ids"]

        if self.evaluation_mode == "fact":
            # Fact-level: extract chunk_ids from each fact object
            all_chunk_ids: Set[str] = set()
            for fact in document_ids:
                all_chunk_ids.update(fact["chunk_ids"])
            return all_chunk_ids
        else:
            # Document-level: document_ids is already a flat list
            return set(document_ids)

    def _get_final_answer_document_ids(self, query_id: str) -> Set[str]:
        """Get document IDs that correspond to "final answer" documents.

        For document-level datasets: By default, returns all document_ids.
        Subclasses can override this to return only "gold" or "final answer" documents.

        For fact-level datasets: Returns chunk_ids from facts where is_final_answer=True.
        """
        document_ids = self._query_index[query_id]["document_ids"]

        if self.evaluation_mode == "fact":
            # Fact-level: extract chunk_ids only from final answer facts
            final_answer_chunk_ids: Set[str] = set()
            for fact in document_ids:
                if fact.get("is_final_answer", False):
                    final_answer_chunk_ids.update(fact["chunk_ids"])
            return final_answer_chunk_ids
        else:
            # Document-level: by default, all documents are considered "final answer"
            # Subclasses can override to provide gold-only documents
            return set(document_ids)

    def _get_final_answer_facts(self, query_id: str) -> List[FactItem]:
        """Get facts that are marked as final answer.

        Only meaningful for fact-level datasets.
        Returns facts where is_final_answer=True.
        """
        if self.evaluation_mode != "fact":
            return []
        document_ids = self._query_index[query_id]["document_ids"]
        return [fact for fact in document_ids if fact.get("is_final_answer", False)]

    def evaluate_results_recall(
        self, query_id: str, retrieved_chunk_ids: List[str]
    ) -> float:
        """Evaluate the recall of the retrieved chunk ids for a given query.

        For document-level evaluation:
            Recall = True Positives / (True Positives + False Negatives)
            where positives are document IDs.

        For fact-level evaluation:
            Recall = (facts found) / (total facts)
            A fact is considered found if ANY of its chunk_ids are in the retrieved set.
        """
        retrieved_set = set(retrieved_chunk_ids)

        if self.evaluation_mode == "fact":
            # Fact-level recall: count facts where at least one chunk_id is retrieved
            facts = self._query_index[query_id]["document_ids"]
            if len(facts) == 0:
                return 0.0

            facts_found = sum(
                1
                for fact in facts
                if set(fact["chunk_ids"]).intersection(retrieved_set)
            )
            return facts_found / len(facts)
        else:
            # Document-level recall
            retrieved_document_ids_set: Set[str] = chunk_ids_to_doc_ids(retrieved_set)
            relevant_document_ids_set: Set[str] = set(
                self._query_index[query_id]["document_ids"]
            )

            true_positives = len(
                retrieved_document_ids_set.intersection(relevant_document_ids_set)
            )
            false_negatives = len(
                relevant_document_ids_set - retrieved_document_ids_set
            )
            if true_positives + false_negatives == 0:
                return 0.0
            return true_positives / (true_positives + false_negatives)

    def evaluate_results_final_answer_recall(
        self, query_id: str, retrieved_chunk_ids: List[str]
    ) -> float:
        """Evaluate the final answer recall of the retrieved chunk ids for a given query.

        This metric measures recall specifically on "final answer" or "gold" documents/facts:

        For document-level evaluation (e.g., BrowseCompPlus):
            Uses _get_final_answer_document_ids() which can be overridden by subclasses
            to return only "gold" documents (excluding "evidence" documents).
            Recall = (gold docs found) / (total gold docs)

        For fact-level evaluation:
            Only considers facts where is_final_answer=True.
            Recall = (final answer facts found) / (total final answer facts)
            A fact is found if ANY of its chunk_ids are in the retrieved set.
        """
        retrieved_set = set(retrieved_chunk_ids)

        if self.evaluation_mode == "fact":
            # Fact-level: only count final answer facts
            final_answer_facts = self._get_final_answer_facts(query_id)
            if len(final_answer_facts) == 0:
                return 0.0

            facts_found = sum(
                1
                for fact in final_answer_facts
                if set(fact["chunk_ids"]).intersection(retrieved_set)
            )
            return facts_found / len(final_answer_facts)
        else:
            # Document-level: use final answer document IDs
            retrieved_document_ids_set: Set[str] = chunk_ids_to_doc_ids(retrieved_set)
            final_answer_document_ids_set: Set[str] = (
                self._get_final_answer_document_ids(query_id)
            )

            if len(final_answer_document_ids_set) == 0:
                return 0.0

            true_positives = len(
                retrieved_document_ids_set.intersection(final_answer_document_ids_set)
            )
            return true_positives / len(final_answer_document_ids_set)

    def evaluate_results_precision(
        self, query_id: str, retrieved_chunk_ids: List[str]
    ) -> float:
        """Evaluate the precision of the retrieved chunk ids for a given query.

        For document-level evaluation:
            Precision = True Positives / (True Positives + False Positives)
            where positives are document IDs.

        For fact-level evaluation:
            Precision = (relevant chunks retrieved) / (total chunks retrieved)
            A chunk is relevant if it appears in any fact's chunk_ids.
        """
        retrieved_set = set(retrieved_chunk_ids)

        if self.evaluation_mode == "fact":
            # Fact-level precision: what fraction of retrieved chunks are relevant
            if len(retrieved_set) == 0:
                return 0.0

            all_relevant_chunk_ids = self._get_all_relevant_chunk_ids(query_id)
            relevant_retrieved = len(retrieved_set.intersection(all_relevant_chunk_ids))
            return relevant_retrieved / len(retrieved_set)
        else:
            # Document-level precision
            retrieved_document_ids_set: Set[str] = chunk_ids_to_doc_ids(retrieved_set)
            relevant_document_ids_set: Set[str] = set(
                self._query_index[query_id]["document_ids"]
            )

            true_positives = len(
                retrieved_document_ids_set.intersection(relevant_document_ids_set)
            )
            false_positives = len(
                retrieved_document_ids_set - relevant_document_ids_set
            )
            if true_positives + false_positives == 0:
                return 0.0
            return true_positives / (true_positives + false_positives)

    def evaluate_results_f1_score(
        self, query_id: str, retrieved_chunk_ids: List[str]
    ) -> float:
        """Evaluate the F1 score of the retrieved chunk ids for a given query.

        F1 score is defined as 2 * (Precision * Recall) / (Precision + Recall)
        Works for both document-level and fact-level evaluation modes.
        """
        precision = self.evaluate_results_precision(query_id, retrieved_chunk_ids)
        recall = self.evaluate_results_recall(query_id, retrieved_chunk_ids)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @classmethod
    def from_known_dataset(cls, name: "SearchDatasetName") -> "SearchDataset":
        """Backward-compatible factory method. Prefer get_dataset(name_str) instead."""
        return get_dataset(name.value)


# ============================================================================
# Pre-Split Dataset Base Class
# ============================================================================


class PreSplitSearchDataset(SearchDataset):
    """Dataset with canonical, separately released train and test splits."""

    HF_PATH_TRAIN: str
    HF_PATH_TEST: str
    HF_SPLIT_TRAIN: str = "train"
    HF_SPLIT_TEST: str = "test"

    def __init__(self, requested_split: str | None = None) -> None:
        self.requested_split = requested_split
        super().__init__()

    def _load_dataset(self) -> None:
        # Evaluation must not require access to an unrelated training release.
        names = ("test",) if self.requested_split == "test" else (
            ("train",) if self.requested_split in {"train", "rl", "sft"} else ("train", "test")
        )
        loaded = {}
        for name in names:
            split = load_local_dataset_split(self.name, name)
            if split is None:
                repo = self.HF_PATH_TEST if name == "test" else self.HF_PATH_TRAIN
                split = load_hf_dataset_first_available(repo, split_preferences=(name, "train", "test"))
            if self.name == "web" and "gold_document_ids" not in split.column_names:
                split = split.add_column("gold_document_ids", split["document_ids"])
            loaded[name] = split
        self._presplit_train_ids = [str(q) for q in loaded["train"]["query_id"]] if "train" in loaded else []
        self._presplit_test_ids = [str(q) for q in loaded["test"]["query_id"]] if "test" in loaded else []
        self._search_queries_dataset = datasets.concatenate_datasets(list(loaded.values()))
        self._post_load_setup()

    def _post_load_setup(self) -> None:
        pass

    def _create_train_test_split(self) -> None:
        self._train_query_ids = self._presplit_train_ids
        self._test_query_ids = self._presplit_test_ids


class SingleSplitSearchDataset(SearchDataset):
    """Dataset helper for eval-only corpora that expose a single HF split.

    For these datasets we typically want deterministic sampling from the full set,
    so we expose all query IDs through both train and test partitions.
    """

    HF_PATH: str
    HF_SPLIT_PREFERENCES: Tuple[str, ...] = ("test", "train", "validation")

    def _load_dataset(self) -> None:
        self._search_queries_dataset = load_hf_dataset_first_available(
            self.HF_PATH, split_preferences=self.HF_SPLIT_PREFERENCES
        )

    def _create_train_test_split(self) -> None:
        all_query_ids = sorted(self._query_index.keys())
        self._train_query_ids = all_query_ids
        self._test_query_ids = all_query_ids


# ============================================================================
# BrowseComp+ Dataset
# ============================================================================


class BrowseCompPlusDataset(SearchDataset):
    """BrowseComp+ search dataset."""

    _gold_document_ids: dict[str, Set[str]]  # Maps query_id -> gold document IDs

    # Multiple replicas for load balancing
    CHROMA_COLLECTIONS = [f"browsecompplus_openai_11_replica_{i}" for i in range(1, 45)]

    @property
    def name(self) -> str:
        return "browsecompplus"

    def _load_dataset(self) -> None:
        cfg = config.get_config()

        qrels_gold = self._load_qrels(cfg.browsecompplus_qrels_gold_path)
        qrels_evidence = self._load_qrels(cfg.browsecompplus_qrels_evidence_path)

        # Store gold document IDs separately for final_answer_recall
        self._gold_document_ids = {
            query_id: set(doc_ids) for query_id, doc_ids in qrels_gold.items()
        }

        # Combine qrels_gold and qrels_evidence for overall recall
        qrels: dict[str, list] = defaultdict(list)
        for query_id, doc_ids in qrels_gold.items():
            qrels[query_id].extend(doc_ids)
        for query_id, doc_ids in qrels_evidence.items():
            qrels[query_id].extend(doc_ids)

        queries = self._load_queries(cfg.browsecompplus_queries_path)
        answers = self._load_decrypted_answers(cfg.browsecompplus_answers_path)

        query_ids = list(queries.keys())
        self._search_queries_dataset = datasets.Dataset.from_dict(
            {
                "query_id": query_ids,
                "query": [queries[query_id] for query_id in query_ids],
                "document_ids": [qrels[query_id] for query_id in query_ids],
                "answer": [answers[query_id] for query_id in query_ids],
            }
        )

    def _get_final_answer_document_ids(self, query_id: str) -> Set[str]:
        """Return only gold document IDs (excluding evidence documents)."""
        return self._gold_document_ids.get(query_id, set())

    @staticmethod
    def _load_qrels(path: str) -> dict:
        """Load qrels from a TREC-format file."""
        qrels: dict[str, dict[str, int]] = {}
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                query_id = parts[0]
                doc_id = parts[2]
                relevance = int(parts[3])
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = relevance
        return qrels

    @staticmethod
    def _load_queries(path: str) -> dict:
        """Load queries from a TSV file."""
        queries = {}
        with open(path) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                query_id = row[0]
                query_text = row[1]
                queries[query_id] = query_text
        return queries

    @staticmethod
    def _load_decrypted_answers(path: str) -> dict:
        """Load decrypted answers from a JSONL file."""
        answers = {}
        with open(path, "r") as f:
            for line in f:
                doc = json.loads(line)
                answers[doc["query_id"]] = doc["answer"]
        return answers


# ============================================================================
# Other Datasets
# ============================================================================


# ============================================================================
class WebDataset(PreSplitSearchDataset):
    """Released Web retrieval benchmark with document-level labels."""

    HF_PATH_TRAIN = "kellyhongg/1_17_web_train"
    HF_PATH_TEST = "kellyhongg/web_1_17_test"
    CHROMA_COLLECTIONS_TRAIN = ["web_qwen3_embedding_8b_4096_local"]
    CHROMA_COLLECTIONS_TEST = ["web_qwen3_embedding_8b_4096_local"]

    @property
    def name(self) -> str:
        return "web"

    @property
    def evaluation_mode(self) -> Literal["document", "fact"]:
        return "document"

    def _post_load_setup(self) -> None:
        values = self._search_queries_dataset["gold_document_ids"]
        gold_ids = [ast.literal_eval(x) if isinstance(x, str) else x for x in values]
        self._gold_document_ids = {
            str(qid): {normalize_document_id(str(doc_id)) for doc_id in doc_ids}
            for qid, doc_ids in zip(self._search_queries_dataset["query_id"], gold_ids)
        }

    def _get_final_answer_document_ids(self, query_id: str) -> Set[str]:
        return self._gold_document_ids.get(query_id, set())


class SECDataset(PreSplitSearchDataset):
    """Released SEC filings benchmark with fact-level evidence labels."""

    HF_PATH_TRAIN = "kellyhongg/1_18_sec_train"
    HF_PATH_TEST = "kellyhongg/sec_test_new"
    CHROMA_COLLECTIONS_TRAIN = ["sec_qwen3_embedding_8b_4096_local"]
    CHROMA_COLLECTIONS_TEST = ["sec_qwen3_embedding_8b_4096_local"]

    @property
    def name(self) -> str:
        return "sec"

    @property
    def evaluation_mode(self) -> Literal["document", "fact"]:
        return "fact"


class LongSealQADataset(SingleSplitSearchDataset):
    """LongSeal using the official per-query 30-document corpus."""

    HF_PATH = "vtllms/sealqa"

    def _load_dataset(self) -> None:
        local_path = TRANSFER_DATA_ROOT / "longseal_test.parquet"
        if local_path.exists():
            raw = datasets.load_dataset("parquet", data_files=str(local_path), split="train")
        else:
            cfg = config.get_config()
            raw = datasets.load_dataset(
                self.HF_PATH, "longseal", split="test", token=(cfg.huggingface_token.get_secret_value() or None)
            )
        rows = []
        self._documents_by_query = {}
        for i, row in enumerate(raw):
            qid = str(i)
            urls = [normalize_document_id(x) for x in _coerce_string_list(row["urls"])]
            gold_documents = row["golds"]
            distractor_documents = row["30_docs"]
            if isinstance(gold_documents, str):
                gold_documents = ast.literal_eval(gold_documents)
            if isinstance(distractor_documents, str):
                distractor_documents = ast.literal_eval(distractor_documents)
            # LongSeal stores answer-bearing pages separately from the long-context
            # distractor bundle. Both belong to the retrievable corpus; their order
            # must not reveal which documents are gold.
            documents = list(gold_documents or []) + list(distractor_documents or [])
            random.Random(f"longseal-{qid}").shuffle(documents)
            cleaned = []
            seen_urls = set()
            for j, document in enumerate(documents or []):
                doc = dict(document)
                doc["url"] = normalize_document_id(
                    str(doc.get("url") or f"longseal://{qid}/{j}")
                )
                if doc["url"] in seen_urls:
                    continue
                seen_urls.add(doc["url"])
                cleaned.append(doc)
            self._documents_by_query[qid] = cleaned
            rows.append({
                "query_id": qid,
                "query": row["question"],
                "document_ids": urls,
                "answer": row["answer"],
            })
        self._search_queries_dataset = datasets.Dataset.from_list(rows)

    def get_documents(self, query_id: str):
        return self._documents_by_query[str(query_id)]

    @property
    def name(self) -> str:
        return "longsealqa"


# Additional Benchmark Datasets (Kelly April 2026 refresh)
# ============================================================================


class BCPlusDataset(SingleSplitSearchDataset):
    """BrowseComp+ benchmark loaded directly from HuggingFace."""

    HF_PATH = "kellyhongg/bc_plus"
    HF_SPLIT_PREFERENCES = ("test", "train")
    CHROMA_COLLECTIONS = ["browsecompplus_openai_11_replica_1"]

    _gold_document_ids: dict[str, Set[str]]

    @property
    def name(self) -> str:
        return "bc_plus"

    def _load_dataset(self) -> None:
        self._search_queries_dataset = load_hf_dataset_first_available(
            self.HF_PATH, split_preferences=self.HF_SPLIT_PREFERENCES
        )

        gold_document_ids = [
            ast.literal_eval(docids) if isinstance(docids, str) else docids
            for docids in self._search_queries_dataset["gold_document_ids"]
        ]
        self._gold_document_ids = {
            str(query_id): {
                normalize_document_id(str(doc_id)) for doc_id in doc_ids
            }
            for query_id, doc_ids in zip(
                self._search_queries_dataset["query_id"], gold_document_ids
            )
        }

    def _get_final_answer_document_ids(self, query_id: str) -> Set[str]:
        return self._gold_document_ids.get(query_id, set())


# ============================================================================
# SEC Filings Dataset (legacy - uses HuggingFace kellyhongg/sec_filings)
# ============================================================================


# ============================================================================
# QA-Only Benchmark Datasets (no document_ids, answer-evaluation only)
# ============================================================================


class LocalMultiHopQADataset(SingleSplitSearchDataset):
    """Local QA benchmark evaluated by answer EM/F1 over shared Wiki18."""

    DATASET_NAME = ""
    CHROMA_COLLECTIONS = ["wiki18_100w_qwen3_embedding_8b_4096"]

    @property
    def name(self) -> str:
        return self.DATASET_NAME

    @property
    def is_answer_evaluation(self) -> bool:
        return True

    def _load_dataset(self) -> None:
        path = REPO_ROOT / "data" / self.DATASET_NAME / "dev.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing QA dataset: {path}")
        rows = []
        self._golden_answers = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                raw = json.loads(line)
                answers = _coerce_string_list(raw.get("golden_answers"))
                if not answers:
                    raise ValueError(f"{path}:{line_number} has no golden_answers")
                self._golden_answers[str(raw["id"])] = answers
                rows.append({"query_id": str(raw["id"]),
                             "query": str(raw["question"]),
                             "document_ids": [], "answer": answers[0],
                             "golden_answers": answers})
        self._search_queries_dataset = datasets.Dataset.from_list(rows)

    def get_golden_answers(self, query_id: str) -> List[str]:
        return list(self._golden_answers[query_id])


class TwoWikiMultihopQADataset(LocalMultiHopQADataset):
    DATASET_NAME = "2wikimultihopqa"


class HotpotQADataset(LocalMultiHopQADataset):
    DATASET_NAME = "hotpotqa"


class MusiqueDataset(LocalMultiHopQADataset):
    DATASET_NAME = "musique"


# ============================================================================
# Dataset Registry & Factory
# ============================================================================


DATASET_REGISTRY: dict[str, type[SearchDataset]] = {
    "browsecompplus": BrowseCompPlusDataset,
    "bc_plus": BCPlusDataset,
    "web": WebDataset,
    "sec": SECDataset,
    "longsealqa": LongSealQADataset,
    "2wikimultihopqa": TwoWikiMultihopQADataset,
    "hotpotqa": HotpotQADataset,
    "musique": MusiqueDataset,
}


def get_dataset(name: str, split: str | None = None) -> SearchDataset:
    """Create a configured dataset by name."""
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {available}")
    cls = DATASET_REGISTRY[name]
    return cls(requested_split=split) if issubclass(cls, PreSplitSearchDataset) else cls()
