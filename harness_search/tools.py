"""

Tool definitions, implementations, and provider format converters.

This module provides a composable tool system for creating SearchAgent
instances with different tool configurations for research and experimentation.

"""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    TYPE_CHECKING,
)
import os
import random
import sys
import threading
try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass
import chromadb
from chromadb.api.types import SearchResult
import openai
import tenacity

from chromadb.utils.embedding_functions import Bm25EmbeddingFunction
from pydantic import BaseModel, Field
from harness_search.utils import ProviderFormat
from harness_search.actions import canonical_tool_name
import json
import re
import structlog
import time
from pathlib import Path

from harness_search.rerank import Reranker

logger = structlog.get_logger("search_agent.tools")


class LocalHybridIndex:
    """Qdrant dense + bm25s sparse index with reciprocal-rank fusion."""

    def __init__(self, qdrant_url: str, collection: str, bm25_dir: str,
                 embedding_url: str, embedding_model: str, rrf_k: int = 60,
                 qdrant_path: Optional[str] = None):
        import bm25s
        import Stemmer
        from qdrant_client import QdrantClient

        self.qdrant = (
            QdrantClient(path=qdrant_path)
            if qdrant_path
            else QdrantClient(url=qdrant_url, timeout=180)
        )
        self.collection = collection
        self.bm25 = bm25s.BM25.load(bm25_dir, mmap=True, load_corpus=False)
        self.stemmer = Stemmer.Stemmer("english")
        with (Path(bm25_dir) / "chunk_ids.jsonl").open() as handle:
            self.chunk_ids = [json.loads(line) for line in handle]
        self.point_ids = {chunk_id: point_id for point_id, chunk_id in enumerate(self.chunk_ids)}
        self.embedding_client = openai.OpenAI(
            api_key="EMPTY", base_url=embedding_url, timeout=180, max_retries=2
        )
        self.embedding_model = embedding_model
        self.rrf_k = rrf_k
        self.qdrant_timeout = int(os.getenv("LOCAL_HYBRID_QDRANT_TIMEOUT", "600"))

    def _query_vector(self, query: str) -> List[float]:
        prompt = (
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            f"Query: {query}"
        )
        response = self.embedding_client.embeddings.create(
            model=self.embedding_model, input=[prompt], encoding_format="float"
        )
        return response.data[0].embedding

    def search(self, query: str, knn_limit: int, search_limit: int,
               ignore_ids: List[str]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        import bm25s

        dense = self.qdrant.query_points(
            collection_name=self.collection, query=self._query_vector(query),
            limit=knn_limit, with_payload=True, with_vectors=False,
            timeout=self.qdrant_timeout,
        ).points
        query_tokens = bm25s.tokenize(
            query, stopwords="en", stemmer=self.stemmer, show_progress=False
        )
        indices, _ = self.bm25.retrieve(query_tokens, k=knn_limit, show_progress=False)
        sparse_ids = [self.chunk_ids[int(i)] for i in indices[0]]

        scores: Dict[str, float] = {}
        payloads: Dict[str, Dict[str, Any]] = {}
        for rank, point in enumerate(dense, 1):
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id"))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            payloads[chunk_id] = payload
        for rank, chunk_id in enumerate(sparse_ids, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = [x for x, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
                  if x not in set(ignore_ids)][:search_limit]
        missing = [x for x in ranked if x not in payloads]
        if missing:
            point_ids = [self.point_ids[x] for x in missing]
            for point in self.qdrant.retrieve(
                self.collection, ids=point_ids, with_payload=True,
                timeout=self.qdrant_timeout,
            ):
                payload = point.payload or {}
                payloads[str(payload.get("chunk_id"))] = payload
        documents = [str(payloads[x].get("document", "")) for x in ranked]
        metadata = [cast(Dict[str, Any], payloads[x].get("metadata", {})) for x in ranked]
        return ranked, documents, metadata

    def read_source(self, source: str, limit: int = 300) -> Tuple[List[str], List[str]]:
        from qdrant_client import models as qmodels

        point_id = self.point_ids.get(source)
        if point_id is not None:
            result = self.qdrant.retrieve(
                collection_name=self.collection, ids=[point_id],
                with_payload=True, with_vectors=False,
                timeout=self.qdrant_timeout,
            )
            pairs = [(str((p.payload or {}).get("chunk_id")), str((p.payload or {}).get("document", ""))) for p in result]
            return [x[0] for x in pairs], [x[1] for x in pairs]

        result, _ = self.qdrant.scroll(
            collection_name=self.collection,
            scroll_filter=qmodels.Filter(must=[qmodels.FieldCondition(
                key="metadata.source", match=qmodels.MatchValue(value=source)
            )]), limit=limit, with_payload=True, with_vectors=False,
            timeout=self.qdrant_timeout,
        )
        pairs = [(str((p.payload or {}).get("chunk_id")), str((p.payload or {}).get("document", ""))) for p in result]
        return [x[0] for x in pairs], [x[1] for x in pairs]

    def grep(self, pattern: str, limit: int = 5, candidate_limit: int = 2000
             ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """Regex-filter BM25 candidates using the corpus text stored in Qdrant."""
        import bm25s

        try:
            expression = re.compile(pattern)
        except re.error:
            expression = re.compile(re.escape(pattern))
        query_tokens = bm25s.tokenize(
            pattern, stopwords="en", stemmer=self.stemmer, show_progress=False
        )
        indices, _ = self.bm25.retrieve(
            query_tokens, k=min(candidate_limit, len(self.chunk_ids)), show_progress=False
        )
        candidate_ids = [self.chunk_ids[int(i)] for i in indices[0]]
        point_ids = [self.point_ids[x] for x in candidate_ids]
        payload_by_id: Dict[str, Dict[str, Any]] = {}
        for start in range(0, len(point_ids), 256):
            points = self.qdrant.retrieve(
                self.collection, ids=point_ids[start:start + 256],
                with_payload=True, with_vectors=False,
                timeout=self.qdrant_timeout,
            )
            for point in points:
                payload = point.payload or {}
                payload_by_id[str(payload.get("chunk_id"))] = payload
        ids: List[str] = []
        documents: List[str] = []
        metadata: List[Dict[str, Any]] = []
        for chunk_id in candidate_ids:
            payload = payload_by_id.get(chunk_id, {})
            document = str(payload.get("document", ""))
            if expression.search(document):
                ids.append(chunk_id)
                documents.append(document)
                metadata.append(cast(Dict[str, Any], payload.get("metadata", {})))
                if len(ids) >= limit:
                    break
        return ids, documents, metadata


_LOCAL_HYBRID: Optional[LocalHybridIndex] = None
_LOCAL_HYBRID_LOCK = threading.Lock()


def _get_local_hybrid() -> Optional[LocalHybridIndex]:
    global _LOCAL_HYBRID
    bm25_dir = os.environ.get("LOCAL_HYBRID_BM25_DIR")
    if not bm25_dir:
        return None
    if _LOCAL_HYBRID is None:
        with _LOCAL_HYBRID_LOCK:
            if _LOCAL_HYBRID is None:
                _LOCAL_HYBRID = LocalHybridIndex(
                    qdrant_url=os.environ.get("LOCAL_HYBRID_QDRANT_URL", "http://127.0.0.1:6333"),
                    collection=os.environ["LOCAL_HYBRID_QDRANT_COLLECTION"],
                    bm25_dir=bm25_dir,
                    embedding_url=os.environ.get("LOCAL_HYBRID_EMBEDDING_URL", "http://127.0.0.1:8012/v1"),
                    embedding_model=os.environ.get("LOCAL_HYBRID_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"),
                    rrf_k=int(os.environ.get("LOCAL_HYBRID_RRF_K", "60")),
                    qdrant_path=os.environ.get("LOCAL_HYBRID_QDRANT_PATH"),
                )
    return _LOCAL_HYBRID


def _read_positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("invalid_int_env", name=name, value=raw, default=default)
        return default
    if value < 1:
        logger.warning("invalid_positive_int_env", name=name, value=raw, default=default)
        return default
    return value


CHROMA_SEARCH_MAX_CONCURRENCY = _read_positive_int_env(
    "CHROMA_SEARCH_MAX_CONCURRENCY", 8
)
_CHROMA_SEARCH_SEMAPHORE = threading.BoundedSemaphore(CHROMA_SEARCH_MAX_CONCURRENCY)


# ============================================================================
# Shared Retry Helpers
# ============================================================================


@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=15),
    before_sleep=lambda retry_state: logger.warning(
        "Retrying ChromaDB search...",
        attempt=retry_state.attempt_number,
        error=str(retry_state.outcome.exception()) if retry_state.outcome else None,
    ),
)
def _search_with_retry(
    collection: chromadb.Collection, search: chromadb.Search
) -> SearchResult:
    """Execute a ChromaDB search with retry logic for transient errors."""
    start = time.perf_counter()
    with _CHROMA_SEARCH_SEMAPHORE:
        result = collection.search(search)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if elapsed_ms > 4500:
        logger.warning(
            "Extremely slow query",
            elapsed_ms=round(elapsed_ms, 1),
            chroma_max_concurrency=CHROMA_SEARCH_MAX_CONCURRENCY,
        )
    return result


if TYPE_CHECKING:
    from harness_search.config import Config


# ============================================================================
# Tool Schema Definitions (Provider-Agnostic) & Provider Formats
# ============================================================================


class ToolSchema(BaseModel):
    """Provider-agnostic tool schema definition."""

    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = Field(default_factory=list)

    def _to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }

    # TODO: better name for this - is it legacy openai?
    def _to_qwen_moonshot_format(self) -> Dict[str, Any]:
        """Convert to Qwen function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }

    def _to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool use format."""
        # Enhance parameter descriptions for Anthropic format
        enhanced_properties = {}
        for key, value in self.parameters.items():
            enhanced_properties[key] = {
                "type": value.get("type", "string"),
                "description": value.get("description", f"The {key} parameter"),
            }

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": enhanced_properties,
                "required": self.required,
            },
        }

    def to_provider_format(self, provider: ProviderFormat) -> Dict[str, Any]:
        """Convert to the specified provider format."""
        format_map = {
            ProviderFormat.OPENAI: self._to_openai_format,
            # TODO: seperate these formats
            ProviderFormat.QWEN_MOONSHOT: self._to_qwen_moonshot_format,
            ProviderFormat.ANTHROPIC: self._to_anthropic_format,
            ProviderFormat.OPENAI_HARMONY: self._to_qwen_moonshot_format,  # The harmony format is the same as the qwen format
        }
        return format_map[provider]()


# ============================================================================
# Tool Definitions
# ============================================================================

SEARCH_SCHEMA = ToolSchema(
    name="search",
    description=(
        "Searches the corpus for relevant documents based on the input query. Returns a section of the document that is relevant to the query."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": (
                "The search query to find relevant documents in the corpus."
            ),
        }
    },
    required=["query"],
)

READ_SCHEMA = ToolSchema(
    name="read",
    description="Reads the content of a document based on its ID.",
    parameters={
        "doc_id": {
            "type": "string",
            "description": "The unique identifier of the document to read.",
        }
    },
    required=["doc_id"],
)

GREP_CORPUS_SCHEMA = ToolSchema(
    name="grep_corpus",
    description=(
        "Performs a regex search on the corpus to find documents matching the query."
    ),
    parameters={
        "pattern": {
            "type": "string",
            "description": "The regex query to search for in the corpus.",
        }
    },
    required=["pattern"],
)

MULTI_TOOL_USE_SCHEMA = ToolSchema(
    name="multi_tool_use",
    description=(
        "Allows the agent to use multiple tools in parallel to gather information."
    ),
    parameters={
        "tool_calls": {
            "type": "array",
            "description": "List of tool calls to execute in parallel.",
            "items": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "parameters": {"type": "object"},
                },
                "required": ["tool_name", "parameters"],
            },
        }
    },
    required=["tool_calls"],
)

PRUNE_CHUNKS_SCHEMA = ToolSchema(
    name="prune_chunks",
    description="Prunes the chunks by id that are not relevant to the main question from the history of the conversation.",
    parameters={
        "chunk_ids": {"type": "array", "items": {"type": "string"}},
    },
    required=["chunk_ids"],
)


class ToolCallMetadata(BaseModel):
    """Metadata and auxiliary information about a tool call."""

    pass


class Tool(ABC, BaseModel):
    """Base class for tools bound to a schema."""

    tool_schema: ToolSchema

    @abstractmethod
    def __call__(
        self,
        params: Dict[Any, Any],
        overrides: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[str, Optional[ToolCallMetadata]]:
        """
        Args:
            params: The parameters to pass to the tool. Usually provided by the model in the form of a dictionary.
            overrides: Optional overrides to the tool's parameters. These are provided by the system/runtime to control the tool's behavior with
            domain specific knowledge.

        Returns:
            A tuple of the tool output and the tool metadata.
        """
        pass

    def get_format(self, provider: ProviderFormat) -> Dict[str, Any]:
        """Get the tool schema in the specified provider format."""
        return self.tool_schema.to_provider_format(provider)

    def __repr__(self) -> str:
        return f"Tool(name={self.tool_schema.name!r})"


class SerializedTool(Tool):
    """Lightweight placeholder used when deserializing trajectories from JSON.

    Serialized trajectories only persist the tool schema, not the concrete tool implementation
    (which may depend on live clients or other runtime state). This stub preserves the schema so
    downstream formatting logic can continue to operate without requiring the original dependencies.
    """

    def __call__(
        self,
        params: Dict[Any, Any],
        overrides: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[str, Optional[ToolCallMetadata]]:
        raise NotImplementedError(
            "SerializedTool is a placeholder and cannot be executed."
        )


# ============================================================================
# Tool Implementations
# ============================================================================

# Constants and helpers (these should be imported/configured elsewhere)
DOC_TRUNCATION = 51200000
# Default snippet truncation in characters (~512 tokens ≈ 2048 chars for English)
DEFAULT_SNIPPET_MAX_CHARS = 2048


class SearchToolCallMetadata(ToolCallMetadata):
    """
    Metadata about a search corpus tool call.

    - returned_chunk_ids: The chunks that were returned after reranking. These are formatted as <docid>_<chunk_id>.
    - pre_rerank_chunk_ids: The chunks before reranking (original hybrid search order). Only populated when a reranker is used.
    """

    returned_chunk_ids: List[str]
    pre_rerank_chunk_ids: Optional[List[str]] = None


class SearchTool(Tool):
    """A tool that searches the corpus for relevant documents based on the input query.

    Args:
        chroma_client: The Chroma client to use for searching the corpus.
        openai_client: The OpenAI client to use for creating embeddings.
        chroma_collection_name: The name of a Chroma collection, or a list of collection names for load balancing.
            When multiple collections are provided, one is randomly selected for each search request.
        reranker: Optional reranker to reorder results by relevance.
    """

    _chroma_client: chromadb.ClientAPI
    _openai_client: openai.OpenAI
    _bm25_ef: Optional[Bm25EmbeddingFunction]  # Chroma-only sparse embedding
    _openai_ef_name: str = (
        "text-embedding-3-small"  # TODO: consider allowing this to be a field for experiment tracking
    )
    _collections: List[chromadb.Collection]
    _reranker: Optional[Reranker] = (
        None  # TODO: consider allowing this to be a field for experiment tracking
    )
    tool_schema: ToolSchema

    def __init__(
        self,
        chroma_client: chromadb.ClientAPI,
        openai_client: openai.OpenAI,
        chroma_collection_name: Union[str, List[str]],
        openai_ef_name: str = "text-embedding-3-small",
        reranker: Optional[Reranker] = None,
        snippet_max_chars: Optional[int] = None,
        knn_limit: int = 25,
        search_limit: int = 50,
        display_limit: int = 10,
    ) -> None:
        super().__init__(tool_schema=SEARCH_SCHEMA)
        self._chroma_client = chroma_client
        self._openai_client = openai_client
        self._bm25_ef = (
            None if os.environ.get("LOCAL_HYBRID_BM25_DIR")
            else Bm25EmbeddingFunction(avg_len=4000, task="query")
        )
        if isinstance(chroma_collection_name, str):
            collection_names = [chroma_collection_name]
        else:
            collection_names = chroma_collection_name
        self._collections = [] if os.environ.get("LOCAL_HYBRID_BM25_DIR") else [
            self._chroma_client.get_collection(name) for name in collection_names
        ]
        self._openai_ef_name = openai_ef_name
        self._reranker = reranker
        self._snippet_max_chars = snippet_max_chars
        self._knn_limit = knn_limit
        self._search_limit = search_limit
        self._display_limit = display_limit

    def __call__(
        self, params: Dict[Any, Any], overrides: Optional[Dict[Any, Any]] = None
    ) -> Tuple[str, Optional[SearchToolCallMetadata]]:
        log = logger.bind(tool=self.tool_schema.name)
        if not isinstance(params, dict) or "query" not in params:
            log.error("invalid_params", params_type=type(params).__name__)
            raise ValueError(f"Invalid params type: {type(params)}")

        query = params["query"]
        ignore_ids = []
        if overrides is not None and "ignore_ids" in overrides:
            # Ignore ids are used to filter out chunks that have already been retrieved
            ignore_ids = overrides["ignore_ids"]
        log.info(
            "search",
            query=query,
            ignore_ids=len(ignore_ids),
        )

        local_hybrid = _get_local_hybrid()
        if local_hybrid is not None:
            ids, documents, _ = local_hybrid.search(
                query, self._knn_limit, self._search_limit, ignore_ids
            )
        else:
            assert self._bm25_ef is not None
            sparse_vector = self._bm25_ef([query])[0]
            dense_vecs = self.create_embeddings([query])

            search = (
            chromadb.Search()
            .rank(
                chromadb.Rrf(
                    [
                        chromadb.Knn(
                            key="bm25_vector",
                            query=sparse_vector,
                            return_rank=True,
                            limit=self._knn_limit,
                            default=20,
                        ),
                        chromadb.Knn(
                            key="#embedding",
                            query=dense_vecs[0],
                            return_rank=True,
                            limit=self._knn_limit,
                            default=20,
                        ),
                    ]
                )
            )
            .select(chromadb.Key.DOCUMENT, chromadb.Key.METADATA)
            .limit(self._search_limit)
        )
            if ignore_ids:
                search = search.where(chromadb.Key.ID.not_in(ignore_ids))

            # Randomly select a collection for load balancing
            collection = random.choice(self._collections)
            res = _search_with_retry(collection, search)
            ids = res["ids"][0]
            documents = res["documents"][0]

        # Get max_tokens override if provided
        max_tokens_override = (
            overrides.get("max_tokens")
            if overrides and "max_tokens" in overrides
            else None
        )

        # Rerank results if a reranker is provided
        token_counts: List[Optional[int]] = [None] * len(ids)
        if self._reranker is not None:
            rerank_results = self._reranker(
                query, cast(List[str], documents), max_tokens=max_tokens_override
            )
            # Reorder ids, documents, and token counts based on reranked order
            reranked_ids = [ids[r.original_index] for r in rerank_results]
            reranked_documents = [r.document for r in rerank_results]
            token_counts = [r.tokens for r in rerank_results]
            ids = reranked_ids
            documents = reranked_documents
            log.info("reranked_results", num_results=len(ids))

        formatted = [
            "\n# DOCUMENT ID: {}{} \n{}".format(
                id,
                f" ({tokens} tokens)" if tokens is not None else "",
                doc[:DOC_TRUNCATION],
            )
            for id, doc, tokens in zip(ids, cast(List[str], documents), token_counts)
        ][:10]

        return (
            "\n".join(formatted) if len(ids) > 0 else "No results found",
            SearchToolCallMetadata(returned_chunk_ids=ids[: len(formatted)]),
        )

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        resp = self._openai_client.embeddings.create(
            model="text-embedding-3-small", input=texts, encoding_format="float"
        )
        return [e.embedding for e in resp.data]


class GrepCorpusToolCallMetadata(ToolCallMetadata):
    """
    Metadata about a grep corpus tool call.

    - returned_chunk_ids: The chunks that were found for the query. These are formatted as <docid>_<chunk_id>.
    """

    returned_chunk_ids: List[str]


class GrepCorpusTool(Tool):
    """Implementation for the grep_corpus tool.

    Args:
        chroma_client: The Chroma client to use for searching the corpus.
        chroma_collection_name: The name of a Chroma collection, or a list of collection names for load balancing.
            When multiple collections are provided, one is randomly selected for each search request.
        token_counter: Optional callable that counts tokens in a string.
    """

    _chroma_client: chromadb.ClientAPI
    _collections: List[chromadb.Collection]
    _token_counter: Optional[Callable[[str], int]] = None
    tool_schema: ToolSchema

    def __init__(
        self,
        chroma_client: chromadb.ClientAPI,
        chroma_collection_name: Union[str, List[str]],
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        super().__init__(tool_schema=GREP_CORPUS_SCHEMA)
        self._chroma_client = chroma_client
        self._token_counter = token_counter
        # Support both single collection name and list of collection names for load balancing
        if isinstance(chroma_collection_name, str):
            collection_names = [chroma_collection_name]
        else:
            collection_names = chroma_collection_name
        self._collections = [] if os.environ.get("LOCAL_HYBRID_BM25_DIR") else [
            self._chroma_client.get_collection(name) for name in collection_names
        ]

    def __call__(
        self, params: Dict[Any, Any], overrides: Optional[Dict[Any, Any]] = None
    ) -> Tuple[str, Optional[ToolCallMetadata]]:
        log = logger.bind(tool=self.tool_schema.name)
        if not isinstance(params, dict) or "pattern" not in params:
            log.error("invalid_params", params_type=type(params).__name__)
            raise ValueError(f"Invalid params type: {type(params)}")

        query = params["pattern"]
        log.info("grep_corpus", pattern=query)
        # TODO: grep limit is very high, we should probably limit it more
        local_hybrid = _get_local_hybrid()
        if local_hybrid is not None:
            ids, documents, _ = local_hybrid.grep(query, limit=5)
        else:
            search = (
                chromadb.Search()
                .where(chromadb.Key.DOCUMENT.regex(query))
                .select(chromadb.Key.DOCUMENT, chromadb.Key.METADATA)
            ).limit(5)
            # Randomly select a collection for load balancing
            collection = random.choice(self._collections)
            res = _search_with_retry(collection, search)
            ids = res["ids"][0]
            documents = res["documents"][0]
            _ = [
                metadata["source"] for metadata in res["metadatas"][0]
            ]  # Get the doc ids, dropped for now

        # Calculate token counts if token_counter is available
        token_counts: List[Optional[int]] = (
            [self._token_counter(doc) for doc in documents]
            if self._token_counter is not None
            else [None] * len(documents)
        )

        formatted = [
            "\n# DOCUMENT ID: {}{} \n{}".format(
                id,
                f" ({tokens} tokens)" if tokens is not None else "",
                doc[:DOC_TRUNCATION],
            )
            for id, doc, tokens in zip(ids, documents, token_counts)
        ]
        return (
            "\n".join(formatted) if len(ids) > 0 else "No results found",
            GrepCorpusToolCallMetadata(returned_chunk_ids=ids),
        )


class ReadTool(Tool):
    """A tool that reads the content of a document based on its ID.

    Args:
        chroma_client: The Chroma client to use for reading documents.
        chroma_collection_name: The name of a Chroma collection, or a list of collection names for load balancing.
            When multiple collections are provided, one is randomly selected for each read request.
        reranker: Optional reranker to reorder chunks by relevance to a query (provided via overrides).
        token_counter: Optional callable that counts tokens in a string.
        max_tokens: Maximum tokens for the output. If exceeded and reranker + query available,
            reranks to select most relevant chunks within budget.
    """

    tool_schema: ToolSchema
    _chroma_client: chromadb.ClientAPI
    _collections: List[chromadb.Collection]
    _reranker: Optional[Reranker] = (
        None  # TODO: consider allowing this to be a field for experiment tracking
    )
    _token_counter: Optional[Callable[[str], int]] = None
    _max_tokens: Optional[int] = None

    def __init__(
        self,
        chroma_client: chromadb.ClientAPI,
        chroma_collection_name: Union[str, List[str]],
        reranker: Optional[Reranker] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        if max_tokens is not None and token_counter is None:
            raise ValueError("token_counter is required when max_tokens is specified")
        super().__init__(tool_schema=READ_SCHEMA)
        self._chroma_client = chroma_client
        # Support both single collection name and list of collection names for load balancing
        if isinstance(chroma_collection_name, str):
            collection_names = [chroma_collection_name]
        else:
            collection_names = chroma_collection_name
        self._collections = [] if os.environ.get("LOCAL_HYBRID_BM25_DIR") else [
            self._chroma_client.get_collection(name) for name in collection_names
        ]
        self._reranker = reranker
        self._token_counter = token_counter
        self._max_tokens = max_tokens

    def __call__(
        self, params: Dict[Any, Any], overrides: Optional[Dict[Any, Any]] = None
    ) -> Tuple[str, Optional[ToolCallMetadata]]:
        log = logger.bind(tool=self.tool_schema.name)
        if not isinstance(params, dict) or (
            "doc_id" not in params and "id" not in params
        ):
            log.error("invalid_params", params_type=type(params).__name__)
            raise ValueError(f"Invalid params type: {type(params)}")

        doc_id = (
            params["doc_id"] if "doc_id" in params else params["id"]
        )  # Models seem to get confused between doc_id and id, so we support both
        log.info("read", doc_id=doc_id)
        # Model may call with <docid> or <docid>_<chunk_id>, so we need to handle both
        if "_" in doc_id:
            doc_id = doc_id.split("_")[0]
        local_hybrid = _get_local_hybrid()
        if local_hybrid is not None:
            ids, documents = local_hybrid.read_source(doc_id)
        else:
            search = (
            chromadb.Search()
            .where(chromadb.Key("source") == doc_id)
            .select(chromadb.Key.DOCUMENT, chromadb.Key.METADATA)
        ).limit(300)
        # They are named <docid>_<chunk_id>, so we need to get the docid and sort by chunk_id
        # Then reassemble the chunks into a single document
        # Randomly select a collection for load balancing
            collection = random.choice(self._collections)
            res = _search_with_retry(collection, search)
            ids = res["ids"][0]
            documents = res["documents"][0]

        # Sort by chunk_id first to get document order
        zipped = list(zip(ids, documents))
        sorted_zipped = sorted(
            zipped, key=lambda x: int(x[0].split("_")[1]) if "_" in x[0] else 0
        )
        ids = [x[0] for x in sorted_zipped]
        documents = [x[1] for x in sorted_zipped]

        # Assemble full document first
        assembled_document = "".join(cast(List[str], documents))

        # Check if we need to truncate based on token limit
        query = overrides.get("query") if overrides else None
        # Allow max_tokens override from caller (e.g., when budget is low)
        max_tokens = (
            overrides.get("max_tokens")
            if overrides and "max_tokens" in overrides
            else None
        ) or self._max_tokens

        # If we have a reranker and query, use reranking to select relevant chunks
        # The reranker handles token budget internally
        if self._reranker is not None and query is not None and max_tokens is not None:
            rerank_results = self._reranker(
                query, cast(List[str], documents), max_tokens=max_tokens
            )
            # Reranker returns results in relevance order, truncated to fit max_tokens
            # Build set of selected chunk indices
            selected_indices = {r.original_index for r in rerank_results}
            # Filter to only selected chunks, maintaining document order
            filtered_docs = [
                documents[i] for i in range(len(documents)) if i in selected_indices
            ]
            assembled_document = "".join(filtered_docs)
            log.info(
                "reranked_and_filtered",
                original_chunks=len(documents),
                kept_chunks=len(filtered_docs),
            )
        elif self._token_counter is not None and max_tokens is not None:
            # Fallback: no reranker available, truncate by tokens from the start
            total_tokens = self._token_counter(assembled_document)
            if total_tokens > max_tokens:
                log.info(
                    "document_exceeds_token_limit",
                    total_tokens=total_tokens,
                    max_tokens=max_tokens,
                )
                truncated_docs = []
                current_tokens = 0
                for doc in documents:
                    doc_tokens = self._token_counter(doc)
                    if current_tokens + doc_tokens > max_tokens:
                        break
                    truncated_docs.append(doc)
                    current_tokens += doc_tokens
                assembled_document = "".join(truncated_docs)
                log.info(
                    "truncated_by_tokens",
                    original_chunks=len(documents),
                    kept_chunks=len(truncated_docs),
                )

        # Add token count header if token_counter is available
        if self._token_counter is not None:
            doc_tokens = self._token_counter(assembled_document)
            return (f"# Document ({doc_tokens} tokens)\n{assembled_document}", None)

        return (assembled_document, None)


class PruneChunksTool(Tool):
    """A tool that prunes the chunks that are not relevant to the main question from the history of the conversation.

    Given to the model so that it can prune its own context based on the main question and the history of the conversation.

    The tool is functionally a no-op, we detect its usage in the trajectory and prune the chunks from subsequent turns of the model

    """

    tool_schema: ToolSchema

    def __init__(self) -> None:
        super().__init__(tool_schema=PRUNE_CHUNKS_SCHEMA)

    def __call__(
        self, params: Dict[Any, Any], overrides: Optional[Dict[Any, Any]] = None
    ) -> Tuple[str, Optional[ToolCallMetadata]]:
        log = logger.bind(tool=self.tool_schema.name)
        if not isinstance(params, dict) or "chunk_ids" not in params:
            log.error("invalid_params", params_type=type(params).__name__)
            raise ValueError(f"Invalid params type: {type(params)}")

        log.info("prune_chunks", chunk_ids=len(params["chunk_ids"]))
        return ("Pruned", None)


class MultiToolUseTool(Tool):
    """A tool that allows the agent to use multiple tools in parallel to gather information.

    This is used to patch models that don't natively support parallel tool use.

    It should never be saved in a trajectory, it is only used to patch models that don't natively support parallel tool use.

    """

    tool_schema: ToolSchema
    toolset: "ToolSet"

    def __init__(self, toolset: "ToolSet") -> None:
        super().__init__(tool_schema=MULTI_TOOL_USE_SCHEMA)
        self.toolset = toolset

    def __call__(
        self, params: Dict[Any, Any], overrides: Optional[Dict[Any, Any]] = None
    ) -> Tuple[str, Optional[ToolCallMetadata]]:
        results: List[str] = []
        for tool_call in params["tool_calls"]:
            tool = self.toolset.get_tool(tool_call["tool_name"])
            if tool is None:
                raise ValueError(f"Tool {tool_call['tool_name']} not found in toolset")
            res, _ = tool(tool_call["parameters"])
            results.append(res)
        return (json.dumps(results), None)


class UserTextTool(Tool):
    """A tool that allows the agent to produce text for the user.

    This tool is never actually given to the agent, it is only used to represent the user's text in the trajectory
    since abstraction wise treating sending text to the user as a tool call makes sense.

    """

    tool_schema: ToolSchema

    def __init__(self) -> None:
        super().__init__(
            tool_schema=ToolSchema(
                name="user_text",
                description="Produces text for the user.",
                parameters={},
                required=[],
            )
        )

    def __call__(
        self, params: Dict[Any, Any], overrides: Optional[Dict[Any, Any]] = None
    ) -> Tuple[str, ToolCallMetadata]:
        raise ValueError("UserTextTool should not be called directly")


# ============================================================================
# Tool Registry & Composition
# ============================================================================


class ToolSet(BaseModel):
    """A composable set of tools for a SearchAgent."""

    tools: Dict[str, Tool] = Field(default_factory=dict)
    name: Optional[str] = None

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to this set."""
        if tool.tool_schema.name in self.tools:
            raise ValueError(f"Tool with name {tool.tool_schema.name} already exists")
        self.tools[tool.tool_schema.name] = tool

    def remove_tool(self, name: str) -> None:
        """Remove a tool from this set."""
        if name in self.tools:
            del self.tools[name]

    def shallow_copy(
        self, include_list: Optional[List[Type[Tool]]] = None
    ) -> "ToolSet":
        """Shallow copy this toolset, only including tools that are in the include list."""
        if include_list is None:
            include_list = []
        include_types: Tuple[Type[Tool], ...] = tuple(include_list)
        included_tools = {
            name: tool
            for name, tool in self.tools.items()
            if isinstance(tool, include_types)
        }
        return ToolSet(tools=included_tools, name=self.name)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Resolve Method action names and legacy checkpoint tool names."""
        return self.tools.get(canonical_tool_name(name))

    def get_formats(self, provider: ProviderFormat) -> List[Dict[str, Any]]:
        """Get all enabled tools in the specified provider format."""
        return [tool.get_format(provider) for tool in self.tools.values()]

    def __repr__(self) -> str:
        tool_names = ", ".join(sorted(self.tools.keys()))
        name_str = f" ({self.name})" if self.name else ""
        total_count = len(self.tools)
        return f"ToolSet{name_str}[{total_count} tools: {tool_names}]"

    @classmethod
    def from_config(
        cls,
        config: "Config",
        *,
        chroma_collection_name: Union[str, List[str]],
        name: Optional[str] = None,
        reranker: Optional[Reranker] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        max_tokens: Optional[int] = None,
        search_knn_limit: int = 25,
        search_limit: int = 50,
        search_display_limit: int = 10,
        snippet_max_chars: Optional[int] = None,
    ) -> "ToolSet":
        """
        Build a ToolSet with concrete tool implementations using runtime configuration.

        Args:
            config: Runtime configuration providing client constructors.
            chroma_collection_name: Name of a Chroma collection, or a list of collection names
                for load balancing. When multiple collections are provided, one is randomly
                selected for each tool request.
            name: Optional name for the toolset instance.
            reranker: Optional reranker for reordering search results.
            token_counter: Optional callable that counts tokens in a string.
            max_tokens: Maximum tokens for ReadTool or SearchTool output.
            search_knn_limit: Per-retriever KNN limit for SearchTool (default 25).
            search_limit: Overall search result limit after RRF fusion (default 50).
            search_display_limit: Number of results to format for the model (default 10).
            snippet_max_chars: Max characters per search result snippet (None = no limit).
        """

        chroma_client = config.get_chroma_client()
        openai_client = config.get_openai_client()

        toolset = cls(name=name)

        search_tool = SearchTool(
            chroma_client=chroma_client,
            openai_client=openai_client,
            chroma_collection_name=chroma_collection_name,
            reranker=reranker,
            snippet_max_chars=snippet_max_chars,
            knn_limit=search_knn_limit,
            search_limit=search_limit,
            display_limit=search_display_limit,
        )
        toolset.add_tool(search_tool)

        toolset.add_tool(
            GrepCorpusTool(
                chroma_client=chroma_client,
                chroma_collection_name=chroma_collection_name,
                token_counter=token_counter,
            )
        )

        toolset.add_tool(
            ReadTool(
                chroma_client=chroma_client,
                chroma_collection_name=chroma_collection_name,
                reranker=reranker,
                token_counter=token_counter,
                max_tokens=max_tokens,
            )
        )

        toolset.add_tool(PruneChunksTool())

        return toolset
