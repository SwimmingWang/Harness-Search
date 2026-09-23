from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Callable, List, Optional

import requests
import structlog

logger = structlog.get_logger("search_agent.rerank")


@dataclass
class RerankResult:
    """Result of reranking a single document."""

    document: str
    score: float
    original_index: int
    tokens: Optional[int] = None  # Token count, populated if token_counter is available


class Reranker(ABC):
    """Abstract base class for reranking documents based on a query."""

    def __init__(
        self,
        token_counter: Optional[Callable[[str], int]] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize the reranker.

        Args:
            token_counter: Optional callable that counts tokens in a string.
            max_tokens: Maximum total tokens for the output. Documents are returned
                in reranked order until this budget is exhausted.

        Raises:
            ValueError: If max_tokens is specified without a token_counter.
        """
        if max_tokens is not None and token_counter is None:
            raise ValueError("token_counter is required when max_tokens is specified")
        self.token_counter = token_counter
        self.max_tokens = max_tokens

    def _truncate_results(
        self, results: List[RerankResult], max_tokens: Optional[int] = None
    ) -> List[RerankResult]:
        """Truncate results to fit within max_tokens total.

        Also populates the tokens field for each result if token_counter is available.

        Args:
            results: List of RerankResult objects to truncate.
            max_tokens: Optional override for max_tokens. If not provided,
                uses the instance's max_tokens setting.
        """
        # If we have a token_counter, populate tokens for all results
        if self.token_counter is not None:
            for result in results:
                result.tokens = self.token_counter(result.document)

        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if self.token_counter is None or effective_max_tokens is None:
            return results

        truncated: List[RerankResult] = []
        total_tokens = 0
        for result in results:
            doc_tokens = result.tokens  # Already calculated above
            assert doc_tokens is not None
            if total_tokens + doc_tokens > effective_max_tokens:
                logger.info(
                    "truncating_results",
                    kept=len(truncated),
                    dropped=len(results) - len(truncated),
                    total_tokens=total_tokens,
                    max_tokens=effective_max_tokens,
                )
                break
            truncated.append(result)
            total_tokens += doc_tokens

        return truncated

    @abstractmethod
    def _rerank(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
    ) -> List[RerankResult]:
        """
        Rerank documents based on relevance to the query.

        Subclasses must implement this method to perform the actual reranking.

        Args:
            query: The search query to rank documents against.
            documents: List of document strings to rerank.
            instruction: Optional instruction for the reranker.

        Returns:
            List of RerankResult objects sorted by relevance (highest first).
        """
        pass

    def __call__(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: The search query to rank documents against.
            documents: List of document strings to rerank.
            instruction: Optional instruction for the reranker.
            max_tokens: Optional override for max_tokens budget. If provided,
                overrides the instance's max_tokens for this call only.

        Returns:
            List of RerankResult objects sorted by relevance (highest first),
            truncated to fit within max_tokens if token_counter is provided.
        """
        start = time.perf_counter()
        results = self._rerank(query, documents, instruction)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > 1500:
            logger.warning(
                "Extremely slow reranking",
                elapsed_ms=round(elapsed_ms, 1),
            )
        return self._truncate_results(results, max_tokens=max_tokens)


class VLLMQwen3Reranker(Reranker):
    """Reranker backed by a local vLLM server running Qwen3-Reranker-8B.

    Serves Qwen3-Reranker-8B locally via vLLM's /score endpoint (original
    Qwen3-reranker sequence-classification conversion).

    Server launch:
        vllm serve Qwen/Qwen3-Reranker-8B --port 8011 \
          --hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"],
                           "classifier_from_token": ["no", "yes"],
                           "is_original_qwen3_reranker": true}'
    Point at it with VLLM_RERANKER_URL (default http://127.0.0.1:8011).
    """

    PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    DEFAULT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "Qwen/Qwen3-Reranker-8B",
        token_counter: Optional[Callable[[str], int]] = None,
        max_tokens: Optional[int] = None,
        batch_size: int = 32,
        timeout_s: int = 360,
    ):
        super().__init__(token_counter=token_counter, max_tokens=max_tokens)
        import os

        self.base_url = (
            base_url or os.getenv("VLLM_RERANKER_URL", "http://127.0.0.1:8011")
        ).rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.timeout_s = timeout_s

    def _rerank(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
    ) -> List[RerankResult]:
        if not documents:
            return []
        if instruction is None:
            instruction = self.DEFAULT_INSTRUCTION

        text_1 = f"{self.PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n"
        scores: List[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            payload = {
                "model": self.model,
                "text_1": text_1,
                "text_2": [f"<Document>: {doc}{self.SUFFIX}" for doc in batch],
                "truncate_prompt_tokens": 8192,
            }
            last_error: Optional[Exception] = None
            for attempt in range(3):
                try:
                    response = requests.post(
                        f"{self.base_url}/score",
                        json=payload,
                        timeout=self.timeout_s,
                    )
                    response.raise_for_status()
                    data = response.json()["data"]
                    scores.extend(float(item["score"]) for item in data)
                    last_error = None
                    break
                except requests.exceptions.RequestException as exc:
                    last_error = exc
                    logger.warning(
                        "vllm_rerank_retry", attempt=attempt + 1, error=str(exc)
                    )
                    time.sleep(2**attempt)
            if last_error is not None:
                logger.error("vllm_rerank_failed", error=str(last_error))
                raise last_error

        results = [
            RerankResult(document=doc, score=score, original_index=idx)
            for idx, (doc, score) in enumerate(zip(documents, scores))
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results


if __name__ == "__main__":
    import argparse
    import tiktoken

    parser = argparse.ArgumentParser(description="Run reranker example")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Maximum tokens for output (default: 30)",
    )
    args = parser.parse_args()

    logger.info("Running reranker example", max_tokens=args.max_tokens)

    # Simple token counter just to demonstrate the concept, not accurate token for all models of course
    enc = tiktoken.get_encoding("o200k_harmony")
    token_counter = lambda text: len(enc.encode(text))

    reranker: Reranker = VLLMQwen3Reranker(
        token_counter=token_counter,
        max_tokens=args.max_tokens,
    )

    query = "What is the capital of China?"
    documents = [
        "The capital of France is Paris.",
        "The capital of China is Beijing.",
        "The capital of Poland is Warsaw.",
        "The capital of Germany is Berlin.",
        "Chocolate is a delicious treat.",
        "Pizza is a food",
        "China has a population of 1.4 billion.",
        "Germany has a population of 83 million.",
        "Poland has a population of 38 million.",
        "Warsaw is the capital of Poland.",
        "Berlin is the capital of Germany.",
        "Paris is the capital of France.",
        "Beijing is the capital of China.",
        "Warsaw is the capital of Poland.",
        "Berlin is the capital of Germany.",
        "Shanghai is not the capital of China.",
        "Japan is closer to China than to the United States.",
        "The capital of China has been Beijing for a long time.",
    ]
    results = reranker(query, documents)
    logger.info("rerank_complete", num_results=len(results), max_tokens=args.max_tokens)
    for result in results:
        logger.info("result", score=result.score, document=result.document)
