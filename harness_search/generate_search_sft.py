"""CLI for generating search SFT trajectories in bulk."""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import structlog
import tiktoken

from harness_search.agent import (
    AnthropicAgentInferenceModel,
    Agent,
    AgentInferenceModel,
    OpenAIAgentInferenceModel,
    TinkerAgentInferenceModel,
    TokenBudgetRetrievalSubagent,
)
from harness_search.config import get_config
from datagen.search_dataset import SearchDataset, get_dataset, DATASET_REGISTRY
from openai_harmony import (
    HarmonyEncodingName,
    RenderConversationConfig,
    load_harmony_encoding,
)
from harness_search.prompts import get_retrieval_subagent_prompt
from harness_search.rerank import VLLMQwen3Reranker
from harness_search.tasks import SearchTaskOutput
from harness_search.tools import ToolSet
from harness_search.trajectory import Observation, Trajectory


logger = structlog.get_logger("search_agent.datagen.search_sft")


class SearchSFTGenerator:
    """Generate search trajectories for SFT training with retry + parallelism."""

    def __init__(
        self,
        *,
        dataset: SearchDataset,
        agent_factory: Callable[[], Agent],
        output_dir: Path,
        # TODO: rename num_documents to num_queries
        num_documents: Optional[int],  # None means use all available queries
        num_rollouts: int = 1,
        num_threads: int = 1,
        seed: int = 0,
        max_retries: int = 3,
        overwrite: bool = False,
    ) -> None:
        if num_documents is not None and num_documents < 1:
            raise ValueError("num_documents must be >= 1 or None for all")
        if num_rollouts < 1:
            raise ValueError("num_rollouts must be >= 1")
        if num_threads < 1:
            raise ValueError("num_threads must be >= 1")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")

        self.dataset = dataset
        self.agent_factory = agent_factory
        self.output_dir = output_dir
        self.num_documents = num_documents
        self.num_rollouts = num_rollouts
        self.num_threads = num_threads
        self.seed = seed
        self.max_retries = max_retries
        self.overwrite = overwrite
        self.logger = logger.bind(generator="SearchSFTGenerator")

    def run(self) -> List[Tuple[str, int]]:
        """Generate trajectories for the selected queries.

        Returns:
            A list of (query_id, rollout_idx) tuples that failed to generate after retries.
        """

        self.output_dir.mkdir(parents=True, exist_ok=True)
        query_ids = self._select_query_ids()

        # Create all (query_id, rollout_idx) pairs to process
        tasks = [
            (query_id, rollout_idx)
            for query_id in query_ids
            for rollout_idx in range(self.num_rollouts)
        ]

        self.logger.info(
            "generation_started",
            total_queries=len(query_ids),
            num_rollouts=self.num_rollouts,
            total_tasks=len(tasks),
            output_dir=str(self.output_dir),
            num_threads=self.num_threads,
        )

        failed: List[Tuple[str, int]] = []
        if self.num_threads == 1:
            for query_id, rollout_idx in tasks:
                if not self._generate_with_retries(query_id, rollout_idx):
                    failed.append((query_id, rollout_idx))
        else:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(
                        self._generate_with_retries, query_id, rollout_idx
                    ): (query_id, rollout_idx)
                    for query_id, rollout_idx in tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        success = future.result()
                    except Exception as exc:  # pragma: no cover - safety net
                        self.logger.exception(
                            "generation_unexpected_exception",
                            query_id=task[0],
                            rollout_idx=task[1],
                            error=str(exc),
                        )
                        success = False
                    if not success:
                        failed.append(task)

        self.logger.info(
            "generation_finished",
            total=len(tasks),
            succeeded=len(tasks) - len(failed),
            failed=len(failed),
        )
        if failed:
            self.logger.warning("failed_tasks", tasks=failed)
        return failed

    def _select_query_ids(self) -> List[str]:
        available_query_ids = self.dataset.get_all_query_ids(split="train")

        # If num_documents is None, use all available queries
        if self.num_documents is None:
            self.logger.info(
                "using_all_queries",
                available_count=len(available_query_ids),
            )
            rng = random.Random(self.seed)
            rng.shuffle(available_query_ids)
            return available_query_ids

        if len(available_query_ids) < self.num_documents:
            raise ValueError(
                f"Requested {self.num_documents} queries but dataset only has "
                f"{len(available_query_ids)} entries."
            )
        rng = random.Random(self.seed)
        rng.shuffle(available_query_ids)
        selected = available_query_ids[: self.num_documents]
        self.logger.debug("selected_queries", count=len(selected))
        return selected

    def _get_output_filename(self, query_id: str, rollout_idx: int) -> str:
        """Get the output filename for a given query and rollout index."""
        if self.num_rollouts == 1:
            return f"{query_id}.json"
        return f"{query_id}_rollout_{rollout_idx}.json"

    def _generate_with_retries(self, query_id: str, rollout_idx: int = 0) -> bool:
        filename = self._get_output_filename(query_id, rollout_idx)
        output_path = self.output_dir / filename
        if output_path.exists() and not self.overwrite:
            self.logger.info(
                "skipping_existing_output",
                query_id=query_id,
                rollout_idx=rollout_idx,
                path=str(output_path),
            )
            return True

        for attempt in range(1, self.max_retries + 1):
            try:
                output = self._generate_single(query_id)
                self._write_output(output, rollout_idx)
                self.logger.info(
                    "generation_success",
                    query_id=query_id,
                    rollout_idx=rollout_idx,
                    attempt=attempt,
                    path=str(output_path),
                )
                return True
            except Exception as exc:
                self.logger.warning(
                    "generation_attempt_failed",
                    query_id=query_id,
                    rollout_idx=rollout_idx,
                    attempt=attempt,
                    error=str(exc),
                )
        self.logger.error(
            "generation_failed", query_id=query_id, rollout_idx=rollout_idx
        )
        return False

    def _generate_single(self, query_id: str) -> SearchTaskOutput:
        agent = self.agent_factory()
        _, query_text = self.dataset.get_query_by_id(query_id)

        initial_observation = Observation(
            observations=[get_retrieval_subagent_prompt(query_text)],
            sources=["user"],
            tool_metadata=[None],
        )
        trajectory = agent(initial_observation=initial_observation)

        output = SearchTaskOutput(
            trajectory=trajectory,
            query_id=query_id,
            dataset_name=self.dataset.name,
        )
        output.log_trajectory_stats()
        return output

    def _write_output(self, output: SearchTaskOutput, rollout_idx: int = 0) -> None:
        filename = self._get_output_filename(output.query_id, rollout_idx)
        output_path = self.output_dir / filename
        if output_path.exists() and not self.overwrite:
            return
        with output_path.open("w") as fp:
            json.dump(output.model_dump(mode="json"), fp, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate search trajectories for SFT training."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="browsecompplus",
        choices=list(DATASET_REGISTRY.keys()),
        help="Name of the search dataset to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write SearchTaskOutput JSON files.",
    )
    parser.add_argument(
        "--num-queries",
        type=str,
        required=True,
        help="Number of queries to sample and generate trajectories for. Use 'all' to use all available queries.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Number of rollouts (trajectories) to generate per query.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to select the subset of queries.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker threads to use for generation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of attempts per query before giving up.",
    )
    # TODO: make this config
    parser.add_argument(
        "--chroma-collection",
        type=str,
        required=True,
        help="Chroma collection name backing the retrieval tools.",
    )
    parser.add_argument(
        "--inference-provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "tinker", "openai"],
        help="Inference provider used to run the agent.",
    )
    parser.add_argument(
        "--anthropic-model",
        type=str,
        default="claude-opus-4-5-20251101",  # "claude-sonnet-4-5-20250929",
        help="Anthropic model name when provider is anthropic.",
    )
    parser.add_argument(
        "--tinker-model",
        type=str,
        default="Qwen/Qwen3.5-27B",
        help="Tinker base model when provider is tinker.",
    )
    parser.add_argument(
        "--tinker-model-path",
        type=str,
        default=None,
        help="Optional Tinker sampler weights path overriding --tinker-model.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-5",
        help="OpenAI model name (e.g., chatgpt-5) when provider is openai.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens for the inference provider.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for the inference provider.",
    )
    parser.add_argument(
        "--threshold-budget",
        type=int,
        default=16384,
        help="Token threshold that triggers pruning in the retrieval subagent.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=32768,
        help="Maximum token budget allowed for the retrieval subagent.",
    )
    parser.add_argument(
        "--max-trajectory-length",
        type=int,
        default=128,
        help="Maximum number of steps per trajectory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if they already exist.",
    )
    parser.add_argument(
        "--rerank-max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for reranker and ReadTool output.",
    )
    return parser.parse_args()


def create_inference_model_factory(
    args: argparse.Namespace,
    strict_mode: bool = True,
) -> Callable[[], AgentInferenceModel]:
    """Create a factory for inference models based on the provider.

    Args:
        args: Parsed command line arguments.
        strict_mode: When True (default), use strict JSON parsing and raise on
            malformed model outputs. When False, use json_repair for parsing
            and retry on certain parsing errors. Only applies to TinkerAgentInferenceModel.
    """
    config = get_config()
    provider = args.inference_provider

    if provider == "anthropic":
        anthropic_client = config.get_anthropic_client()

        return lambda: AnthropicAgentInferenceModel(
            anthropic_client=anthropic_client,
            model=args.anthropic_model,
            max_tokens=args.max_completion_tokens,
            temperature=args.temperature,
        )

    if provider == "tinker":
        tinker_client = config.get_tinker_service_client()
        tinker_model_path = getattr(args, "tinker_model_path", None)
        if tinker_model_path:
            sampling_client = tinker_client.create_sampling_client(
                model_path=tinker_model_path
            )
        else:
            sampling_client = tinker_client.create_sampling_client(
                base_model=args.tinker_model
            )

        return lambda: TinkerAgentInferenceModel(
            tinker_sampling_client=sampling_client,
            model=args.tinker_model,
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            strict_mode=strict_mode,
        )

    if provider == "openai":
        openai_client = config.get_openai_client()
        openai_api_style = os.getenv("OPENAI_API_STYLE", "responses").strip().lower()
        if openai_api_style not in {"responses", "chat_completions", "auto"}:
            raise ValueError(
                "OPENAI_API_STYLE must be one of: responses, chat_completions, auto"
            )

        openai_model = getattr(args, "openai_model", "gpt-5")
        return lambda: OpenAIAgentInferenceModel(
            openai_client=openai_client,
            model=openai_model,
            max_output_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            api_style=openai_api_style,
        )

    raise ValueError(f"Unsupported inference provider: {provider}")


def build_agent_factory(
    *,
    args: argparse.Namespace,
    inference_model_factory: Callable[[], AgentInferenceModel],
    toolset: ToolSet,
) -> Callable[[], TokenBudgetRetrievalSubagent]:
    harmony_enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def token_counter(trajectory: Trajectory) -> int:
        return len(
            harmony_enc.render_conversation(
                trajectory.to_openai_harmony_format(),
                config=RenderConversationConfig(auto_drop_analysis=False),
            )
        )

    def factory() -> TokenBudgetRetrievalSubagent:
        inference_model = inference_model_factory()
        return TokenBudgetRetrievalSubagent(
            toolset=toolset,
            inference_model=inference_model,
            token_counter=token_counter,
            max_trajectory_length=args.max_trajectory_length,
            threshold_budget=args.threshold_budget,
            token_budget=args.token_budget,
        )

    return factory


def main() -> None:
    args = parse_args()

    dataset = get_dataset(args.dataset_name)

    # Parse num_queries: either "all" or an integer
    if args.num_queries.lower() == "all":
        num_queries = None  # Signal to use all available queries
    else:
        try:
            num_queries = int(args.num_queries)
        except ValueError:
            raise ValueError(
                f"--num-queries must be 'all' or an integer, got: {args.num_queries}"
            )

    # Create reranker with tiktoken-based token counter
    config = get_config()
    tiktoken_encoding = tiktoken.get_encoding("o200k_harmony")
    rerank_token_counter = lambda text: len(tiktoken_encoding.encode(text))
    reranker = VLLMQwen3Reranker(
        token_counter=rerank_token_counter,
        max_tokens=args.rerank_max_tokens,
    )

    # Create shared toolset
    toolset = ToolSet.from_config(
        config,
        chroma_collection_name=args.chroma_collection,
        reranker=reranker,
        token_counter=rerank_token_counter,
        max_tokens=args.rerank_max_tokens,
    )

    inference_model_factory = create_inference_model_factory(args)
    agent_factory = build_agent_factory(
        args=args,
        inference_model_factory=inference_model_factory,
        toolset=toolset,
    )

    generator = SearchSFTGenerator(
        dataset=dataset,
        agent_factory=agent_factory,
        output_dir=args.output_dir,
        num_documents=num_queries,
        num_rollouts=args.num_rollouts,
        num_threads=args.num_workers,
        seed=args.seed,
        max_retries=args.max_retries,
        overwrite=args.overwrite,
    )
    failed = generator.run()

    if failed:
        logger.error("generation_completed_with_failures", failed_tasks=failed)
        raise SystemExit(1)
    logger.info("generation_completed_successfully")


if __name__ == "__main__":
    main()
