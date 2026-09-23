"""Evaluate Harness-Search against a local vLLM OpenAI-compatible endpoint.

This evaluator uses local vLLM sampling
client with raw token-id calls to vLLM /v1/completions. It is intended for
parity checks of the released Hugging Face checkpoint served by vLLM.
"""

from __future__ import annotations

import argparse
import json_repair
import asyncio
import json
import os
import random
import re
import string
from collections import Counter
import time
from pathlib import Path
from typing import Any, Dict, List

import structlog
import tiktoken

# Allow direct execution while keeping imports package-relative.
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datagen.search_dataset import SearchDataset, get_dataset
from harness_search.config import get_config
from harness_search.tools import (
    GrepCorpusTool,
    PruneChunksTool,
    ReadTool,
    SearchTool,
    ToolSet,
    UserTextTool,
)
from harness_search.longseal_tools import create_longseal_toolset
from runtime.environment import MAX_TURNS, SEARCH_DISPLAY_LIMIT, HarnessSearchEnv
from harness_search.retrieval_policy import VllmRetrievalPolicy, ChatRetrievalPolicy

logger = structlog.get_logger("harness_search_evaluate")

SAVE_FULL_TRAJECTORIES = os.environ.get("SAVE_FULL_TRAJECTORIES", "0") == "1"


def save_full_trajectory(env: HarnessSearchEnv) -> None:
    traj_root = os.environ.get("TRAJECTORY_SAVE_PATH") or os.environ.get(
        "LOG_PATH", "./tmp/rl_ultra_v3"
    )
    full_dir = os.path.join(traj_root, "full")
    os.makedirs(full_dir, exist_ok=True)

    turns = []
    for i, (action, obs) in enumerate(zip(env._all_actions, env._all_observations)):
        turn_record = {"turn": i}
        if action.reasoning:
            turn_record["reasoning"] = action.reasoning

        tool_calls = []
        for tool, params in zip(action.tools, action.params):
            name = "user_text" if isinstance(tool, UserTextTool) else tool.tool_schema.name
            tool_calls.append({"tool": name, "params": params})
        turn_record["tool_calls"] = tool_calls

        tool_returns = []
        for j, obs_text in enumerate(obs.observations):
            tr = {"text": obs_text}
            if j < len(obs.tool_metadata) and obs.tool_metadata[j] is not None:
                try:
                    tr["metadata"] = obs.tool_metadata[j].model_dump()
                except Exception:
                    tr["metadata"] = str(obs.tool_metadata[j])
            tool_returns.append(tr)
        turn_record["tool_returns"] = tool_returns
        turns.append(turn_record)

    record = {
        "query_id": env.query_id,
        "query_text": env.wm.query,
        "dataset": env.dataset.name,
        "system_prompt": env.system_prompt,
        "turns": turns,
        "curated_ids": env.wm.curated_ids,
        "curated_importance": dict(env.wm.curated_importance),
        "curated_intents": dict(env.wm.curated_notes),
        "latest_audit": env.summary_auditor.latest,
        "current_intent": env.wm.current_intent,
        "intent_history": list(env.wm.intent_history),
        "reward": env._terminal_reward,
        "metrics": {
            k: v
            for k, v in env._terminal_metrics.items()
            if isinstance(v, (int, float, str, bool))
        },
        "turn_trace": getattr(env, "_turn_trace", []),
    }
    qid_safe = str(env.query_id).replace("/", "_")
    with open(os.path.join(full_dir, f"{qid_safe}.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)


def _normalize_qa_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _qa_em_f1(prediction: str, gold_answers: List[str]) -> tuple[float, float]:
    pred = _normalize_qa_answer(prediction)
    em = 0.0
    best_f1 = 0.0
    for gold_raw in gold_answers:
        gold = _normalize_qa_answer(gold_raw)
        em = max(em, float(pred == gold))
        pred_tokens, gold_tokens = pred.split(), gold.split()
        if not pred_tokens and not gold_tokens:
            score = 1.0
        elif not pred_tokens or not gold_tokens:
            score = 0.0
        else:
            common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
            if not common:
                score = 0.0
            else:
                precision = common / len(pred_tokens)
                recall = common / len(gold_tokens)
                score = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, score)
    return em, best_f1


async def _generate_qa_answer(env: HarnessSearchEnv, policy) -> str:
    evidence = []
    for doc_id in env.wm.curated_ids[:30]:
        store = env.wm.doc_store.get(doc_id, {})
        text = store.get("full_text") or store.get("snippet") or ""
        evidence.append(f"DOCUMENT {doc_id}:\n{text[:2500]}")
    prompt = (
        "Answer the question using only the retrieved evidence. Return JSON only as "
        "{\"answer\": \"...\"}. Give the shortest exact answer; for yes/no questions "
        "return exactly yes or no. If evidence is insufficient, make the best supported "
        "short answer and do not explain.\n\nQUESTION:\n" + env.wm.query
        + "\n\nEVIDENCE:\n" + ("\n\n".join(evidence) or "(no curated evidence)")
        + "\n\nEVIDENCE SUMMARY:\n"
        + str(env.summary_auditor.latest.get("summary", ""))[:3000]
    )
    base_url = policy.base_url.rstrip("/")
    url = base_url + "/chat/completions" if base_url.endswith("/v1") else base_url + "/v1/chat/completions"
    payload = {"model": policy.model, "messages": [
        {"role": "system", "content": "You are a concise multi-hop QA answerer."},
        {"role": "user", "content": prompt}], "temperature": 0.0,
        "max_tokens": 256, "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False}}
    if isinstance(policy, ChatRetrievalPolicy):
        if "dashscope.aliyuncs.com" in policy.base_url:
            payload["enable_thinking"] = False
            payload.pop("response_format", None)
        data = await asyncio.to_thread(policy._post, payload)
    else:
        data = await asyncio.to_thread(policy._post_json_url, url, payload)
    content = data["choices"][0]["message"].get("content") or ""
    parsed = json_repair.loads(content)
    if isinstance(parsed, dict) and parsed.get("answer") is not None:
        return str(parsed["answer"]).strip()
    return str(content).strip()


async def run_single_episode(
    env: HarnessSearchEnv,
    policy,
) -> Dict:
    ob, stop_condition = await env.initial_observation()
    turns = 0
    start = time.time()
    env._turn_trace = []
    # This is deliberately independent of the environment's semantic turn
    # budget. Mandatory checkpoint actions can run before the environment's
    # max-turn termination branch, so a bad state transition could otherwise
    # keep an episode alive forever. Allow a small checkpoint grace window,
    # then fail this query instead of blocking the whole evaluation.
    hard_action_limit = env.max_turns + 12

    while True:
        if turns >= hard_action_limit:
            logger.error(
                "episode_hard_action_limit",
                qid=env.query_id,
                actions=turns,
                max_turns=env.max_turns,
                intent_revision_required=env._intent_revision_required,
                checkpoint_curate_required=env._checkpoint_curate_required,
                terminal_curate_required=env._terminal_curate_required,
                pool=len(env.wm.pool_ids),
                curated=len(env.wm.curated_ids),
            )
            raise RuntimeError(
                f"Episode exceeded hard action limit ({hard_action_limit}); "
                "possible checkpoint state loop"
            )
        if env._intent_revision_required:
            required_action = "redirect"
        elif env._checkpoint_curate_required or env._terminal_curate_required:
            required_action = "curate"
        else:
            required_action = "policy_choice"

        input_state = env.wm.to_text()
        if required_action in {"redirect", "curate"}:
            if hasattr(policy, "_forced_tool_context"):
                forced_prompt, _ = policy._forced_tool_context(env, required_action)
                prompt = (
                    "[SYSTEM]\nYou are the policy model at a mandatory tool checkpoint. "
                    "Return only the JSON arguments for the required tool. Do not answer "
                    "the query.\n\n[USER]\n" + forced_prompt
                )
            else:
                # ChatRetrievalPolicy builds its forced-tool context inside
                # forced_tool_call(); this prompt is trace-only and must not
                # invoke GPT-OSS adapter internals or mutate Qwen candidate state.
                prompt = input_state
        else:
            try:
                prompt = env.enc.decode(list(ob.to_ints()))
            except Exception:
                prompt = input_state

        if env._intent_revision_required:
            ac_with_logprobs = await policy.forced_tool_call(env, "redirect")
        elif env._checkpoint_curate_required:
            ac_with_logprobs = await policy.forced_tool_call(env, "curate")
        elif env._terminal_curate_required:
            ac_with_logprobs = await policy.forced_tool_call(env, "curate")
        else:
            if isinstance(policy, ChatRetrievalPolicy):
                ac_with_logprobs = await policy.action(env)
            else:
                ac_with_logprobs = await policy(ob, stop_condition)
        action_count_before = len(env._all_actions)
        observation_count_before = len(env._all_observations)
        step_result = await asyncio.wait_for(
            env.step(ac_with_logprobs.tokens),
            timeout=getattr(policy, "timeout", 300) + 30,
        )
        try:
            model_output = env.enc.decode(list(ac_with_logprobs.tokens))
        except Exception:
            model_output = str(ac_with_logprobs.tokens)

        action_record = {}
        if len(env._all_actions) > action_count_before:
            action = env._all_actions[-1]
            action_record = {
                "reasoning": action.reasoning or "",
                "tool_calls": [
                    {
                        "tool": (
                            "user_text" if isinstance(tool, UserTextTool)
                            else tool.tool_schema.name
                        ),
                        "params": params,
                    }
                    for tool, params in zip(action.tools, action.params)
                ],
            }
        tool_returns = []
        if len(env._all_observations) > observation_count_before:
            tool_returns = list(env._all_observations[-1].observations)
        env._turn_trace.append({
            "turn": turns + 1,
            "required_action": required_action,
            "input_state": input_state,
            "prompt": prompt,
            "model_output": model_output,
            "parsed_action": action_record,
            "tool_returns": tool_returns,
            "relevance_judge_calls_total": env.memory_operator.relevance_judge.calls,
            "summary_auditor_calls_total": env.summary_auditor.calls,
            "current_intent_after_turn": env.wm.current_intent,
            "curated_ids_after_turn": list(env.wm.curated_ids),
            "summary_auditor_after_turn": dict(env.summary_auditor.latest),
        })
        turns += 1
        if step_result.episode_done:
            if not env._terminal_metrics:
                env._terminal_reward, env._terminal_metrics = env._compute_terminal_reward()
                for metric_name, metric_value in (step_result.metrics or {}).items():
                    env._terminal_metrics.setdefault(metric_name, metric_value)
            break
        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition

    elapsed = time.time() - start
    result = {
        "reward": env._terminal_reward,
        "turns": turns,
        "n_curated": len(env.wm.curated_ids),
        "n_pool": len(env.wm.pool_ids),
        "elapsed_s": round(elapsed, 1),
        "error": env._terminal_metrics.get("no_error", 1.0) == 0.0,
        "tool_types_used": list(env._tool_types_used),
        "total_curate_calls": env._total_curate_calls,
        "policy_curate_prompts": env._terminal_curate_prompts,
        "intent_checkpoint_prompts": env._intent_checkpoint_prompt_total,
        "checkpoint_curate_prompts": env._checkpoint_curate_prompt_total,
        "runtime_intent_revisions": env._intent_revision_count,
        "intent_revisions": len(env.wm.intent_history),
        "current_intent": env.wm.current_intent,
        "relevance_judge_calls": env.memory_operator.relevance_judge.calls,
        "summary_auditor_calls": env.summary_auditor.calls,
        "summary_answer_ready": env.summary_auditor.answer_ready,
        "curated_with_intent": sum(
            1 for doc_id in env.wm.curated_ids if env.wm.curated_notes.get(doc_id)
        ),
        "curated_intents": dict(env.wm.curated_notes),
        "latest_audit": env.summary_auditor.latest,
        # Evaluation-only diagnostics. Gold IDs are never placed in policy or
        # judge context; these aggregate recalls only identify which stage lost
        # already-retrieved evidence.
        "relevance_judge_gold_recall": (
            len(set(env.memory_operator.relevance_annotations) & set(env._gold_doc_ids))
            / max(len(set(env._gold_doc_ids)), 1)
        ),
        "native_curation_exposed_gold_recall": (
            len(
                set(getattr(env, "_native_curation_exposed_ids", set()))
                & set(env._gold_doc_ids)
            ) / max(len(set(env._gold_doc_ids)), 1)
        ),
    }
    result.update(env._terminal_metrics)
    return result


async def eval_single_query(
    qid: str,
    dataset: SearchDataset,
    toolset: ToolSet,
    search_tool: Any,
    text_token_counter,
    policy,
    max_turns: int,
) -> Dict:
    _, query_text = dataset.get_query_by_id(qid)
    if dataset.name == "longsealqa":
        toolset, search_tool = create_longseal_toolset(dataset.get_documents(qid))
    env = HarnessSearchEnv(
        toolset=toolset,
        search_tool=search_tool,
        query_id=qid,
        query_text=query_text,
        dataset=dataset,
        text_token_counter=text_token_counter,
        max_turns=max_turns,
    )
    if dataset.name == "longsealqa":
        # URL document IDs legitimately contain underscores. They must not be
        # interpreted as chunk suffixes by the fixed-corpus normalizer.
        env._normalize_ids = False
        env.memory_operator.configure(normalize_ids=False)
    try:
        result = await run_single_episode(env=env, policy=policy)
        result["query_id"] = qid
        result["query"] = query_text[:80]
        if getattr(dataset, "is_answer_evaluation", False):
            predicted = await _generate_qa_answer(env, policy)
            gold_answers = dataset.get_golden_answers(qid)
            answer_em, answer_f1 = _qa_em_f1(predicted, gold_answers)
            result.update({"predicted_answer": predicted,
                           "golden_answers": gold_answers,
                           "answer_em": answer_em, "answer_f1": answer_f1})
        if SAVE_FULL_TRAJECTORIES:
            save_full_trajectory(env)
        logger.info(
            "episode_result",
            qid=qid,
            recall=round(result.get("recall", 0), 3),
            trajectory_recall=round(result.get("trajectory_recall", 0), 3),
            final_answer_recall=round(result.get("final_answer_recall", 0), 3),
            reward=round(result.get("reward", 0), 3),
            curated=result["n_curated"],
            pool=result["n_pool"],
            turns=result["turns"],
            error=result["error"],
            time=result["elapsed_s"],
        )
        return result
    except Exception as exc:
        actual_turns = len(env._all_actions)
        logger.exception("episode_failed", qid=qid, error_type=type(exc).__name__, error=str(exc), turns=actual_turns)
        return {
            "query_id": qid,
            "query": query_text[:80],
            "error": True,
            "reward": 0,
            "recall": 0,
            "trajectory_recall": 0,
            "final_answer_recall": 0,
            "precision": 0,
            "n_curated": len(env.wm.curated_ids),
            "n_pool": len(env.wm.pool_ids),
            "turns": actual_turns,
            "total_curate_calls": 0,
            "policy_curate_prompts": 0,
            "intent_checkpoint_prompts": 0,
            "checkpoint_curate_prompts": 0,
            "runtime_intent_revisions": 0,
        }


async def eval_queries(
    query_ids: List[str],
    dataset: SearchDataset,
    toolset: ToolSet,
    search_tool: Any,
    text_token_counter,
    policy,
    max_turns: int,
    parallel: int,
    partial_output: Path | None = None,
) -> List[Dict]:
    sem = asyncio.Semaphore(parallel)
    write_lock = asyncio.Lock()
    completed = 0

    async def bounded(qid: str) -> Dict:
        nonlocal completed
        async with sem:
            result = await eval_single_query(
                qid,
                dataset,
                toolset,
                search_tool,
                text_token_counter,
                policy,
                max_turns,
            )
        if partial_output is not None:
            async with write_lock:
                completed += 1
                partial_output.parent.mkdir(parents=True, exist_ok=True)
                with partial_output.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, default=str) + "\n")
                logger.info(
                    "partial_result_saved",
                    path=str(partial_output),
                    completed=completed,
                    total=len(query_ids),
                    qid=qid,
                )
        return result

    return list(await asyncio.gather(*(bounded(qid) for qid in query_ids)))


def summarize_results(results: List[Dict]) -> Dict:
    n = len(results)

    def mean(key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in results) / max(n, 1)

    return {
        "n": n,
        "errors": sum(1 for r in results if r.get("error")),
        "recall": mean("recall"),
        "trajectory_recall": mean("trajectory_recall"),
        "final_answer_recall": mean("final_answer_recall"),
        "precision": mean("precision"),
        "reward": mean("reward"),
        "turns": mean("turns"),
        "n_curated": mean("n_curated"),
        "n_pool": mean("n_pool"),
        "total_curate_calls": mean("total_curate_calls"),
        "policy_curate_prompts": mean("policy_curate_prompts"),
        "intent_checkpoint_prompts": mean("intent_checkpoint_prompts"),
        "checkpoint_curate_prompts": mean("checkpoint_curate_prompts"),
        "runtime_intent_revisions": mean("runtime_intent_revisions"),
        "relevance_judge_calls": mean("relevance_judge_calls"),
        "summary_auditor_calls": mean("summary_auditor_calls"),
        "summary_answer_ready": mean("summary_answer_ready"),
        "curated_with_intent": mean("curated_with_intent"),
        "answer_em": mean("answer_em"),
        "answer_f1": mean("answer_f1"),
    }


def print_results_table(name: str, results: List[Dict]) -> None:
    summary = summarize_results(results)
    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(f"{'=' * 80}")
    print(f"  n: {summary['n']}  errors: {summary['errors']}")
    print(f"  Recall:              {summary['recall']:.4f}")
    print(f"  Trajectory Recall:   {summary['trajectory_recall']:.4f}")
    print(f"  Final-Answer Recall: {summary['final_answer_recall']:.4f}")
    print(f"  Precision:           {summary['precision']:.4f}")
    print(f"  Reward:              {summary['reward']:.4f}")
    if any("answer_em" in row for row in results):
        print(f"  Answer EM:           {summary['answer_em']:.4f}")
        print(f"  Answer F1:           {summary['answer_f1']:.4f}")
    print(f"  Turns:               {summary['turns']:.2f}")
    print(f"{'=' * 80}\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="browsecompplus")
    parser.add_argument("--split", default="test", choices=["all", "test", "train", "rl"])
    parser.add_argument("--collection-split", default="test", choices=["test", "train", "rl"])
    parser.add_argument(
        "--n-queries",
        type=int,
        default=0,
        help="Number of queries to sample; 0 evaluates the complete selected split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-ids", nargs="*", default=None)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--fallback-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--api-key", default=os.environ.get("POLICY_MODEL_API_KEY", ""))
    parser.add_argument("--policy-protocol", choices=["vllm_tokens", "qwen_chat"], default="vllm_tokens")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--partial-output",
        default=None,
        help="Append one JSON line per completed query so interrupted runs keep progress.",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default="vllm",
        choices=["vllm", "none"],
        help="Reranker backend: vllm (local Qwen3-Reranker-8B) or none.",
    )
    args = parser.parse_args()

    config = get_config()
    tiktoken_enc = tiktoken.get_encoding("o200k_harmony")
    text_token_counter = lambda text: len(tiktoken_enc.encode(text))

    dataset = get_dataset(args.dataset, split=args.split)
    openai_client = config.get_openai_client()

    try:
        _reranker_backend = getattr(args, "reranker", "vllm")
        if _reranker_backend == "none":
            reranker = None
        else:
            from harness_search.rerank import VLLMQwen3Reranker

            reranker = VLLMQwen3Reranker(token_counter=text_token_counter, max_tokens=4096)
    except Exception:
        reranker = None

    if dataset.name == "longsealqa":
        # LongSeal carries a different 30-document corpus for every query. The
        # real toolset is therefore constructed inside eval_single_query.
        toolset = ToolSet(name="longseal_placeholder")
        search_tool = None
    else:
        collection_names = dataset.get_chroma_collections(split=args.collection_split)
        chroma_client = config.get_chroma_client()
        search_tool = SearchTool(
            chroma_client=chroma_client,
            openai_client=openai_client,
            chroma_collection_name=collection_names,
            reranker=reranker,
            snippet_max_chars=2048,
            display_limit=SEARCH_DISPLAY_LIMIT,
        )
        toolset = ToolSet(name=f"{args.dataset}_toolset")
        toolset.add_tool(search_tool)
        toolset.add_tool(
            GrepCorpusTool(
                chroma_client=chroma_client,
                chroma_collection_name=collection_names,
                token_counter=text_token_counter,
            )
        )
        toolset.add_tool(
            ReadTool(
                chroma_client=chroma_client,
                chroma_collection_name=collection_names,
                reranker=reranker,
                token_counter=text_token_counter,
                max_tokens=4096,
            )
        )
        toolset.add_tool(PruneChunksTool())

    if search_tool is None and dataset.name != "longsealqa":
        raise RuntimeError(f"search tool is missing for dataset {dataset.name}")

    if args.split == "all":
        all_qids = dataset.get_all_query_ids()
    elif args.split == "test":
        all_qids = dataset.get_test_query_ids()
    elif args.split == "rl":
        all_qids = dataset.get_rl_query_ids()
    else:
        all_qids = dataset.get_all_query_ids(split="train")

    if args.query_ids:
        known_qids = set(all_qids)
        query_ids = [qid for qid in args.query_ids if qid in known_qids]
        if not query_ids:
            raise ValueError("No valid query IDs remained after filtering")
    else:
        rng = random.Random(args.seed)
        # A non-positive limit means "evaluate the complete selected split".
        # The launch scripts intentionally default N_QUERIES to 0 for full runs.
        if args.n_queries <= 0:
            query_ids = list(all_qids)
        else:
            query_ids = rng.sample(all_qids, min(args.n_queries, len(all_qids)))

    if args.policy_protocol == "qwen_chat":
        if not args.api_key:
            raise ValueError("--api-key or POLICY_MODEL_API_KEY is required for qwen_chat")
        policy = ChatRetrievalPolicy(
            base_url=args.base_url, model=args.model,
            fallback_model=args.fallback_model, api_key=args.api_key,
            max_tokens=args.max_tokens, temperature=args.temperature,
            top_p=args.top_p, timeout=args.timeout,
        )
    else:
        policy = VllmRetrievalPolicy(
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
        )

    logger.info(
        "evaluating_vllm",
        model=args.model,
        base_url=args.base_url,
        n=len(query_ids),
        parallel=args.parallel,
    )
    results = await eval_queries(
        query_ids=query_ids,
        dataset=dataset,
        toolset=toolset,
        search_tool=search_tool,
        text_token_counter=text_token_counter,
        policy=policy,
        max_turns=args.max_turns,
        parallel=args.parallel,
        partial_output=Path(args.partial_output) if args.partial_output else None,
    )
    print_results_table(args.model, results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            args.model: [
                {
                    k: v
                    for k, v in r.items()
                    if isinstance(v, (int, float, str, bool, list, dict))
                }
                for r in results
            ],
            "_summary": summarize_results(results),
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("results_saved", path=str(output_path))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    asyncio.run(main())
