"""Tinker adapter, context rendering, rewards and rollout utilities."""
# Allow direct execution from subdirectories while keeping imports package-relative.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))



import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import structlog
import tinker
from openai_harmony import Conversation, HarmonyEncodingName, Message, Role, load_harmony_encoding
from tinker_cookbook.rl.types import (
    Env,
    Observation as TinkerObservation,
    StopCondition,
    Action as TinkerAction,
    StepResult,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    Trajectory as TinkerTrajectory,
    Transition as TinkerTransition,
    TrajectoryGroup,
    Metrics,
)

from harness_search.agent import TinkerAgentInferenceModel
from datagen.search_dataset import SearchDataset, get_dataset
from harness_search.rerank import VLLMQwen3Reranker, Reranker
from harness_search.trajectory import Action, Observation, ObservationBuilder
from harness_search.tools import (
    Tool,
    ToolSet,
    ToolSchema,
    SearchTool,
    GrepCorpusTool,
    ReadTool,
    PruneChunksTool,
    UserTextTool,
)

from harness_search.ultra_core import (
    WorkingMemorySnapshot,
    build_result_summary,
    compute_reward,
    get_system_prompt,
    render_context_within_budget,
    FAN_OUT_SEARCH_SCHEMA,
    CURATE_SCHEMA,
    END_SCHEMA,
    REVIEW_DOCS_SCHEMA,
    REDIRECT_SCHEMA,
    build_rerank_instruction,
    V8D_INTENT_STATE_TRACKING,
    FORMAT_RETRY_PROMPT,
    FORMAT_ERROR_PENALTY,
    RECENT_K,
    SEARCH_DISPLAY_LIMIT,
    WINDOW_SIZE,
    MAX_WINDOWS,
    CONSEC_SEARCH_PENALTY,
    MAX_CONSEC_BEFORE_PENALTY,
    CONSEC_SEARCH_PENALTY_CAP,
    MAX_TURNS,
)

from harness_search.memory_operator import MemoryOperator
from harness_search.model_settings import V3_JUDGES_ENABLED
from harness_search.summary_auditor import SummaryAuditor
from harness_search.actions import METHOD_ACTIONS, canonical_tool_name
from harness_search import ActionProposal, HarnessBudget, HarnessSearch

logger = structlog.get_logger("harness_search.runtime")

# Save trajectory details for debugging
SAVE_TRAJECTORIES = os.environ.get("SAVE_TRAJECTORIES", "1") == "1"
TRAJECTORY_SAVE_PATH = os.environ.get("TRAJECTORY_SAVE_PATH", None)
# Compatibility: existing launch scripts configure S_max through this name.
INTENT_SEARCHES_PER_STAGE = int(os.environ.get("INTENT_SEARCHES_PER_STAGE", "5"))

# Optional per-curate recall-delta shaping.
# v8d default is terminal-reward-centric, so keep this OFF unless explicitly enabled.
DELTA_RECALL_BONUS = float(os.environ.get("DELTA_RECALL_BONUS", "0.0"))
# Reward mode: terminal-only by default (Context-1 style).
# When enabled, per-turn snapshots are not used for window rewards.
USE_TERMINAL_ONLY_REWARD = os.environ.get("USE_TERMINAL_ONLY_REWARD", "1") == "1"
# Rollout mode: full-trajectory by default (Context-1 style).
# Set USE_WINDOW_SLICING=1 to enable legacy window-slicing training.
USE_WINDOW_SLICING = os.environ.get("USE_WINDOW_SLICING", "0") == "1"
# Optional debug gate: restrict training to a fixed set of query IDs.
# Example: FORCE_QUERY_IDS="1029,579,751,605,638"
FORCE_QUERY_IDS_RAW = os.environ.get("FORCE_QUERY_IDS", "").strip()
FORCE_QUERY_IDS = {
    q.strip() for q in FORCE_QUERY_IDS_RAW.split(",") if q.strip()
}


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Stubs (for toolset registration — dispatch handled by env)
# ═══════════════════════════════════════════════════════════════════════════════



class FanOutSearchTool(Tool):
    tool_schema: ToolSchema
    def __init__(self):
        super().__init__(tool_schema=FAN_OUT_SEARCH_SCHEMA)
    def __call__(self, params, overrides=None):
        raise NotImplementedError("Handled by env")


class CurateTool(Tool):
    tool_schema: ToolSchema
    def __init__(self):
        super().__init__(tool_schema=CURATE_SCHEMA)
    def __call__(self, params, overrides=None):
        raise NotImplementedError("Handled by env")


class EndTool(Tool):
    tool_schema: ToolSchema
    def __init__(self):
        super().__init__(tool_schema=END_SCHEMA)
    def __call__(self, params, overrides=None):
        return "Search concluded.", None


class ReviewDocsTool(Tool):
    tool_schema: ToolSchema
    def __init__(self):
        super().__init__(tool_schema=REVIEW_DOCS_SCHEMA)
    def __call__(self, params, overrides=None):
        raise NotImplementedError("Handled by env")


class RedirectTool(Tool):
    """Meta-search tool backed by a separate local intent model."""
    tool_schema: ToolSchema
    def __init__(self):
        super().__init__(tool_schema=REDIRECT_SCHEMA)
    def __call__(self, params, overrides=None):
        raise NotImplementedError("Handled by env")


# ═══════════════════════════════════════════════════════════════════════════════
# RL Environment
# ═══════════════════════════════════════════════════════════════════════════════

class HarnessSearchEnv(Env):
    """Tinker/Harmony adapter for the provider-independent HarnessSearch loop."""

    def __init__(
        self,
        toolset: ToolSet,
        search_tool: SearchTool,
        query_id: str,
        query_text: str,
        dataset: SearchDataset,
        text_token_counter: Optional[Callable[[str], int]] = None,
        max_turns: int = MAX_TURNS,
        rollout_idx: int = 0,
    ):
        self.toolset = toolset
        self.search_tool = search_tool
        self.query_id = query_id
        self.query_text = query_text
        self.dataset = dataset
        self.text_token_counter = text_token_counter
        self.max_turns = max_turns
        self.rollout_idx = rollout_idx

        self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_condition: StopCondition = [200002, 200012]

        self._normalize_ids = (
            getattr(dataset, "evaluation_mode", "document") == "document"
        )

        self.memory_operator = MemoryOperator(
            query_id=query_id, query_text=query_text, toolset=toolset,
            search_tool=search_tool, normalize_ids=self._normalize_ids,
            text_token_counter=text_token_counter,
        )
        self.summary_auditor = SummaryAuditor(query_id)
        self.harness = HarnessSearch(
            self.memory_operator, self.summary_auditor,
            HarnessBudget(max_turns=max_turns, searches_per_curation=INTENT_SEARCHES_PER_STAGE),
            audit_enabled=V3_JUDGES_ENABLED,
        )
        self.system_prompt = get_system_prompt(query_text, max_turns=max_turns)

        self._all_actions: List[Action] = []
        self._all_observations: List[Observation] = []
        self._wm_snapshots: List[WorkingMemorySnapshot] = []
        self._result_summaries: List[str] = []
        self._reward_at_turn: List[float] = []


        self._terminal_reward: float = 0.0
        self._terminal_metrics: Metrics = {}
        self._episode_ended: bool = False
        self._current_turn: int = 0
        self._format_retries: int = 0
        self._turns_since_curate: int = 0
        self._total_curate_calls: int = 0
        self._tool_types_used: Set[str] = set()
        self._prev_recall: float = 0.0
        self._curate_recall_deltas: List[float] = []
        # v8d: track token usage for [Context: X/Y] marker, dataset name for
        # adaptive rerank instruction, and first-search auto-populate state.
        self._approx_prompt_tokens: int = 0
        self._dataset_name: str = getattr(dataset, "name", "web")
        self._openai_client = None  # lazily acquired when needed
        # Build rerank instruction once per episode (cheap if LLM path disabled)
        self.memory_operator.configure(rerank_instruction=build_rerank_instruction(
            query=query_text,
            dataset_name=self._dataset_name,
            openai_client=None,
            use_llm=False,
        ))

        # Pre-compute gold data for fast per-curate recall checks
        self._gold_fact_chunk_sets: Optional[List[Set[str]]] = None
        self._gold_doc_ids: Optional[Set[str]] = None
        self._eval_mode = getattr(dataset, "evaluation_mode", "document")
        try:
            qi = dataset._query_index.get(query_id, {})
            if self._eval_mode == "fact":
                self._gold_fact_chunk_sets = [
                    set(f["chunk_ids"]) for f in qi.get("document_ids", [])
                ]
                # Keep a flattened view for generic evaluation diagnostics.
                # Fact-level reward/recall still uses _gold_fact_chunk_sets.
                self._gold_doc_ids = set().union(*self._gold_fact_chunk_sets)
            else:
                self._gold_doc_ids = set(qi.get("document_ids", []))
        except Exception:
            pass

    @property
    def wm(self):
        """Detached read access; all memory writes belong to MemoryOperator."""
        return self.memory_operator.view

    # ── Environment Interface ──────────────────────────────────────────────

    async def initial_observation(self) -> Tuple[TinkerObservation, StopCondition]:
        initial_result = await asyncio.to_thread(self.harness.initialize)
        logger.info("initial_intent", qid=self.query_id, result=initial_result[:300])
        self._wm_snapshots.append(self.wm.snapshot())

        tokens = render_context_within_budget(
            system_prompt=self.system_prompt,
            wm_text=self.wm.to_text(),
            recent_actions=[],
            recent_observations=[],
            result_summaries=None,
            enc=self.enc,
            nudge_prompt=self.harness.action_hint(),
            allowed_tools=self.harness.allowed_tools,
        )
        return tinker.ModelInput.from_ints(tokens), self.stop_condition

    async def step(self, action_tokens: TinkerAction) -> StepResult:
        full_toolset = self._build_full_toolset()

        # Parse action tokens
        try:
            action = TinkerAgentInferenceModel.harmony_tinker_tokens_to_action(
                self.enc, action_tokens, full_toolset,
            )
        except Exception as e:
            try:
                raw_action = self.enc.decode(list(action_tokens))
            except Exception:
                raw_action = "<decode unavailable>"
            # Some local gpt-oss/vLLM releases occasionally render the Harmony
            # recipient as literal header text ("to=functions.NAME") instead
            # of the recipient special token. Recover the final tool call and
            # render it back through openai-harmony before treating the turn as
            # a format failure. Selecting the final occurrence also handles a
            # response that emitted an extra reasoning message first.
            repaired_action = None
            matches = list(re.finditer(r"to=functions\.([A-Za-z_][A-Za-z0-9_]*)", raw_action))
            if matches:
                match = matches[-1]
                tool_name = canonical_tool_name(match.group(1))
                if tool_name in full_toolset.tools:
                    tail = raw_action[match.end():]
                    marker = tail.rfind("<|message|>")
                    if marker >= 0:
                        payload_text = tail[marker + len("<|message|>"):]
                    else:
                        payload_text = tail
                    payload_text = re.split(
                        r"<\|(?:call|end|return|start)\|>", payload_text, maxsplit=1
                    )[0].strip()
                    try:
                        params = json.loads(payload_text)
                        if isinstance(params, dict):
                            message = (
                                Message.from_role_and_content(
                                    Role.ASSISTANT, json.dumps(params)
                                )
                                .with_channel("commentary")
                                .with_recipient("functions." + tool_name)
                                .with_content_type("<|constrain|>json")
                            )
                            repaired_tokens = self.enc.render_conversation(
                                Conversation.from_messages([message])
                            )
                            repaired_action = (
                                TinkerAgentInferenceModel.harmony_tinker_tokens_to_action(
                                    self.enc, repaired_tokens, full_toolset,
                                )
                            )
                            logger.info(
                                "format_repaired_literal_recipient",
                                qid=self.query_id,
                                tool=tool_name,
                            )
                    except Exception as repair_exc:
                        logger.warning(
                            "format_repair_failed",
                            qid=self.query_id,
                            tool=tool_name,
                            error=str(repair_exc)[:240],
                            payload=payload_text[:1000],
                        )
            if repaired_action is not None:
                action = repaired_action
            else:
                logger.warning(
                    "format_error_raw_action",
                    qid=self.query_id,
                    error=str(e),
                    raw_action=raw_action[:8000],
                )
                return self._handle_format_error(str(e))

        if (len(action.tools) != 1 or len(action.params) != 1 or len(action.sources) != 1
                or isinstance(action.tools[0], UserTextTool)):
            return self._handle_format_error("Propose exactly one tool operation per turn")

        tool = action.tools[0]
        name = canonical_tool_name(tool.tool_schema.name)
        pool_size_before = self.wm.get_pool_size()
        result = await asyncio.to_thread(
            self.harness.step, ActionProposal(name, action.params[0]),
        )
        builder = ObservationBuilder()
        builder.add_observation(result.text, source=action.sources[0], tool_metadata=result.metadata)
        observation = builder.build()
        self._all_actions.append(action)
        self._all_observations.append(observation)
        self._current_turn = self.harness.turns
        self._total_curate_calls = self.harness.curate_calls
        self._format_retries = 0
        has_curate = name == "curate" and result.committed
        self._turns_since_curate = 0 if has_curate else self._turns_since_curate + 1
        if result.committed:
            self._tool_types_used.add(name)
        self._wm_snapshots.append(self.wm.snapshot())
        self._result_summaries.append(build_result_summary(
            obs_text=result.text, tool_names=[name] if result.committed else [],
            wm=self.wm, turns_since_curate=self._turns_since_curate,
            tool_types_used=self._tool_types_used, current_turn=self._current_turn,
            pool_size_before=pool_size_before,
        ) + "\n" + self.harness.action_hint())

        if not USE_TERMINAL_ONLY_REWARD:
            turn_reward = self._compute_reward_at_current_state()
            if (not V8D_INTENT_STATE_TRACKING and not has_curate
                    and self._turns_since_curate > MAX_CONSEC_BEFORE_PENALTY
                    and self.wm.get_pool_size() > 0):
                turn_reward -= min(
                    CONSEC_SEARCH_PENALTY * (self._turns_since_curate - MAX_CONSEC_BEFORE_PENALTY),
                    CONSEC_SEARCH_PENALTY_CAP,
                )
            if has_curate:
                recall = self._fast_recall(self.wm.curated_ids)
                delta = recall - self._prev_recall
                self._curate_recall_deltas.append(delta)
                turn_reward += DELTA_RECALL_BONUS * delta
                self._prev_recall = recall
            self._reward_at_turn.append(turn_reward)
        return self._operation_step_result(result)

    @property
    def required_action(self):
        return self.harness.required_action

    def policy_toolset(self):
        """Expose the current admissible tools to a policy adapter."""
        toolset = self._build_full_toolset()
        toolset.tools = {name: tool for name, tool in toolset.tools.items()
                         if name in self.harness.allowed_tools}
        return toolset

    def _operation_step_result(self, result):
        self._current_turn = self.harness.turns
        if result.episode_done:
            self._terminal_reward, self._terminal_metrics = self._compute_terminal_reward()
            self._terminal_metrics.update({
                "max_turns_reached": float(result.termination_reason == "max_turns"),
                "audit_accepted": float(result.termination_reason == "answer_ready"),
                "audit_rejections": float(self.harness.audit_rejections),
            })
            if result.action == "invalid":
                self._terminal_reward = FORMAT_ERROR_PENALTY
                self._terminal_metrics.update({
                    "no_error": 0.0, "format_error": 1.0,
                    "final_reward": FORMAT_ERROR_PENALTY,
                })
            self._episode_ended = True
            self._save_trajectory()
            return StepResult(
                reward=self._terminal_reward, episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition, metrics=self._terminal_metrics,
            )
        tokens = (self._render_next_context() if result.committed else
                  self._render_retry_context(result.text))
        return StepResult(
            reward=0.0, episode_done=False,
            next_observation=tinker.ModelInput.from_ints(tokens),
            next_stop_condition=self.stop_condition,
        )

    # ── Context Rendering (single pathway via ultra_core) ──────────────────

    def _render_next_context(self) -> List[int]:
        """Render context for the next turn using render_context_within_budget."""
        # Render current committed memory so an audit gap survives history trimming.
        wm_text = self.wm.to_text()
        recent_actions = self._all_actions[-RECENT_K:]
        recent_obs = self._all_observations[-RECENT_K:]
        recent_summaries = self._result_summaries[-RECENT_K:]
        nudge = self.harness.action_hint()

        tokens = render_context_within_budget(
            system_prompt=self.system_prompt,
            wm_text=wm_text,
            recent_actions=recent_actions,
            recent_observations=recent_obs,
            result_summaries=recent_summaries,
            enc=self.enc,
            nudge_prompt=nudge,
            allowed_tools=self.harness.allowed_tools,
        )
        # v8d: stash size so the next tool output can append an accurate marker.
        self._approx_prompt_tokens = len(tokens)
        self.memory_operator.set_prompt_tokens(len(tokens))
        return tokens

    def _render_retry_context(
        self, retry_prompt: str = FORMAT_RETRY_PROMPT,
    ) -> List[int]:
        """Re-render current context with retry prompt appended."""
        wm_text = self.wm.to_text()
        recent_actions = self._all_actions[-RECENT_K:]
        recent_obs = self._all_observations[-RECENT_K:]
        recent_summaries = self._result_summaries[-RECENT_K:]
        retry_prompt += "\n" + self.harness.action_hint()

        return render_context_within_budget(
            system_prompt=self.system_prompt,
            wm_text=wm_text,
            recent_actions=recent_actions,
            recent_observations=recent_obs,
            result_summaries=recent_summaries,
            enc=self.enc,
            retry_prompt=retry_prompt,
            allowed_tools=self.harness.allowed_tools,
        )

    # ── Format Error Handling ──────────────────────────────────────────────

    def _handle_format_error(self, error_msg: str) -> StepResult:
        self._format_retries += 1
        logger.warning("format_retry", qid=self.query_id, error=error_msg[:200])
        return self._operation_step_result(self.harness.reject(error_msg))

    # ── Tool schemas; dispatch and transitions belong to the harness ────────

    def _build_full_toolset(self) -> ToolSet:
        ts = ToolSet(name="harness_search_toolset")
        ts.tools.update({name: tool for name, tool in self.toolset.tools.items()
                         if name in METHOD_ACTIONS})
        ts.tools.update({
            "fan_out_search": FanOutSearchTool(), "curate": CurateTool(),
            "end": EndTool(), "review_docs": ReviewDocsTool(), "redirect": RedirectTool(),
        })
        return ts

    # ── Reward ─────────────────────────────────────────────────────────────

    def _fast_recall(self, curated_ids: List[str]) -> float:
        """Lightweight recall using pre-computed gold data. Avoids repeated
        _query_index lookups and set constructions on the hot path."""
        if not curated_ids:
            return 0.0
        if self._eval_mode == "fact" and self._gold_fact_chunk_sets is not None:
            if not self._gold_fact_chunk_sets:
                return 0.0
            curated_set = set(curated_ids)
            found = sum(
                1 for gset in self._gold_fact_chunk_sets
                if gset & curated_set
            )
            return found / len(self._gold_fact_chunk_sets)
        if self._gold_doc_ids is not None:
            if not self._gold_doc_ids:
                return 0.0
            # curated_ids are already normalized to doc IDs by WorkingMemory,
            # so no further rsplit needed (avoids double-stripping doc IDs
            # that contain underscores, e.g. web datasets).
            found = len(set(curated_ids) & self._gold_doc_ids)
            return found / len(self._gold_doc_ids)
        return self.dataset.evaluate_results_recall(self.query_id, curated_ids)

    def _evaluate_and_compute_reward(self, is_terminal: bool) -> Tuple[float, Metrics]:
        """Evaluate recall/precision via dataset, then delegate to compute_reward."""
        curated = self.wm.curated_ids
        pool = self.wm.pool_ids

        if curated:
            recall = self.dataset.evaluate_results_recall(self.query_id, curated)
            precision = self.dataset.evaluate_results_precision(self.query_id, curated)
            fa_recall = self.dataset.evaluate_results_final_answer_recall(self.query_id, curated)
        else:
            recall = precision = fa_recall = 0.0

        traj_recall = (
            self.dataset.evaluate_results_recall(self.query_id, pool)
            if pool else 0.0
        )
        traj_fa_recall = (
            self.dataset.evaluate_results_final_answer_recall(self.query_id, pool)
            if pool else 0.0
        )

        return compute_reward(
            recall=recall,
            precision=precision,
            final_answer_recall=fa_recall,
            trajectory_recall=traj_recall,
            n_curated=len(curated),
            turn=self._current_turn,
            total_curate_calls=self._total_curate_calls,
            n_unique_tools=len(self._tool_types_used - {"end"}),
            is_terminal=is_terminal,
            trajectory_fa_recall=traj_fa_recall,
        )

    def _compute_reward_at_current_state(self) -> float:
        reward, _ = self._evaluate_and_compute_reward(is_terminal=False)
        return reward

    def _compute_terminal_reward(self) -> Tuple[float, Metrics]:
        reward, metrics = self._evaluate_and_compute_reward(is_terminal=True)
        metrics.update({
            "relevance_judge_calls": float(self.memory_operator.relevance_judge.calls),
            "summary_auditor_calls": float(self.summary_auditor.calls),
            "summary_answer_ready": float(self.summary_auditor.answer_ready),
            "curated_with_intent": float(sum(
                1 for doc_id in self.wm.curated_ids if self.wm.curated_notes.get(doc_id)
            )),
        })
        return reward, metrics

    # ── Trajectory Saving ──────────────────────────────────────────────────

    def _save_trajectory(self) -> None:
        if not SAVE_TRAJECTORIES:
            return
        try:
            save_dir = TRAJECTORY_SAVE_PATH or os.environ.get("LOG_PATH", "./tmp/rl_ultra_v3")
            save_dir = os.path.join(save_dir, "trajectories")
            os.makedirs(save_dir, exist_ok=True)

            record = {
                "query_id": self.query_id,
                "dataset": self.dataset.name,
                "normalize_ids": self._normalize_ids,
                "reward": self._terminal_reward,
                "metrics": {
                    k: v for k, v in self._terminal_metrics.items()
                    if isinstance(v, (int, float, str, bool))
                },
                "turns": self._current_turn,
                "termination_reason": self.harness.termination_reason,
                "curated_ids": self.wm.curated_ids,
                # Persist v8d per-doc tags so downstream analysis can filter
                # to high-confidence subsets (e.g., very_high/high only).
                "curated_importance": dict(self.wm.curated_importance),
                "curated_intents": dict(self.wm.curated_notes),
                "relevance_annotations": dict(self.memory_operator.relevance_annotations),
                "latest_audit": self.summary_auditor.latest,
                "pool_ids": list(self.wm.pool_ids),
                "pool_size": len(self.wm.pool_ids),
                "reward_at_turn": [round(r, 4) for r in self._reward_at_turn],
                "search_history": self.wm.search_history,
                "curate_recall_deltas": [round(d, 4) for d in self._curate_recall_deltas],
            }
            save_file = os.path.join(save_dir, "episodes.jsonl")
            with open(save_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning("save_error", error=str(e)[:200])


# ═══════════════════════════════════════════════════════════════════════════════
# Window Slicing
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WindowSlice:
    trajectory: TinkerTrajectory
    reward: float
    stage: int  # start_turn // window_size — used for normalization bucketing


def slice_trajectory_into_windows(
    trajectory: TinkerTrajectory,
    terminal_reward: float,
    reward_at_turn: Optional[List[float]] = None,
    window_size: int = WINDOW_SIZE,
    max_windows: int = MAX_WINDOWS,
) -> List[WindowSlice]:
    """Slice a full-episode trajectory into up to *max_windows* non-overlapping
    windows, evenly distributed across the episode.  The last window always
    covers the final turn so the terminal reward signal is never lost.

    Each window gets the reward at its END turn (curated set quality at that
    point).  Per-transition rewards are zeroed to avoid double-counting with
    final_rewards_G.

    Returns WindowSlice objects carrying a ``stage`` field (start_turn //
    window_size) so callers can normalize rewards across rollouts by episode
    stage rather than ordinal window index.
    """
    transitions = trajectory.transitions
    n = len(transitions)

    if n == 0:
        return []

    def _zero_rewards(ts):
        return [
            TinkerTransition(
                ob=t.ob, ac=t.ac, reward=0.0,
                episode_done=t.episode_done, metrics=t.metrics, logs=t.logs,
            )
            for t in ts
        ]

    def _make_window(start: int, end: int) -> WindowSlice:
        window_transitions = _zero_rewards(transitions[start:end])
        final_ob = transitions[end].ob if end < n else trajectory.final_ob
        if reward_at_turn is not None and end - 1 < len(reward_at_turn):
            window_reward = reward_at_turn[end - 1]
        else:
            window_reward = terminal_reward
        return WindowSlice(
            trajectory=TinkerTrajectory(transitions=window_transitions, final_ob=final_ob),
            reward=window_reward,
            stage=start // window_size,
        )

    if n <= window_size:
        return [_make_window(0, n)]

    # Build all non-overlapping windows first
    all_starts: List[int] = []
    start = 0
    while start < n:
        all_starts.append(start)
        start += window_size
        if start + 1 >= n:
            break

    if len(all_starts) <= max_windows:
        return [
            _make_window(s, min(s + window_size, n)) for s in all_starts
        ]

    # Too many windows — subsample to max_windows.
    # Always keep the last window; pick (max_windows - 1) evenly from the rest.
    last_idx = len(all_starts) - 1
    rest_indices = list(range(last_idx))
    num_pick = max_windows - 1

    if num_pick <= 0:
        picked = []
    elif num_pick >= len(rest_indices):
        picked = rest_indices
    else:
        picked = [
            rest_indices[round(i * (len(rest_indices) - 1) / (num_pick - 1))]
            for i in range(num_pick)
        ]
        seen: set = set()
        unique: List[int] = []
        for idx in picked:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)
        picked = unique

    selected_indices = picked + [last_idx]
    return [
        _make_window(all_starts[i], min(all_starts[i] + window_size, n))
        for i in selected_indices
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Group Builder + Rollout
# ═══════════════════════════════════════════════════════════════════════════════

class SlidingWindowGroupBuilder(EnvGroupBuilder):
    def __init__(
        self,
        query_id: str,
        query_text: str,
        dataset: SearchDataset,
        toolset: ToolSet,
        search_tool: SearchTool,
        group_size: int,
        text_token_counter: Optional[Callable[[str], int]] = None,
    ):
        self.query_id = query_id
        self.query_text = query_text
        self.dataset = dataset
        self.toolset = toolset
        self.search_tool = search_tool
        self.group_size = group_size
        self.text_token_counter = text_token_counter
        self._envs: List[HarnessSearchEnv] = []

    async def make_envs(self) -> Sequence[HarnessSearchEnv]:
        self._envs = [
            HarnessSearchEnv(
                toolset=self.toolset,
                search_tool=self.search_tool,
                query_id=self.query_id,
                query_text=self.query_text,
                dataset=self.dataset,
                text_token_counter=self.text_token_counter,
                rollout_idx=i,
            )
            for i in range(self.group_size)
        ]
        return self._envs

    async def compute_group_rewards(
        self, trajectory_group: List[TinkerTrajectory], env_group: Sequence[Env],
    ) -> List[Tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> List[str]:
        return [self.dataset.name, "search"]


async def do_group_rollout_with_windows(
    env_group_builder: SlidingWindowGroupBuilder,
    policy,
) -> TrajectoryGroup:
    """Run full rollouts, then slice windows and position-normalize rewards."""
    from tinker_cookbook.rl.rollouts import do_single_rollout

    envs = await env_group_builder.make_envs()
    full_trajectories = await asyncio.gather(
        *[do_single_rollout(policy, env) for env in envs]
    )

    terminal_rewards = [env._terminal_reward for env in envs]
    terminal_metrics = [env._terminal_metrics for env in envs]
    per_turn_rewards = [env._reward_at_turn for env in envs]

    # Error = format error only (negative terminal reward from parse failure).
    # No-curate episodes (terminal reward = -0.2) are NOT errors — they're
    # valid episodes that should participate in position normalization.
    is_format_error = [env._terminal_reward == FORMAT_ERROR_PENALTY for env in envs]

    # Slice into windows
    all_rollout_windows: List[List[WindowSlice]] = []
    for traj, reward, turn_rewards in zip(
        full_trajectories, terminal_rewards, per_turn_rewards,
    ):
        windows = slice_trajectory_into_windows(
            traj,
            reward,
            reward_at_turn=None if USE_TERMINAL_ONLY_REWARD else turn_rewards,
        )
        all_rollout_windows.append(windows)

    max_windows = max(len(w) for w in all_rollout_windows) if all_rollout_windows else 0

    # Normalize rewards using two kinds of groups:
    #   "final"  — last window of every rollout (the terminal‐reward signal,
    #              always compared across rollouts regardless of episode length)
    #   stage N  — non‐final windows bucketed by episode stage
    #              (start_turn // window_size) so windows covering similar
    #              turns are compared even when subsampling differs.
    FINAL_KEY = -1  # sentinel for the "final" normalization group
    group_rewards: Dict[int, List[float]] = {}
    for rollout_idx, windows in enumerate(all_rollout_windows):
        if is_format_error[rollout_idx]:
            continue
        for i, ws in enumerate(windows):
            key = FINAL_KEY if i == len(windows) - 1 else ws.stage
            group_rewards.setdefault(key, []).append(ws.reward)

    group_means: Dict[int, float] = {
        k: sum(rs) / len(rs) for k, rs in group_rewards.items()
    } if group_rewards else {}

    all_means = list(group_means.values())
    global_mean = sum(all_means) / len(all_means) if all_means else 0.0

    all_window_trajectories: List[TinkerTrajectory] = []
    all_window_rewards: List[float] = []
    all_window_metrics: List[Metrics] = []

    for rollout_idx, windows in enumerate(all_rollout_windows):
        for i, ws in enumerate(windows):
            if is_format_error[rollout_idx]:
                normalized = ws.reward
            else:
                key = FINAL_KEY if i == len(windows) - 1 else ws.stage
                if key in group_means:
                    normalized = ws.reward - group_means[key] + global_mean
                else:
                    normalized = ws.reward
            all_window_trajectories.append(ws.trajectory)
            all_window_rewards.append(normalized)
            all_window_metrics.append(terminal_metrics[rollout_idx])

    stage_log = {
        ("final" if k == FINAL_KEY else f"stg{k}"): round(m, 3)
        for k, m in sorted(group_means.items())
    }
    logger.info(
        "windows_created",
        query_id=env_group_builder.query_id,
        reward_mode="terminal_only" if USE_TERMINAL_ONLY_REWARD else "per_turn_window",
        rollouts=len(envs),
        errors=sum(is_format_error),
        max_windows=max_windows,
        windows=[len(w) for w in all_rollout_windows],
        group_means=stage_log,
        terminal=[round(r, 3) for r in terminal_rewards],
    )

    return TrajectoryGroup(
        trajectories_G=all_window_trajectories,
        final_rewards_G=all_window_rewards,
        metrics_G=all_window_metrics,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetToolsetPair:
    dataset: SearchDataset
    toolset: ToolSet
    search_tool: SearchTool
    max_train_queries: Optional[int] = None

    @property
    def name(self) -> str:
        return self.dataset.name


class SlidingWindowRLDataset(RLDataset):
    def __init__(
        self,
        dataset_pairs: List[DatasetToolsetPair],
        batch_size: int,
        group_size: int,
        text_token_counter: Optional[Callable[[str], int]] = None,
        seed: int = 0,
        epochs: int = 1,
        query_split: str = "train",
    ):
        self.dataset_pairs = dataset_pairs
        self.batch_size = batch_size
        self.group_size = group_size
        self.text_token_counter = text_token_counter
        self.seed = seed
        self.epochs = epochs
        self.query_split = query_split

        import random as rng_module
        self._query_to_pair: Dict[str, DatasetToolsetPair] = {}
        self._base_query_ids: List[str] = []

        for pair in dataset_pairs:
            query_ids = pair.dataset.get_all_query_ids(split=self.query_split)

            # Optional debug mode for exact query-set parity checks.
            if FORCE_QUERY_IDS:
                before = len(query_ids)
                query_ids = [qid for qid in query_ids if str(qid) in FORCE_QUERY_IDS]
                logger.info(
                    "force_query_filter",
                    dataset=pair.name,
                    before=before,
                    after=len(query_ids),
                    requested=len(FORCE_QUERY_IDS),
                )

            if pair.max_train_queries and len(query_ids) > pair.max_train_queries:
                rng = rng_module.Random(seed)
                query_ids = rng.sample(query_ids, pair.max_train_queries)
            for qid in query_ids:
                unique_qid = f"{pair.name}::{qid}"
                self._query_to_pair[unique_qid] = pair
                self._base_query_ids.append(unique_qid)

        if not self._base_query_ids:
            raise ValueError(
                "No training queries selected. Check TRAIN_DATASETS / FORCE_QUERY_IDS."
            )

        self._batches_per_epoch = max(1, len(self._base_query_ids) // self.batch_size)
        self._epoch_query_ids: List[List[str]] = []
        for epoch in range(self.epochs):
            epoch_ids = self._base_query_ids.copy()
            rng = rng_module.Random(seed + epoch)
            rng.shuffle(epoch_ids)
            self._epoch_query_ids.append(epoch_ids)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        epoch = index // self._batches_per_epoch
        batch_in_epoch = index % self._batches_per_epoch
        query_ids = self._epoch_query_ids[epoch]
        start = batch_in_epoch * self.batch_size
        end = min(start + self.batch_size, len(query_ids))

        builders: List[EnvGroupBuilder] = []
        for unique_qid in query_ids[start:end]:
            pair = self._query_to_pair[unique_qid]
            _, original_qid = unique_qid.split("::", 1)
            _, query_text = pair.dataset.get_query_by_id(original_qid)
            builder = SlidingWindowGroupBuilder(
                query_id=original_qid,
                query_text=query_text,
                dataset=pair.dataset,
                toolset=pair.toolset,
                search_tool=pair.search_tool,
                group_size=self.group_size,
                text_token_counter=self.text_token_counter,
            )
            builders.append(builder)
        return builders

    def __len__(self) -> int:
        return self._batches_per_epoch * self.epochs


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import chz
    import tiktoken
    from tinker_cookbook.rl import train
    from harness_search.config import get_config

    DATASETS = os.environ.get("TRAIN_DATASETS", "browsecompplus").split(",")
    SIMPLE_DATASET_MAX_QUERIES = int(os.environ.get("SIMPLE_DATASET_MAX_QUERIES", "150"))
    # Backward compatible split handling:
    # - RL_QUERY_SPLIT controls which query IDs are sampled for rollouts.
    # - RL_COLLECTION_SPLIT controls which Chroma collections back tools.
    # - RL_DATA_SPLIT remains accepted as a legacy alias for RL_QUERY_SPLIT.
    RL_QUERY_SPLIT = (
        os.environ.get("RL_QUERY_SPLIT")
        or os.environ.get("RL_DATA_SPLIT")
        or "train"
    ).strip() or "train"
    RL_COLLECTION_SPLIT = (
        os.environ.get("RL_COLLECTION_SPLIT")
        or RL_QUERY_SPLIT
    ).strip() or RL_QUERY_SPLIT
    LOG_PATH = os.environ.get("LOG_PATH", "./tmp/rl_ultra_v3")
    LOAD_CHECKPOINT_PATH = os.environ.get("LOAD_CHECKPOINT_PATH", "")
    SFT_CHECKPOINT_PATH = os.environ.get("SFT_CHECKPOINT_PATH", "")
    TTL_SECONDS_RAW = os.environ.get("TTL_SECONDS", "none").strip().lower()
    if TTL_SECONDS_RAW in ("", "none", "null"):
        TTL_SECONDS: Optional[int] = None
    else:
        TTL_SECONDS = int(TTL_SECONDS_RAW)
    MODEL_NAME = "Qwen/Qwen3.5-27B"
    KL_PENALTY_COEF = float(os.environ.get("KL_PENALTY_COEF", "0.005"))

    # Guardrails for checkpoint type wiring:
    # - policy init should come from /weights/
    # - KL reference should come from /sampler_weights/
    if LOAD_CHECKPOINT_PATH and "/sampler_weights/" in LOAD_CHECKPOINT_PATH:
        inferred_policy_ckpt = LOAD_CHECKPOINT_PATH.replace(
            "/sampler_weights/", "/weights/"
        )
        logger.warning(
            "policy_checkpoint_looks_like_sampler_path",
            provided=LOAD_CHECKPOINT_PATH,
            inferred=inferred_policy_ckpt,
            action="using_inferred_weights_path",
        )
        LOAD_CHECKPOINT_PATH = inferred_policy_ckpt

    if SFT_CHECKPOINT_PATH and "/weights/" in SFT_CHECKPOINT_PATH:
        inferred_kl_ref = SFT_CHECKPOINT_PATH.replace(
            "/weights/", "/sampler_weights/"
        )
        logger.warning(
            "kl_reference_looks_like_weights_path",
            provided=SFT_CHECKPOINT_PATH,
            inferred=inferred_kl_ref,
            action="using_inferred_sampler_path",
        )
        SFT_CHECKPOINT_PATH = inferred_kl_ref

    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
    GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
    LEARNING_RATE = 1e-5
    TEMPERATURE = float(os.environ.get("ROLLOUT_TEMPERATURE", "1.0"))
    LOSS_FN = "cispo"
    LORA_RANK = 32
    MAX_TOKENS = 2048
    SEED = 42
    EPOCHS = int(os.environ.get("EPOCHS", "2"))
    THREAD_POOL_SIZE = 256
    NUM_SUBSTEPS = 4

    if os.environ.get("SMALL_SCALE_TEST"):
        BATCH_SIZE = 4
        GROUP_SIZE = 2
        EPOCHS = 1

    config = get_config()
    tiktoken_encoding = tiktoken.get_encoding("o200k_harmony")
    text_token_counter = lambda text: len(tiktoken_encoding.encode(text))

    # Reranker backend for search/read tools (v8d expects this on in full runs).
    reranker_backend = os.environ.get("RERANKER_BACKEND", "vllm").strip().lower()
    reranker_max_tokens = int(os.environ.get("RERANKER_MAX_TOKENS", "4096"))
    reranker: Optional[Reranker] = None
    if reranker_backend in ("", "none", "off", "disabled"):
        reranker = None
    elif reranker_backend == "vllm":
        reranker = VLLMQwen3Reranker(
            token_counter=text_token_counter,
            max_tokens=reranker_max_tokens,
        )
    else:
        raise ValueError(
            f"Unsupported RERANKER_BACKEND='{reranker_backend}'. "
            "Expected one of: none, vllm."
        )
    logger.info(
        "reranker_configured",
        backend=reranker_backend,
        enabled=bool(reranker),
        max_tokens=reranker_max_tokens,
    )
    logger.info(
        "checkpoint_wiring",
        policy_checkpoint=LOAD_CHECKPOINT_PATH or None,
        kl_reference_checkpoint=SFT_CHECKPOINT_PATH or None,
        ttl_seconds=TTL_SECONDS,
        kl_penalty_coef=KL_PENALTY_COEF,
    )

    dataset_pairs: List[DatasetToolsetPair] = []
    for dataset_name in DATASETS:
        dataset_name = dataset_name.strip()
        if not dataset_name:
            continue
        dataset = get_dataset(dataset_name)
        collection_names = dataset.get_chroma_collections(split=RL_COLLECTION_SPLIT)

        chroma_client = config.get_chroma_client()
        openai_client = config.get_openai_client()

        search_tool = SearchTool(
            chroma_client=chroma_client,
            openai_client=openai_client,
            chroma_collection_name=collection_names,
            reranker=reranker,
            snippet_max_chars=2048,
            display_limit=SEARCH_DISPLAY_LIMIT,
        )
        toolset = ToolSet(name=f"{dataset_name}_toolset")
        toolset.add_tool(search_tool)
        toolset.add_tool(GrepCorpusTool(
            chroma_client=chroma_client,
            chroma_collection_name=collection_names,
            token_counter=text_token_counter,
        ))
        toolset.add_tool(ReadTool(
            chroma_client=chroma_client,
            chroma_collection_name=collection_names,
            reranker=reranker,
            token_counter=text_token_counter,
            max_tokens=4096,
        ))
        toolset.add_tool(PruneChunksTool())

        max_q = SIMPLE_DATASET_MAX_QUERIES if dataset_name.endswith("_simple") else None
        pair = DatasetToolsetPair(
            dataset=dataset, toolset=toolset, search_tool=search_tool,
            max_train_queries=max_q,
        )
        dataset_pairs.append(pair)
        logger.info(
            "dataset_ready",
            dataset=dataset_name,
            query_split=RL_QUERY_SPLIT,
            collection_split=RL_COLLECTION_SPLIT,
            queries=len(dataset.get_all_query_ids(split=RL_QUERY_SPLIT)),
            max_train_queries=max_q,
        )

    rl_dataset = SlidingWindowRLDataset(
        dataset_pairs=dataset_pairs,
        batch_size=BATCH_SIZE,
        group_size=GROUP_SIZE,
        text_token_counter=text_token_counter,
        seed=SEED,
        epochs=EPOCHS,
        query_split=RL_QUERY_SPLIT,
    )
    logger.info(
        "dataset_created",
        batches=len(rl_dataset),
        batch_size=BATCH_SIZE,
        group_size=GROUP_SIZE,
        query_split=RL_QUERY_SPLIT,
        collection_split=RL_COLLECTION_SPLIT,
        rollout_temperature=TEMPERATURE,
        use_window_slicing=USE_WINDOW_SLICING,
        window_size=WINDOW_SIZE,
        max_windows=MAX_WINDOWS,
        recent_k=RECENT_K,
    )

    if USE_WINDOW_SLICING:
        # Optional legacy path: monkey-patch do_group_rollout for window slicing.
        import tinker_cookbook.rl.rollouts as rollouts_module
        _original_do_group_rollout = rollouts_module.do_group_rollout

        async def _patched_do_group_rollout(env_group_builder, policy):
            if isinstance(env_group_builder, SlidingWindowGroupBuilder):
                return await do_group_rollout_with_windows(env_group_builder, policy)
            return await _original_do_group_rollout(env_group_builder, policy)

        rollouts_module.do_group_rollout = _patched_do_group_rollout
        logger.info("patched_rollout", rollout_mode="window_slicing")
    else:
        # Default path: context-1 style full trajectories with terminal reward.
        logger.info("using_default_rollout", rollout_mode="full_trajectory")

    @chz.chz
    class UltraRLDatasetBuilder(RLDatasetBuilder):
        async def __call__(self) -> Tuple[RLDataset, Optional[RLDataset]]:
            return rl_dataset, None

    # KL reference is only needed when KL penalty is enabled.
    use_kl_reference = KL_PENALTY_COEF > 0 and bool(SFT_CHECKPOINT_PATH)
    if KL_PENALTY_COEF > 0 and not SFT_CHECKPOINT_PATH:
        logger.warning(
            "kl_penalty_enabled_without_reference",
            kl_penalty_coef=KL_PENALTY_COEF,
            message="KL penalty > 0 but SFT_CHECKPOINT_PATH is empty; training will fail.",
        )

    kl_config = None
    if use_kl_reference:
        kl_config = train.KLReferenceConfig(
            base_model=MODEL_NAME,
            load_checkpoint_path=SFT_CHECKPOINT_PATH,
        )

    blueprint = chz.Blueprint(train.Config).apply({
        "log_path": LOG_PATH,
        "load_checkpoint_path": LOAD_CHECKPOINT_PATH or None,
        "model_name": MODEL_NAME,
        "dataset_builder": UltraRLDatasetBuilder(),
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "num_substeps": NUM_SUBSTEPS,
        "loss_fn": LOSS_FN,
        "loss_fn_config": {"clip_low_threshold": 0, "clip_high_threshold": 5},
        "eval_every": 2,
        "save_every": 1,
        "ttl_seconds": TTL_SECONDS,
        "remove_constant_reward_groups": True,
        "kl_penalty_coef": KL_PENALTY_COEF,
        "kl_reference_config": kl_config,
        "compute_post_kl": bool(use_kl_reference),
    })

    cfg = blueprint.make()
    logger.info("starting_rl_v3", thread_pool_size=THREAD_POOL_SIZE)

    async def run():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
            loop.set_default_executor(executor)
            await train.main(cfg)

    asyncio.run(run())
