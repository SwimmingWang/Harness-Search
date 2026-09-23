"""Offline regression tests for authority boundaries and the original episode flow."""
import json
import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from harness_search.actions import LEGACY_TOOL_NAMES
from harness_search.agent import TinkerAgentInferenceModel
from harness_search.longseal_tools import create_longseal_toolset
from harness_search.memory_operator import MemoryOperator
from harness_search.memory_prompts import MEMORY_OPERATOR_SYSTEM_PROMPT
from harness_search.relevance_judge import RelevanceJudge
from harness_search.retrieval_policy import ChatRetrievalPolicy, VllmRetrievalPolicy
from harness_search.summary_auditor import SummaryAuditor
from harness_search.tools import SearchToolCallMetadata
from runtime.environment import HarnessSearchEnv


DOCUMENTS = [
    {"url": "doc_a", "title": "Bridge", "text": "Alice founded the observatory in 1980."},
    {"url": "doc_b", "title": "Location", "text": "The observatory is in Berlin."},
    {"url": "doc_c", "title": "Unrelated", "text": "Bananas grow in warm regions."},
]
INTENT = {
    "query_explanation": "Find the founder and location of the observatory.",
    "active_direction": "Find independent evidence for the observatory location.",
    "searchable_directions": ["Alice observatory", "observatory Berlin"],
    "no_positive_feedback_directions": [],
    "completed_directions": [],
    "change_summary": "Continue looking for supporting evidence.",
}


class AnnotationClient:
    def __init__(self):
        self.inputs = []
        self.systems = []

    def complete(self, model, system, user):
        payload = json.loads(user)
        self.inputs.append(payload)
        self.systems.append(system)
        return {"documents": [
            {"doc_id": d["doc_id"], "relevant": d["doc_id"] != "doc_c", "intent": "Supports a query constraint."}
            for d in payload["documents"]
        ] + [{"doc_id": "invented", "relevant": True, "intent": "Must not be committed"}]}


class AuditClient:
    def __init__(self, verdict="answer_ready", error=False):
        self.verdict = verdict
        self.error = error
        self.inputs = []

    def complete(self, model, system, user):
        self.inputs.append(json.loads(user))
        if self.error:
            raise RuntimeError("Simulated endpoint failure")
        return {
            "summary": "Alice founded the observatory.", "coverage": {},
            "missing_constraints": [] if self.verdict == "answer_ready" else ["location"],
            "verdict": self.verdict, "reason": "Evidence checked.",
            "next_intent": "Find the location.",
        }


class Dataset:
    name = "longsealqa"
    evaluation_mode = "document"
    _query_index = {"test": {"document_ids": ["doc_a", "doc_b"]}}

    def evaluate_results_recall(self, query_id, ids):
        return len(set(ids) & {"doc_a", "doc_b"}) / 2

    def evaluate_results_precision(self, query_id, ids):
        return len(set(ids) & {"doc_a", "doc_b"}) / max(len(ids), 1)

    def evaluate_results_final_answer_recall(self, query_id, ids):
        return float("doc_a" in ids)


def make_memory(query_id="test"):
    toolset, search = create_longseal_toolset(DOCUMENTS)
    client = AnnotationClient()
    memory = MemoryOperator(
        query_id=query_id, query_text="Who founded the observatory and where is it?",
        toolset=toolset, search_tool=search, normalize_ids=False,
        relevance_judge=RelevanceJudge(query_id, client=client),
    )
    return memory, client


def planner_response():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(INTENT)))])


class MemoryTests(unittest.TestCase):
    def test_retrieval_is_committed_but_annotations_do_not_auto_curate(self):
        memory, client = make_memory()
        _, metadata = memory.search({"query": "observatory"})
        feedback = memory.annotate_retrieval(set(), metadata)
        self.assertEqual(set(memory.view.pool_ids), {"doc_a", "doc_b", "doc_c"})
        self.assertEqual(memory.view.curated_ids, [])
        self.assertEqual(set(memory.relevance_annotations), {"doc_a", "doc_b"})
        self.assertNotIn("invented", memory.relevance_annotations)
        self.assertIn("2/3", feedback)
        self.assertEqual(len(client.inputs), 1)
        self.assertEqual(client.inputs[0]["operation"], "annotate_documents")
        self.assertEqual(client.systems[0], MEMORY_OPERATOR_SYSTEM_PROMPT)
        memory.curate({"add_ids": ["doc_a", "invented", "doc_a"]})
        self.assertEqual(memory.view.curated_ids, ["doc_a"])
        self.assertEqual(memory.view.curated_notes["doc_a"], "Supports a query constraint.")

    def test_view_cannot_commit_nested_mutations_or_expose_mutators(self):
        memory, _ = make_memory()
        memory.search({"query": "observatory"})
        memory.view.pool_ids.clear()
        memory.view.doc_store["doc_a"]["snippet"] = "fabricated"
        self.assertEqual(len(memory.view.pool_ids), 3)
        self.assertNotEqual(memory.view.doc_store["doc_a"]["snippet"], "fabricated")
        with self.assertRaises(AttributeError):
            memory.view.current_intent = "fabricated"
        with self.assertRaises(AttributeError):
            memory.view.curate({"add_ids": ["doc_a"]})
        other, _ = make_memory("other")
        self.assertEqual(other.view.pool_ids, [])

    def test_capacity_and_pool_membership_are_enforced(self):
        memory, _ = make_memory()
        ids = [f"d{i}" for i in range(70)]
        memory.search_tool = lambda params, overrides: (
            "\n".join(f"# DOCUMENT ID: {doc_id}\nUseful text" for doc_id in ids),
            SearchToolCallMetadata(returned_chunk_ids=ids),
        )
        memory.search({"query": "evidence"})
        memory.curate({"add_ids": ids + ["unknown"]})
        self.assertEqual(len(memory.view.curated_ids), 60)
        self.assertTrue(set(memory.view.curated_ids) <= set(memory.view.pool_ids))

    def test_redirect_uses_selected_evidence_and_only_commits_valid_inputs(self):
        memory, annotation_client = make_memory()
        _, meta = memory.search({"query": "observatory"})
        memory.annotate_retrieval(set(), meta)
        completion = Mock(return_value=planner_response())
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        with patch("harness_search.memory_operator.INTENT_MODEL_API_KEY", "offline"), patch("openai.OpenAI", return_value=client):
            invalid = memory.redirect({"doc_ids": ["unknown"], "reasoning": "Choose a new evidence direction."})
            self.assertIn("Invalid IDs", invalid)
            completion.assert_not_called()
            result = memory.redirect({"doc_ids": ["doc_a"], "reasoning": "Find the location after identifying the founder."})
        self.assertTrue(result.startswith("Intent revised"))
        self.assertEqual(json.loads(memory.view.current_intent), INTENT)
        self.assertEqual(memory.view.query, "Who founded the observatory and where is it?")
        messages = completion.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], annotation_client.systems[0])
        self.assertEqual(messages[0]["content"], MEMORY_OPERATOR_SYSTEM_PROMPT)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["operation"], "update_direction")
        self.assertEqual([d["doc_id"] for d in payload["selected_documents"]], ["doc_a"])
        self.assertIn(DOCUMENTS[0]["text"], payload["selected_documents"][0]["text"])
        self.assertNotIn("documents", payload)
        self.assertIsInstance(payload["recent_search_history"], list)
        self.assertIsInstance(payload["direction_ledger"], list)
        memory.start_retrieval_stage()
        self.assertEqual(memory.docs_since_redirect, frozenset())

    def test_initial_intent_retry_preserves_shared_system_and_operation(self):
        memory, _ = make_memory()
        invalid_response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="{invalid JSON")
        )])
        completion = Mock(side_effect=[invalid_response, planner_response()])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        with patch("harness_search.memory_operator.INTENT_MODEL_API_KEY", "offline"), patch("openai.OpenAI", return_value=client):
            result = memory.redirect({"doc_ids": [], "reasoning": "Construct the initial retrieval intent."}, initial=True)
        self.assertTrue(result.startswith("Intent revised"))
        self.assertEqual(completion.call_count, 2)
        for request in completion.call_args_list:
            messages = request.kwargs["messages"]
            self.assertEqual(messages[0]["content"], MEMORY_OPERATOR_SYSTEM_PROMPT)
            payload = json.loads(messages[1]["content"])
            self.assertEqual(payload["operation"], "update_direction")
            self.assertEqual(payload["selected_documents"], [])
            self.assertEqual(payload["recent_search_history"], [])
            self.assertEqual(payload["curated_doc_ids"], [])
        self.assertEqual(memory.view.curated_ids, [])
        self.assertEqual(json.loads(memory.view.current_intent), INTENT)

    def test_relevance_batches_all_new_documents_and_preserves_partial_results(self):
        client = AnnotationClient()
        judge = RelevanceJudge("test", client)
        docs = [{"doc_id": f"d{i}", "text": "Evidence"} for i in range(12)]
        _, annotations = judge.annotate("query", "intent", docs)
        self.assertEqual([len(p["documents"]) for p in client.inputs], [5, 5, 2])
        self.assertEqual(set(annotations), {f"d{i}" for i in range(12)})
        self.assertEqual(judge.calls, 3)
        partial_client = AnnotationClient()
        original = partial_client.complete
        def fail_second(model, system, user):
            if partial_client.inputs:
                raise RuntimeError("Second batch unavailable")
            return original(model, system, user)
        partial_client.complete = fail_second
        _, partial = RelevanceJudge("test", partial_client).annotate("query", "intent", docs)
        self.assertEqual(set(partial), {f"d{i}" for i in range(5)})

    def test_auditor_sees_only_curated_evidence_and_keeps_existing_fallback(self):
        memory, _ = make_memory()
        memory.search({"query": "observatory"})
        memory.curate({"add_ids": ["doc_a"]})
        client = AuditClient()
        auditor = SummaryAuditor("test", client)
        self.assertEqual(auditor.audit(memory.view, 5, True)["verdict"], "answer_ready")
        self.assertEqual([d["doc_id"] for d in client.inputs[0]["curated_documents"]], ["doc_a"])
        self.assertEqual(auditor.audit(memory.view, 0, True)["verdict"], "search_more")
        failed = SummaryAuditor("test", AuditClient(error=True)).audit(memory.view, 5, True)
        self.assertTrue(failed["judge_error"])
        self.assertEqual(failed["verdict"], "answer_ready")
        self.assertEqual(memory.view.curated_ids, ["doc_a"])


class EpisodeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patches = [
            patch("harness_search.memory_operator.INTENT_MODEL_API_KEY", "offline"),
            patch("runtime.environment.SAVE_TRAJECTORIES", False),
            patch("runtime.environment.INTENT_SEARCHES_PER_STAGE", 5),
            patch("harness_search.memory_operator.V3_JUDGES_ENABLED", True),
        ]
        for p in self.patches:
            p.start()
        completion = Mock(return_value=planner_response())
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        self.openai_patch = patch("openai.OpenAI", return_value=client)
        self.openai_patch.start()

    async def asyncTearDown(self):
        self.openai_patch.stop()
        for p in reversed(self.patches):
            p.stop()

    async def make_env(self, max_turns=40):
        toolset, search = create_longseal_toolset(DOCUMENTS)
        env = HarnessSearchEnv(toolset, search, "test", "Who founded the observatory and where?", Dataset(), max_turns=max_turns)
        env._normalize_ids = False
        env.memory_operator.configure(normalize_ids=False)
        env.memory_operator.relevance_judge = RelevanceJudge("test", AnnotationClient())
        env.summary_auditor = SummaryAuditor("test", AuditClient())
        await env.initial_observation()
        return env

    async def step(self, env, name, params):
        action = ChatRetrievalPolicy._to_tokens(env, name, params)
        return await env.step(action.tokens)

    async def test_search_curate_redirect_read_end_preserves_stage_protocol(self):
        env = await self.make_env()
        self.assertTrue(env._retrieval_required)
        for _ in range(5):
            await self.step(env, "search", {"query": "observatory"})
        self.assertTrue(env._checkpoint_curate_required)
        self.assertEqual(env.summary_auditor.calls, 0)
        await self.step(env, "curate", {"add_ids": ["doc_a", "doc_b"]})
        self.assertTrue(env._intent_revision_required)
        self.assertEqual(env.summary_auditor.calls, 0)
        await self.step(env, "redirect", {"doc_ids": ["doc_a"], "reasoning": "Verify the location using the retrieved founder evidence."})
        self.assertTrue(env._retrieval_required)
        self.assertFalse(env._checkpoint_curate_required)
        await self.step(env, "read", {"doc_id": "doc_a"})
        done = await self.step(env, "end", {"reasoning": "All query constraints have evidence."})
        self.assertTrue(done.episode_done)
        self.assertEqual(env.summary_auditor.calls, 1)
        self.assertEqual(env._terminal_metrics["recall"], 1.0)

    async def test_budget_exhaustion_curates_then_audits_without_extending_search(self):
        env = await self.make_env(max_turns=1)
        env.summary_auditor = SummaryAuditor("test", AuditClient("search_more"))
        first = await self.step(env, "search", {"query": "observatory"})
        self.assertFalse(first.episode_done)
        self.assertTrue(env._terminal_curate_required)
        done = await self.step(env, "curate", {"add_ids": ["doc_a"]})
        self.assertTrue(done.episode_done)
        self.assertEqual(env.summary_auditor.calls, 1)
        self.assertEqual(env.summary_auditor.latest["verdict"], "search_more")
        self.assertEqual(env._terminal_metrics["max_turns_reached"], 1.0)

    async def test_old_checkpoint_tool_names_parse_as_method_actions(self):
        env = await self.make_env()
        for old, new in LEGACY_TOOL_NAMES.items():
            params = {"search": {"query": "observatory"}, "read": {"doc_id": "doc_a"}, "redirect": {"doc_ids": [], "reasoning": "Continue searching for evidence."}, "end": {"reasoning": "Evidence is sufficient."}}[new]
            tokens = ChatRetrievalPolicy._to_tokens(env, old, params).tokens
            action = TinkerAgentInferenceModel.harmony_tinker_tokens_to_action(env.enc, tokens, env._build_full_toolset())
            self.assertEqual(action.tools[0].tool_schema.name, new)
        self.assertEqual(set(LEGACY_TOOL_NAMES) & set(env._build_full_toolset().tools), set())

    async def test_policy_proposal_does_not_change_committed_memory(self):
        env = await self.make_env()
        await self.step(env, "search", {"query": "observatory"})
        policy = VllmRetrievalPolicy(base_url="http://unused", model="offline", max_tokens=100, temperature=0, top_p=1, timeout=5)
        before = env.wm.snapshot()
        prompt, eligible = policy._forced_tool_context(env, "curate")
        self.assertIn("CURATION CANDIDATES", prompt)
        self.assertTrue(eligible)
        self.assertEqual(before, env.wm.snapshot())

    async def test_chat_adapter_accepts_legacy_json_without_losing_arguments(self):
        env = await self.make_env()
        policy = ChatRetrievalPolicy(base_url="http://unused", model="offline", fallback_model="offline", api_key="offline", max_tokens=100, temperature=0, top_p=1, timeout=5)
        for payload in [
            {"tool_name": "search_corpus", "arguments": {"query": "specific clue"}},
            {"read_document": {"doc_id": "doc_a"}},
        ]:
            with patch.object(policy, "_request_json", return_value=payload):
                proposed = await policy.action(env)
            action = TinkerAgentInferenceModel.harmony_tinker_tokens_to_action(env.enc, proposed.tokens, env._build_full_toolset())
            name = action.tools[0].tool_schema.name
            self.assertIn(name, {"search", "read"})
            self.assertEqual(action.params[0], {"query": "specific clue"} if name == "search" else {"doc_id": "doc_a"})

    async def test_insufficient_evidence_blocks_early_end_and_requests_redirect(self):
        env = await self.make_env()
        env.summary_auditor = SummaryAuditor("test", AuditClient("search_more"))
        await self.step(env, "search", {"query": "observatory"})
        await self.step(env, "curate", {"add_ids": ["doc_a"]})
        await self.step(env, "redirect", {"doc_ids": ["doc_a"], "reasoning": "Find additional evidence for the observatory location."})
        await self.step(env, "read", {"doc_id": "doc_a"})
        result = await self.step(env, "end", {"reasoning": "Request an evidence sufficiency check."})
        self.assertFalse(result.episode_done)
        self.assertTrue(env._intent_revision_required)
        self.assertEqual(env._terminal_summary_blocks, 1)
        self.assertEqual(env.summary_auditor.calls, 1)


class ConfigurationTests(unittest.TestCase):
    def test_new_names_take_precedence_and_old_model_names_remain_valid(self):
        from harness_search import model_settings
        pairs = {
            "RELEVANCE_JUDGE_MODEL_NAME": "REFERENCE_JUDGE_MODEL_NAME",
            "RELEVANCE_JUDGE_FALLBACK_NAME": "REFERENCE_JUDGE_FALLBACK_NAME",
            "SUMMARY_AUDITOR_MODEL_NAME": "SUMMARY_JUDGE_MODEL_NAME",
            "SUMMARY_AUDITOR_FALLBACK_NAME": "SUMMARY_JUDGE_FALLBACK_NAME",
        }
        try:
            with patch.dict(os.environ):
                for new, old in pairs.items():
                    os.environ.pop(new, None)
                    os.environ[old] = "legacy-" + old
                importlib.reload(model_settings)
                for new, old in pairs.items():
                    self.assertEqual(getattr(model_settings, new), "legacy-" + old)
                    os.environ[new] = "new-" + new
                importlib.reload(model_settings)
                for new in pairs:
                    self.assertEqual(getattr(model_settings, new), "new-" + new)
        finally:
            importlib.reload(model_settings)


if __name__ == "__main__":
    unittest.main()
