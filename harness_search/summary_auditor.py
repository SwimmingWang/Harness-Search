"""Summary Auditor inspects a read-only evidence view; it cannot commit memory."""
from typing import Any, Dict
import json
import structlog
from harness_search.judge_client import JudgeClient
from harness_search.model_settings import SUMMARY_AUDITOR_MODEL_NAME, SUMMARY_MIN_RETRIEVAL_CALLS

logger = structlog.get_logger(__name__)

class SummaryAuditor:
    def __init__(self, query_id: str, client=None):
        self.query_id = query_id
        self.client = client or JudgeClient(query_id)
        self.calls = 0
        self.answer_ready = False
        self.latest: Dict[str, Any] = {}

    def audit(self, memory, retrieval_calls: int, terminal: bool) -> Dict[str, Any]:
        """Assess the curated evidence snapshot at a termination checkpoint."""
        document_store = memory.doc_store
        notes = memory.curated_notes
        evidence = []
        for doc_id in memory.curated_ids[:60]:
            store = document_store.get(doc_id, {})
            evidence.append({
                "doc_id": doc_id,
                "intent": notes.get(doc_id, ""),
                "text": (store.get("full_text") or store.get("snippet") or "")[:900],
            })
        try:
            result = self.client.complete(
                SUMMARY_AUDITOR_MODEL_NAME,
                "You are a summary auditor for a multi-document, multi-hop evidence task, not "
                "a search agent. Judge the UNION of the curated evidence set: different "
                "documents are expected to support different constraints. Never require one "
                "document to satisfy the whole query. Summarize only the curated evidence, "
                "then decide whether the set collectively covers all key constraints in the "
                "immutable query. Return JSON only with: summary, coverage, "
                "missing_constraints, verdict, "
                "reason, next_intent. Coverage must map each satisfied constraint to supporting "
                "doc_ids; missing_constraints must list every unsupported constraint. Verdict "
                "must be answer_ready, revise_summary, or search_more. Use answer_ready only "
                "when every key constraint has direct document support; for search_more, "
                "next_intent must be one "
                "specific missing-information target. Do not invent facts.",
                json.dumps({
                    "original_query": memory.query,
                    "current_query_intent": memory.current_intent,
                    "terminal_check": terminal,
                    "curated_documents": evidence,
                }, ensure_ascii=False),
            )
            verdict = str(result.get("verdict", "search_more")).strip().lower()
            if verdict not in {"answer_ready", "revise_summary", "search_more"}:
                verdict = "search_more"
            if (
                verdict == "answer_ready"
                and retrieval_calls < SUMMARY_MIN_RETRIEVAL_CALLS
            ):
                verdict = "search_more"
                result["reason"] = (
                    "The current evidence is coherent, but only "
                    f"{retrieval_calls}/{SUMMARY_MIN_RETRIEVAL_CALLS} required "
                    "retrieval calls have been completed. Broaden or verify the least "
                    "supported constraint before terminating."
                )
                result["next_intent"] = (
                    "Find an independent or complementary source for the least-supported "
                    "query constraint."
                )
            result["verdict"] = verdict
            self.calls += 1
            self.answer_ready = verdict == "answer_ready"
            self.latest = result
            return result
        except Exception as exc:
            logger.warning("summary_auditor_error", qid=self.query_id, error=str(exc)[:240])
            # Fail open on a terminal check, and preserve the v2 intent flow at a
            # periodic checkpoint when the judge endpoint is unavailable.
            return {
                "verdict": "answer_ready" if terminal else "search_more",
                "summary": "",
                "reason": "Summary Auditor unavailable.",
                "next_intent": "Continue the most discriminating unresolved query constraint.",
                "judge_error": True,
            }
