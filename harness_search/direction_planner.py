"""Memory Operator's advisory Direction Planner; never executes or commits."""
from __future__ import annotations

import json

from harness_search.judge_client import JudgeClient
from harness_search.memory_prompts import MEMORY_OPERATOR_SYSTEM_PROMPT
from harness_search.model_settings import RELEVANCE_JUDGE_MODEL_NAME


class DirectionPlanner:
    def __init__(self, query_id: str, client=None):
        self.client = client if client is not None else JudgeClient(query_id)

    def plan(self, *, query, previous_intent, reasoning, documents,
             search_history, direction_ledger, curated_ids, audit_feedback):
        result = self.client.complete(
            RELEVANCE_JUDGE_MODEL_NAME, MEMORY_OPERATOR_SYSTEM_PROMPT,
            json.dumps({
                "operation": "update_direction",
                "original_query": query,
                "previous_intent": previous_intent,
                "policy_reasoning": reasoning,
                "selected_documents": documents,
                "recent_search_history": search_history,
                "direction_ledger": direction_ledger,
                "curated_doc_ids": curated_ids,
                "summary_auditor_feedback": audit_feedback,
            }, ensure_ascii=False),
        )
        if not isinstance(result, dict):
            raise ValueError("Direction Planner must return a JSON object")
        plan = {}
        for key in ("query_explanation", "active_direction", "change_summary"):
            value = result.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Direction Planner returned empty or invalid {key}")
            plan[key] = value.strip()
        for key in ("searchable_directions", "no_positive_feedback_directions", "completed_directions"):
            value = result.get(key)
            if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
                raise ValueError(f"Direction Planner returned invalid {key}")
            plan[key] = list(dict.fromkeys(x.strip() for x in value))[:5]
        if not plan["searchable_directions"]:
            raise ValueError("Direction Planner returned no searchable directions")
        return plan
