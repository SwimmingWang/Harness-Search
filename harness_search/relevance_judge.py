"""Relevance Judge produces annotations; Memory Operator commits them."""
from typing import Dict, List
import json
import structlog
from harness_search.judge_client import JudgeClient
from harness_search.memory_prompts import MEMORY_OPERATOR_SYSTEM_PROMPT
from harness_search.model_settings import RELEVANCE_JUDGE_MODEL_NAME

logger = structlog.get_logger(__name__)

class RelevanceJudge:
    def __init__(self, query_id: str, client=None):
        self.query_id = query_id
        self.client = client or JudgeClient(query_id)
        self.calls = 0

    def annotate(self, query: str, current_intent: str, docs: List[Dict]):
        annotations = {}
        if not docs:
            return "[RELEVANCE JUDGE] no new documents to annotate.", annotations
        try:
            accepted = 0
            # Judge every newly retrieved document. Fan-out can return more
            # than 20 documents, so silently truncating here loses potentially
            # useful evidence before the policy ever sees an annotation.
            for start in range(0, len(docs), 5):
                batch = docs[start:start + 5]
                result = self.client.complete(
                    RELEVANCE_JUDGE_MODEL_NAME,
                    MEMORY_OPERATOR_SYSTEM_PROMPT,
                    json.dumps({
                        "operation": "annotate_documents",
                        "original_query": query,
                        "current_query_intent": current_intent,
                        "documents": batch,
                    }, ensure_ascii=False),
                )
                self.calls += 1
                logger.info(
                    "relevance_judge_result",
                    qid=self.query_id,
                    batch_size=len(batch),
                    result=json.dumps(result, ensure_ascii=False)[:4000],
                )
                batch_ids = {item["doc_id"] for item in batch}
                for item in result.get("documents", []):
                    if not isinstance(item, dict):
                        continue
                    doc_id = str(item.get("doc_id", ""))
                    if doc_id not in batch_ids or item.get("relevant") is not True:
                        continue
                    intent = str(item.get("intent", "")).strip()
                    if intent:
                        annotations[doc_id] = intent[:240]
                        accepted += 1
            recommended = [
                f"{item['doc_id']}: {annotations[item['doc_id']]}"
                for item in docs
                if item["doc_id"] in annotations
            ][:20]
            detail = "\n".join(recommended) or "(no recommendations in this batch)"
            return (
                f"[RELEVANCE JUDGE] {accepted}/{len(docs)} new documents judged useful.\n"
                "Recommended doc_id -> intent:\n" + detail
            ), annotations
        except Exception as exc:
            logger.warning("relevance_judge_error", qid=self.query_id, error=str(exc)[:240])
            return "[RELEVANCE JUDGE] unavailable; retrieval results remain visible.", annotations
