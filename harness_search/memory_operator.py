"""Memory Operator: the sole owner of committed per-query working memory.

The retrieval policy supplies operations and document selections. This agent
executes the selected retrieval operation, validates and commits its observations,
annotates candidates through RelevanceJudge, and invokes the Direction Planner for
redirect proposals. It never chooses the next operation or approves termination.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog
from harness_search.model_settings import (
    ABLATE_REVIEW_DOCS_UNAVAILABLE, INTENT_MODEL_API_KEY, INTENT_MODEL_BASE_URL,
    INTENT_MODEL_HEADER_API_KEY, INTENT_MODEL_TIMEOUT, INTENT_MODEL_MAX_TOKENS,
    INTENT_MODEL_RETRY_MAX_TOKENS, INTENT_MODEL_NAME, RELEVANCE_JUDGE_MODEL_NAME,
    V3_JUDGES_ENABLED,
)
from harness_search.tools import (
    ToolCallMetadata, SearchToolCallMetadata, GrepCorpusToolCallMetadata,
)
from harness_search.ultra_core import (
    WorkingMemory, parse_doc_texts_from_observation, append_token_marker,
    compress_search_observation, auto_populate_from_first_search,
    AUTO_POPULATE_TOP_K, FAN_OUT_MAX_QUERIES, MAX_REVIEW_DOCS,
    V8D_AUTO_POPULATE_FIRST_SEARCH, V8D_IMPORTANCE_TAGGING,
    V8D_SENTENCE_COMPRESS, V8D_TOKEN_BUDGET_MARKER,
    V8D_INTENT_STATE_TRACKING, V8D_ADAPTIVE_RERANK_INSTRUCTION,
)
from harness_search.relevance_judge import RelevanceJudge
from harness_search.memory_prompts import MEMORY_OPERATOR_SYSTEM_PROMPT

logger = structlog.get_logger(__name__)

class FanOutSearchToolCallMetadata(ToolCallMetadata):
    returned_chunk_ids: List[str]
    queries_executed: int


class MemoryView:
    """Read-only facade; mutable fields are detached copies, never live aliases."""
    __slots__ = ("__memory",)
    _FIELDS = frozenset({
        "query", "turn_number", "curated_ids", "curated_notes", "pool_ids",
        "pool_id_set", "search_history", "intent_direction_states",
        "current_intent", "intent_history", "doc_store", "normalize_ids",
        "curated_importance", "rerank_instruction", "auto_populated", "dup_skipped",
    })

    def __init__(self, memory):
        object.__setattr__(self, "_MemoryView__memory", memory)

    def __setattr__(self, name, value):
        raise AttributeError("Working memory is committed only by MemoryOperator")

    def __getattr__(self, name):
        if name not in self._FIELDS:
            raise AttributeError(name)
        return copy.deepcopy(getattr(self.__memory._wm, name))

    def get_pool_size(self):
        return self.__memory._wm.get_pool_size()

    def to_text(self):
        return self.__memory._wm.to_text()

    def snapshot(self):
        return copy.deepcopy(self.__memory._wm.snapshot())

    def _normalize_id(self, doc_id):
        return self.__memory._wm._normalize_id(doc_id)


class MemoryOperator:
    """A stateful memory agent with explicit commit operations and read-only views."""

    def __init__(self, *, query_id, query_text, toolset, search_tool,
                 normalize_ids=True, text_token_counter=None, relevance_judge=None):
        self.query_id = query_id
        self.toolset = toolset
        self.search_tool = search_tool
        self.text_token_counter = text_token_counter
        self._wm = WorkingMemory(query_text, normalize_ids=normalize_ids)
        self.view = MemoryView(self)
        self.relevance_judge = relevance_judge or RelevanceJudge(query_id)
        self._ids_seen: Set[str] = set()
        self._doc_id_to_query: Dict[str, str] = {}
        self._docs_since_intent: Set[str] = set()
        self._last_intent_doc_ids: List[str] = []
        self._intent_selected_doc_ids: List[str] = []
        self._intent_meta_query_backlog: List[str] = []
        self._intent_pending_recommendations: Dict[str, float] = {}
        self._relevance_annotations: Dict[str, str] = {}
        self._first_search_done = False
        self._approx_prompt_tokens = 0

    @property
    def docs_since_redirect(self):
        return frozenset(self._docs_since_intent)

    @property
    def relevance_annotations(self):
        return dict(self._relevance_annotations)

    def reset_working_memory(self):
        self._wm = WorkingMemory(self._wm.query, normalize_ids=self._wm.normalize_ids)

    def configure(self, *, normalize_ids=None, rerank_instruction=None):
        if normalize_ids is not None:
            self._wm.normalize_ids = normalize_ids
        if rerank_instruction is not None:
            self._wm.rerank_instruction = rerank_instruction

    def set_prompt_tokens(self, count):
        self._approx_prompt_tokens = count

    def advance_turn(self):
        self._wm.advance_turn()

    def complete_curation(self):
        self._intent_pending_recommendations.clear()

    def start_retrieval_stage(self):
        self._docs_since_intent.clear()

    def annotate_retrieval(self, pool_before, metadata):
        new_ids = [doc_id for doc_id in self._wm.pool_ids if doc_id not in pool_before]
        self._docs_since_intent.update(new_ids)
        returned = getattr(metadata, "returned_chunk_ids", []) if metadata else []
        for raw_id in returned:
            doc_id = self._wm._normalize_id(str(raw_id))
            if doc_id in self._wm.pool_ids:
                self._docs_since_intent.add(doc_id)
        if not V3_JUDGES_ENABLED or not new_ids:
            return ""
        documents = []
        for doc_id in dict.fromkeys(new_ids):
            store = self._wm.doc_store.get(doc_id, {})
            text = store.get("full_text") or store.get("snippet") or ""
            documents.append({"doc_id": doc_id, "text": text[:3000]})
        feedback, annotations = self.relevance_judge.annotate(
            self._wm.query, self._wm.current_intent, documents,
        )
        self._relevance_annotations.update(annotations)
        return feedback

    def _maybe_wrap_search_output(
        self,
        output: str,
        query_for_compress: str,
        first_search_ranked_ids: Optional[List[str]] = None,
    ) -> str:
        """v8d wrapper: BM25 compress + auto-populate + token marker."""
        # 1. Sentence-level compression (no-op unless flag on)
        if V8D_SENTENCE_COMPRESS and query_for_compress:
            output = compress_search_observation(query_for_compress, output)

        # 2. Auto-populate the curated set from the first search's top hits
        if (
            V8D_AUTO_POPULATE_FIRST_SEARCH
            and not self._first_search_done
            and first_search_ranked_ids
        ):
            added = auto_populate_from_first_search(
                self._wm, first_search_ranked_ids, top_k=AUTO_POPULATE_TOP_K,
            )
            self._first_search_done = True
            if added > 0:
                output = (
                    output
                    + f"\n\n[AUTO-POPULATED] Top {added} docs from this search have been "
                    "added to your curated set at 'fair' importance. Use `curate` with "
                    "`importance` to promote/demote and `remove_ids` to drop irrelevant ones."
                )

        # 3. Token budget marker (no-op unless flag on)
        if V8D_TOKEN_BUDGET_MARKER and self.text_token_counter is not None:
            try:
                used = self._approx_prompt_tokens + self.text_token_counter(output)
                output = append_token_marker(output, used)
            except Exception:
                pass

        return output


    def search(self, params: Dict) -> Tuple[str, Optional[ToolCallMetadata]]:
        query = params.get("query") or params.get("q", "")
        pool_before = self._wm.get_pool_size()
        # v8d: pipe per-episode rerank instruction through to the search tool.
        overrides: Dict[str, Any] = {"ignore_ids": list(self._ids_seen)}
        if V8D_ADAPTIVE_RERANK_INSTRUCTION and self._wm.rerank_instruction:
            overrides["rerank_instruction"] = self._wm.rerank_instruction
        output, meta = self.search_tool(params, overrides)
        ranked_ids: List[str] = []
        direction_status = ""
        if meta and isinstance(meta, SearchToolCallMetadata):
            ranked_ids = list(meta.returned_chunk_ids)
            self._ids_seen.update(meta.returned_chunk_ids)
            doc_texts = parse_doc_texts_from_observation(output)
            self._wm.add_to_pool(meta.returned_chunk_ids, doc_texts)
            for cid in meta.returned_chunk_ids:
                doc_id = cid.split("_")[0] if "_" in cid else cid
                self._doc_id_to_query.setdefault(doc_id, str(query))
            num_new = self._wm.get_pool_size() - pool_before
            self._wm.add_search_record(
                "search", str(query)[:60], len(meta.returned_chunk_ids),
                num_new=num_new,
            )
            if V8D_INTENT_STATE_TRACKING:
                direction_status = self._wm.record_intent_direction(
                    "search", str(query), len(meta.returned_chunk_ids), num_new
                )
        output = self._maybe_wrap_search_output(
            output, query_for_compress=str(query),
            first_search_ranked_ids=ranked_ids,
        )
        if direction_status:
            output += "\n\n" + direction_status
        return output, meta


    def fan_out_search(self, params: Dict) -> Tuple[str, Optional[FanOutSearchToolCallMetadata]]:
        queries = params.get("queries", [])
        if not isinstance(queries, list) or not queries:
            return "No queries provided.", FanOutSearchToolCallMetadata(
                returned_chunk_ids=[], queries_executed=0,
            )

        queries = queries[:FAN_OUT_MAX_QUERIES]
        all_results: List[str] = []
        all_chunk_ids: List[str] = []
        pool_before = self._wm.get_pool_size()

        for q in queries:
            if not isinstance(q, str) or not q.strip():
                continue
            try:
                overrides: Dict[str, Any] = {"ignore_ids": list(self._ids_seen)}
                if V8D_ADAPTIVE_RERANK_INSTRUCTION and self._wm.rerank_instruction:
                    overrides["rerank_instruction"] = self._wm.rerank_instruction
                output, meta = self.search_tool({"query": q}, overrides)
                all_results.append(output)
                if meta and isinstance(meta, SearchToolCallMetadata):
                    self._ids_seen.update(meta.returned_chunk_ids)
                    doc_texts = parse_doc_texts_from_observation(output)
                    self._wm.add_to_pool(meta.returned_chunk_ids, doc_texts)
                    all_chunk_ids.extend(meta.returned_chunk_ids)
                    for cid in meta.returned_chunk_ids:
                        doc_id = cid.split("_")[0] if "_" in cid else cid
                        self._doc_id_to_query.setdefault(doc_id, str(q))
            except Exception as e:
                logger.warning("fan_out_error", query=str(q)[:100], error=str(e)[:200])
                all_results.append("No results.")

        q_summary = "; ".join(str(q)[:30] for q in queries[:3])
        num_new = self._wm.get_pool_size() - pool_before
        self._wm.add_search_record(
            "fan_out", q_summary, len(all_chunk_ids), num_new=num_new,
        )
        direction_status = ""
        if V8D_INTENT_STATE_TRACKING:
            direction_status = self._wm.record_intent_direction(
                "fan_out", q_summary, len(all_chunk_ids), num_new
            )
        combined = "\n".join(all_results) if all_results else "No results found."
        # v8d: compress (using concatenated query string), auto-populate, token marker
        concat_query = " ".join(str(q) for q in queries if isinstance(q, str))
        combined = self._maybe_wrap_search_output(
            combined,
            query_for_compress=concat_query,
            first_search_ranked_ids=all_chunk_ids,
        )
        if direction_status:
            combined += "\n\n" + direction_status
        return combined, FanOutSearchToolCallMetadata(
            returned_chunk_ids=all_chunk_ids, queries_executed=len(queries),
        )


    def grep(self, params: Dict) -> Tuple[str, Optional[ToolCallMetadata]]:
        grep_tool = self.toolset.get_tool("grep_corpus")
        if grep_tool is None:
            return "grep_corpus not available.", None
        pool_before = self._wm.get_pool_size()
        output, meta = grep_tool(params)
        direction_status = ""
        if meta and isinstance(meta, GrepCorpusToolCallMetadata):
            doc_texts = parse_doc_texts_from_observation(output)
            self._wm.add_to_pool(meta.returned_chunk_ids, doc_texts)
            num_new = self._wm.get_pool_size() - pool_before
            if V8D_INTENT_STATE_TRACKING:
                direction_status = self._wm.record_intent_direction(
                    "grep", str(params.get("pattern", "")), len(meta.returned_chunk_ids), num_new
                )
            self._wm.add_search_record(
                "grep", str(params.get("pattern", ""))[:60],
                len(meta.returned_chunk_ids), num_new=num_new,
            )
        # v8d: grep results can still benefit from sentence-level compression and token marker
        output = self._maybe_wrap_search_output(
            output, query_for_compress=str(params.get("pattern", "")),
            first_search_ranked_ids=None,
        )
        if direction_status:
            output += "\n\n" + direction_status
        return output, meta


    def read(self, params: Dict) -> Tuple[str, Optional[ToolCallMetadata]]:
        read_tool = self.toolset.get_tool("read")
        if read_tool is None:
            return "read not available.", None
        doc_id = params.get("doc_id") or params.get("id", "")
        if self._wm.normalize_ids and "_" in doc_id:
            doc_id = doc_id.split("_")[0]
        overrides = {}
        if doc_id in self._doc_id_to_query:
            overrides["query"] = self._doc_id_to_query[doc_id]
        pool_before = self._wm.get_pool_size()
        output, meta = read_tool(params, overrides or None)
        doc_texts = parse_doc_texts_from_observation(output)
        if doc_texts:
            self._wm.add_to_pool(list(doc_texts.keys()), doc_texts)
        num_new = self._wm.get_pool_size() - pool_before
        self._wm.add_search_record(
            "read", str(doc_id)[:30],
            len(doc_texts) if doc_texts else 1, num_new=num_new,
        )
        # v8d: read returns full text — compression is too aggressive here,
        # but still append token marker.
        if V8D_TOKEN_BUDGET_MARKER and self.text_token_counter is not None:
            try:
                used = self._approx_prompt_tokens + self.text_token_counter(output)
                output = append_token_marker(output, used)
            except Exception:
                pass
        return output, meta


    def curate(self, params: Dict) -> str:
        add_ids = params.get("add_ids", [])
        remove_ids = params.get("remove_ids", [])
        if not isinstance(add_ids, list):
            add_ids = [str(add_ids)] if add_ids else []
        if not isinstance(remove_ids, list):
            remove_ids = [str(remove_ids)] if remove_ids else []

        importance: Optional[Dict[str, str]] = None
        if V8D_IMPORTANCE_TAGGING:
            raw = params.get("importance")
            if isinstance(raw, dict):
                importance = {str(k): str(v) for k, v in raw.items()}

        notes = {
            self._wm._normalize_id(str(doc_id)): self._relevance_annotations.get(
                self._wm._normalize_id(str(doc_id)),
                "Supports the current query intent.",
            )
            for doc_id in add_ids
        }
        return self._wm.curate(add_ids, remove_ids, notes=notes, importance=importance)


    def review(self, params: Dict) -> str:
        if ABLATE_REVIEW_DOCS_UNAVAILABLE:
            self._wm.add_search_record("review", "unavailable", 0)
            return "review_docs: unavailable in this ablation."

        doc_ids = params.get("doc_ids", [])
        if not isinstance(doc_ids, list):
            doc_ids = [str(doc_ids)] if doc_ids else []
        doc_ids = [str(x).strip() for x in doc_ids if x][:MAX_REVIEW_DOCS]
        if not doc_ids:
            return "No doc_ids provided."
        result = self._wm.review_docs(doc_ids)
        self._wm.add_search_record("review", ", ".join(doc_ids[:3]), len(doc_ids))
        return result


    def redirect(self, params: Dict, initial: bool = False) -> str:
        """Use the Direction Planner to revise the advisory search intent."""
        reasoning = str(params.get("reasoning", "")).strip()
        if len(reasoning) < 12:
            return "redirect: reasoning is missing or too short."
        raw_doc_ids = params.get("doc_ids", [])
        if not isinstance(raw_doc_ids, list):
            return "redirect: doc_ids must be a list."
        doc_ids = []
        for raw_id in raw_doc_ids:
            raw_id = str(raw_id).strip()
            if not raw_id:
                continue
            doc_id = self._wm._normalize_id(raw_id)
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
        if len(doc_ids) > 5:
            return "redirect: select at most 5 document IDs."
        if initial and doc_ids:
            return "redirect: initial intent must use an empty document list."
        invalid_ids = [
            doc_id for doc_id in doc_ids
            if doc_id not in self._docs_since_intent or doc_id not in self._wm.pool_ids
        ]
        if not initial and invalid_ids:
            return (
                "redirect: IDs must come from documents found since the previous "
                "intent revision. Invalid IDs: " + ", ".join(invalid_ids[:5])
            )
        selected_documents = [
            {
                "doc_id": doc_id,
                "text": self._wm.doc_store.get(doc_id, {}).get(
                    "full_text", self._wm.doc_store.get(doc_id, {}).get("snippet", "")
                )[:4000],
            }
            for doc_id in doc_ids
        ]
        system = MEMORY_OPERATOR_SYSTEM_PROMPT
        user = json.dumps({
            "operation": "update_direction",
            "original_query": self._wm.query,
            "previous_intent": self._wm.current_intent,
            "policy_reasoning": reasoning,
            "selected_documents": selected_documents,
            "recent_search_history": self._wm.search_history[-12:],
            "direction_ledger": self._wm.intent_direction_states[-12:],
            "curated_doc_ids": self._wm.curated_ids[:30],
        }, ensure_ascii=False)
        if not INTENT_MODEL_API_KEY:
            logger.warning(
                "intent_model_error",
                error="INTENT_MODEL_API_KEY is not configured",
                qid=self.query_id,
            )
            return "redirect: INTENT_MODEL_API_KEY is not configured."
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=INTENT_MODEL_BASE_URL,
                api_key=INTENT_MODEL_API_KEY,
                default_headers={"api-key": INTENT_MODEL_HEADER_API_KEY},
                timeout=INTENT_MODEL_TIMEOUT,
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            response = client.chat.completions.create(
                model=RELEVANCE_JUDGE_MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=INTENT_MODEL_MAX_TOKENS,
                response_format={"type": "json_object"},
                extra_body=(
                    {"reasoning_effort": "low"}
                    if "gpt-oss" in RELEVANCE_JUDGE_MODEL_NAME.lower()
                    else ({"chat_template_kwargs": {"enable_thinking": False}}
                          if "qwen" in RELEVANCE_JUDGE_MODEL_NAME.lower() else None)
                ),
            )
            revised = response.choices[0].message.content or ""
            try:
                revised_obj = json.loads(revised)
            except json.JSONDecodeError as first_exc:
                logger.warning(
                    "intent_json_retry", qid=self.query_id,
                    error=str(first_exc)[:200], output_chars=len(revised),
                )
                retry_messages = messages + [{
                    "role": "user",
                    "content": (
                        "The previous response was invalid or truncated JSON. Regenerate from "
                        "the same evidence as one compact JSON object only. Use at most 3 short "
                        "items per direction list and short strings. Close every array, "
                        "string, and object."
                    ),
                }]
                retry_response = client.chat.completions.create(
                    model=RELEVANCE_JUDGE_MODEL_NAME,
                    messages=retry_messages,
                    temperature=0.0,
                    max_tokens=INTENT_MODEL_RETRY_MAX_TOKENS,
                    response_format={"type": "json_object"},
                    extra_body=(
                        {"reasoning_effort": "low"}
                        if "gpt-oss" in RELEVANCE_JUDGE_MODEL_NAME.lower()
                        else ({"chat_template_kwargs": {"enable_thinking": False}}
                              if "qwen" in RELEVANCE_JUDGE_MODEL_NAME.lower() else None)
                    ),
                )
                revised = retry_response.choices[0].message.content or ""
                revised_obj = json.loads(revised)
            # Retain intent directions as policy-visible planning state; do not execute retrieval here.
            meta_search_queries = revised_obj.get("searchable_directions", [])
            if isinstance(meta_search_queries, list):
                for backlog_query in meta_search_queries:
                    if isinstance(backlog_query, dict):
                        backlog_query = (backlog_query.get("query") or backlog_query.get("direction") or backlog_query.get("description") or "")
                    backlog_query = str(backlog_query).strip()
                    if len(backlog_query) >= 8 and backlog_query not in self._intent_meta_query_backlog:
                        self._intent_meta_query_backlog.append(backlog_query)
            intent_keys = (
                "query_explanation", "active_direction", "searchable_directions",
                "no_positive_feedback_directions", "completed_directions",
                "change_summary",
            )
            revised_obj = {key: revised_obj.get(key, [] if key.endswith("directions") else "") for key in intent_keys}
            for key in ("searchable_directions", "no_positive_feedback_directions", "completed_directions"):
                value = revised_obj.get(key, [])
                revised_obj[key] = value[:5] if isinstance(value, list) else []
            self._intent_pending_recommendations = {}
            revised = json.dumps(revised_obj, ensure_ascii=False, indent=2)
            self._last_intent_doc_ids = list(doc_ids)
            for selected_doc_id in doc_ids:
                if selected_doc_id not in self._intent_selected_doc_ids:
                    self._intent_selected_doc_ids.append(selected_doc_id)
        except Exception as exc:
            logger.warning("intent_model_error", error=str(exc)[:300], qid=self.query_id)
            return (
                "redirect: configured intent model unavailable. "
                f"Check {INTENT_MODEL_BASE_URL} model {INTENT_MODEL_NAME} ({str(exc)[:120]})."
            )
        self._wm.add_search_record("redirect", reasoning[:60], 0, num_new=0)
        return self._wm.redirect(reasoning, revised)
