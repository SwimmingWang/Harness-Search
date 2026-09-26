"""Retrieval Policy: proposes tools and evidence selections without committing memory."""
from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
import json
import json_repair
import re
import urllib.error
import urllib.request
from typing import Dict, List, TYPE_CHECKING
import structlog
from openai_harmony import Conversation, Message, Role
from tinker_cookbook.completers import StopCondition, TokensWithLogprobs
from harness_search.ultra_core import CURATE_SCHEMA, REDIRECT_SCHEMA
from harness_search.tools import UserTextTool
from harness_search.actions import LEGACY_TOOL_NAMES, canonical_tool_name
if TYPE_CHECKING:
    from runtime.environment import HarnessSearchEnv

logger = structlog.get_logger(__name__)

class RetrievalPolicy(ABC):
    """Proposal authority shared by token and chat inference adapters."""

    @abstractmethod
    async def forced_tool_call(self, env, tool_name):
        """Propose arguments for a tool required by the harness."""
        raise NotImplementedError

class VllmRetrievalPolicy(RetrievalPolicy):
    """Token-level policy backed by vLLM raw completions."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        timeout: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

    @property
    def completions_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/completions"
        return f"{self.base_url}/v1/completions"

    async def __call__(self, model_input, stop: StopCondition) -> TokensWithLogprobs:
        prompt_tokens = model_input.to_ints()
        payload = {
            "model": self.model,
            "prompt": prompt_tokens,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
            "return_token_ids": True,
        }
        if stop and all(isinstance(s, int) for s in stop):
            payload["stop_token_ids"] = list(stop)
        elif stop:
            payload["stop"] = list(stop)

        data = await asyncio.to_thread(self._post_json, payload)
        choice = data["choices"][0]
        tokens = (
            choice.get("token_ids")
            or choice.get("tokens")
            or choice.get("text_token_ids")
            or []
        )
        if not tokens:
            raise RuntimeError(f"vLLM response did not include token IDs: {str(data)[:500]}")
        return TokensWithLogprobs(tokens=[int(t) for t in tokens], maybe_logprobs=None)

    @property
    def chat_completions_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def _forced_tool_context(
        self, env: HarnessSearchEnv, tool_name: str
    ) -> tuple[str, List[str]]:
        if tool_name == "redirect":
            eligible = [
                doc_id for doc_id in env.wm.pool_ids
                if doc_id in env.memory_operator.docs_since_redirect
            ]
            instruction = (
                "Select at most five documents found since the previous intent update. "
                "Explain which single active direction is complete or stalled and why the "
                "next intent should keep or change direction."
            )
        else:
            uncurated = [
                doc_id for doc_id in env.wm.pool_ids
                if doc_id not in env.wm.curated_ids
            ]
            # Put Relevance Judge recommendations first, then expose a rolling
            # high-recall window. The policy still has to select every ID
            # explicitly; the runtime never inserts this list into curate.
            eligible = []
            for doc_id in (
                [d for d in uncurated if d in env.memory_operator.relevance_annotations]
                + uncurated
            ):
                if doc_id not in eligible:
                    eligible.append(doc_id)
            eligible = eligible[:40]
            exposed = getattr(env, "_native_curation_exposed_ids", set())
            if not isinstance(exposed, set):
                exposed = set(exposed)
            exposed.update(eligible)
            env._native_curation_exposed_ids = exposed
            instruction = (
                "Call curate for a joint multi-document evidence set. Never require one "
                "document to satisfy the entire query. Review every CANDIDATE below and "
                "explicitly add every document that plausibly supports any one query "
                "constraint, entity bridge, or independent corroborating source. Usually "
                "select 10-25 IDs and at most 30 when that many plausible candidates exist. "
                "A non-empty Relevance Judge intent is a strong recommendation. Keep useful "
                "previously curated evidence, remove only clearly irrelevant documents, and "
                "do not invent IDs."
            )
        doc_lines = []
        if tool_name == "redirect":
            for doc_id in eligible:
                store = env.wm.doc_store.get(doc_id, {})
                snippet = store.get("full_text") or store.get("snippet") or ""
                doc_lines.append(f"{doc_id}: {snippet[:600]}")
        else:
            for doc_id in eligible:
                store = env.wm.doc_store.get(doc_id, {})
                snippet = store.get("full_text") or store.get("snippet") or ""
                judged_intent = env.memory_operator.relevance_annotations.get(doc_id, "")
                doc_lines.append(
                    f"{doc_id} | relevance_annotation={judged_intent or '(none)'} | "
                    f"text={snippet[:500]}"
                )
        if tool_name == "curate":
            prompt = (
                "ORIGINAL QUERY:\n" + env.wm.query
                + "\n\nCURRENT INTENT:\n" + str(env.wm.current_intent)
                + "\n\nALREADY CURATED IDS (keep unless clearly irrelevant):\n"
                + json.dumps(env.wm.curated_ids, ensure_ascii=False)
            )
        else:
            prompt = env.wm.to_text()
        if doc_lines:
            heading = (
                "DOCUMENTS FOUND SINCE LAST INTENT"
                if tool_name == "redirect" else "CURATION CANDIDATES"
            )
            prompt += f"\n\n{heading}:\n" + "\n".join(doc_lines)
        if env.wm.audit_feedback:
            prompt += "\n\nSUMMARY AUDITOR FEEDBACK:\n" + json.dumps(
                env.wm.audit_feedback, ensure_ascii=False
            )[:2500]
        prompt += f"\n\nREQUIRED POLICY ACTION:\n{instruction}"
        return prompt, eligible

    async def forced_tool_call(
        self, env: HarnessSearchEnv, tool_name: str
    ) -> TokensWithLogprobs:
        tool_schema = (
            REDIRECT_SCHEMA if tool_name == "redirect" else CURATE_SCHEMA
        )
        prompt, eligible = self._forced_tool_context(env, tool_name)
        properties = json.loads(json.dumps(tool_schema.parameters))
        if tool_name == "redirect":
            properties["doc_ids"]["maxItems"] = 5
            properties["reasoning"]["minLength"] = 24
            if eligible:
                properties["doc_ids"]["items"]["enum"] = eligible

        else:
            properties["add_ids"]["maxItems"] = 30
            if eligible:
                properties["add_ids"]["items"]["enum"] = eligible
            properties["remove_ids"]["maxItems"] = 5
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(tool_schema.required),
            "additionalProperties": False,
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the policy model at a mandatory tool checkpoint. Return only "
                        "the JSON arguments for the required tool. Do not answer the query."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": tool_name,
                    "strict": True,
                    "schema": schema,
                },
            },
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
        }
        if "gpt-oss" in self.model.lower():
            payload["reasoning_effort"] = "low"
        data = await asyncio.to_thread(
            self._post_json_url, self.chat_completions_url, payload
        )
        response_message = data["choices"][0]["message"]
        content = response_message.get("content") or ""
        if not content:
            final_payload = json.loads(json.dumps(payload))
            final_payload["messages"] = payload["messages"] + [{
                "role": "user",
                "content": (
                    "Your previous attempt stopped in internal reasoning. Emit the final "
                    "JSON arguments for the required tool now, with no prose outside JSON."
                ),
            }]
            final_data = await asyncio.to_thread(
                self._post_json_url, self.chat_completions_url, final_payload
            )
            content = final_data["choices"][0]["message"].get("content") or ""
            logger.info(
                "forced_policy_final_json_retry",
                qid=env.query_id,
                tool=tool_name,
                succeeded=bool(content),
            )
        if not content:
            content = "{}"
        try:
            params = json.loads(content)
        except json.JSONDecodeError as exc:
            params = json_repair.loads(content)
            logger.warning(
                "forced_tool_json_repaired",
                tool=tool_name,
                error=str(exc),
                output_chars=len(content),
                raw_output=content[:8000],
            )
        if not isinstance(params, dict):
            raise ValueError(f"Repaired {tool_name} arguments are not an object")
        if tool_name == "redirect":
            raw_doc_ids = params.get("doc_ids", [])
            if not isinstance(raw_doc_ids, list):
                raw_doc_ids = [raw_doc_ids] if raw_doc_ids else []
            cleaned_doc_ids = []
            valid_ids = set(eligible)
            for raw_id in raw_doc_ids:
                doc_id = env.wm._normalize_id(str(raw_id).strip())
                if doc_id in valid_ids and doc_id not in cleaned_doc_ids:
                    cleaned_doc_ids.append(doc_id)
                if len(cleaned_doc_ids) >= 5:
                    break
            params["doc_ids"] = cleaned_doc_ids
            reasoning = str(params.get("reasoning", "")).strip()
            if len(reasoning) < 24:
                reasoning = (
                    "Review the current active direction against this retrieval stage; "
                    "keep it only if evidence supports progress, otherwise pivot to the "
                    "most discriminating unresolved query constraint."
                )
            params["reasoning"] = reasoning
        if tool_name == "curate":
            def clean_ids(value, valid_ids, limit):
                if not isinstance(value, list):
                    value = [value] if value else []
                cleaned = []
                for raw_id in value:
                    doc_id = env.wm._normalize_id(str(raw_id).strip())
                    if doc_id and doc_id in valid_ids and doc_id not in cleaned:
                        cleaned.append(doc_id)
                    if len(cleaned) >= limit:
                        break
                return cleaned
            policy_add_ids = clean_ids(
                params.get("add_ids", []), set(env.wm.pool_ids), 60
            )
            # Require the planner to explicitly accept/reject each Relevance
            # Judge recommendation. This is still policy-owned curation: Judge
            # IDs are never copied directly into add_ids, and rejected items
            # stay out of the curated set.
            recommended_ids = [
                doc_id for doc_id in eligible
                if doc_id in env.memory_operator.relevance_annotations
                and doc_id not in policy_add_ids
            ][:30]
            if recommended_ids:
                decision_properties = {
                    doc_id: {
                        "type": "boolean",
                        "description": env.memory_operator.relevance_annotations[doc_id][:240],
                    }
                    for doc_id in recommended_ids
                }
                decision_payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are the planner reviewing Relevance Judge evidence for a "
                                "multi-document task. Decide every item independently. Accept "
                                "partial evidence, entity bridges, and corroborating sources; "
                                "reject only when the recommendation clearly conflicts with the "
                                "document/query. Return JSON only."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps({
                                "original_query": env.wm.query,
                                "current_intent": env.wm.current_intent,
                                "recommendations": [
                                    {
                                        "doc_id": doc_id,
                                        "intent": env.memory_operator.relevance_annotations[doc_id],
                                        "text": (
                                            env.wm.doc_store.get(doc_id, {}).get("full_text")
                                            or env.wm.doc_store.get(doc_id, {}).get("snippet")
                                            or ""
                                        )[:700],
                                    }
                                    for doc_id in recommended_ids
                                ],
                            }, ensure_ascii=False),
                        },
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "relevance_recommendation_decisions",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "decisions": {
                                        "type": "object",
                                        "properties": decision_properties,
                                        "required": recommended_ids,
                                        "additionalProperties": False,
                                    },
                                },
                                "required": ["decisions"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "temperature": 0.0,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                }
                if "gpt-oss" in self.model.lower():
                    decision_payload["reasoning_effort"] = "low"
                try:
                    decision_data = await asyncio.to_thread(
                        self._post_json_url, self.chat_completions_url,
                        decision_payload,
                    )
                    decision_content = (
                        decision_data["choices"][0]["message"].get("content") or ""
                    )
                    if not decision_content:
                        final_decision_payload = json.loads(json.dumps(decision_payload))
                        final_decision_payload["messages"] = decision_payload["messages"] + [{
                            "role": "user",
                            "content": (
                                "Your previous attempt stopped in internal reasoning. Emit the "
                                "final JSON decisions for every required doc_id now."
                            ),
                        }]
                        final_decision_data = await asyncio.to_thread(
                            self._post_json_url, self.chat_completions_url,
                            final_decision_payload,
                        )
                        decision_content = (
                            final_decision_data["choices"][0]["message"].get("content") or ""
                        )
                    decision_result = (
                        json_repair.loads(decision_content) if decision_content else {}
                    )
                    decisions = (
                        decision_result.get("decisions", {})
                        if isinstance(decision_result, dict) else {}
                    )
                    accepted_ids = [
                        doc_id for doc_id in recommended_ids
                        if decisions.get(doc_id) is True
                    ]
                    for doc_id in accepted_ids:
                        if doc_id not in policy_add_ids:
                            policy_add_ids.append(doc_id)
                    logger.info(
                        "forced_policy_relevance_decisions",
                        qid=env.query_id,
                        reviewed=len(recommended_ids),
                        accepted=len(accepted_ids),
                        raw=decision_content[:4000],
                    )
                except Exception as exc:
                    logger.warning(
                        "forced_policy_relevance_decisions_error",
                        qid=env.query_id,
                        error=str(exc)[:500],
                    )
            # v3.1 restores Harness-Search curation semantics: only documents
            # explicitly selected by the policy enter the curated set. The
            # relevance judge annotates candidates but never auto-adds them.
            params["add_ids"] = policy_add_ids
            importance = params.get("importance", {})
            if not isinstance(importance, dict):
                importance = {}
            params["importance"] = {
                doc_id: importance[doc_id]
                for doc_id in policy_add_ids if doc_id in importance
            }
            params["remove_ids"] = clean_ids(
                params.get("remove_ids", []), set(env.wm.curated_ids), 5
            )
            if len(policy_add_ids) < 10 and len(eligible) >= 10:
                combined_add_ids = list(policy_add_ids)
                last_retry_params = None
                for review_index in range(3):
                    remaining = [
                        doc_id for doc_id in eligible
                        if doc_id not in combined_add_ids
                    ]
                    if len(combined_add_ids) >= 10 or not remaining:
                        break
                    retry_payload = json.loads(json.dumps(payload))
                    retry_payload["messages"] = [
                        {
                            "role": "system",
                            "content": (
                                "You are the policy model performing an independent high-recall "
                                "curation review. Return only JSON arguments for curate. Select "
                                "legal IDs yourself; the runtime will not add any IDs for you."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt + "\n\nINDEPENDENT RE-REVIEW:\n" + (
                                "Select additional distinct evidence candidates. Do not repeat "
                                f"these IDs already selected by you: {combined_add_ids}. A "
                                "document need not satisfy the whole query: include partial "
                                "support, entity bridges, duplicate corroboration, and Relevance "
                                "Judge recommendations. Return JSON arguments only."
                            ),
                        },
                    ]
                    retry_schema = retry_payload["response_format"]["json_schema"]["schema"]
                    retry_schema["properties"]["add_ids"]["items"]["enum"] = remaining
                    retry_schema["properties"]["add_ids"]["minItems"] = min(10, len(remaining))
                    try:
                        retry_data = await asyncio.to_thread(
                            self._post_json_url, self.chat_completions_url, retry_payload
                        )
                    except Exception as exc:
                        # Local gpt-oss can intermittently reject a valid
                        # multi-message Harmony history. Preserve the policy's
                        # selections from earlier passes instead of failing the
                        # entire benchmark episode.
                        logger.warning(
                            "forced_policy_curate_self_revision_error",
                            qid=env.query_id,
                            review=review_index + 1,
                            error=str(exc)[:500],
                        )
                        break
                    retry_content = retry_data["choices"][0]["message"].get("content") or "{}"
                    try:
                        retry_params = json.loads(retry_content)
                    except json.JSONDecodeError:
                        retry_params = json_repair.loads(retry_content)
                    if not isinstance(retry_params, dict):
                        continue
                    retry_add_ids = clean_ids(
                        retry_params.get("add_ids", []), set(remaining), 30
                    )
                    logger.info(
                        "forced_policy_curate_self_revision",
                        qid=env.query_id,
                        review=review_index + 1,
                        already_selected=len(combined_add_ids),
                        newly_selected=len(retry_add_ids),
                        raw=retry_content[:4000],
                    )
                    for doc_id in retry_add_ids:
                        if doc_id not in combined_add_ids:
                            combined_add_ids.append(doc_id)
                    last_retry_params = retry_params
                if len(combined_add_ids) > len(policy_add_ids):
                    params["add_ids"] = combined_add_ids[:30]
                    if isinstance(last_retry_params, dict):
                        params["remove_ids"] = clean_ids(
                            last_retry_params.get("remove_ids", []),
                            set(env.wm.curated_ids), 5,
                        )
        message = (
            Message.from_role_and_content(Role.ASSISTANT, json.dumps(params))
            .with_channel("commentary")
            .with_recipient("functions." + tool_name)
            .with_content_type("<|constrain|>json")
        )
        tokens = env.enc.render_conversation(Conversation.from_messages([message]))
        selected = params.get("doc_ids", params.get("add_ids", []))
        logger.info(
            "forced_policy_tool_call",
            qid=env.query_id,
            tool=tool_name,
            eligible_docs=len(eligible),
            selected_docs=len(selected),
        )
        return TokensWithLogprobs(tokens=tokens, maybe_logprobs=None)

    def _post_json_url(self, url: str, payload: Dict) -> Dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {exc.code}: {detail[:1000]}") from exc

    def _post_json(self, payload: Dict) -> Dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.completions_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {exc.code}: {detail[:1000]}") from exc


class ChatRetrievalPolicy(RetrievalPolicy):
    """Qwen chat policy adapter that emits actions in the environment's Harmony format."""

    def __init__(self, *, base_url: str, model: str, fallback_model: str,
                 api_key: str, max_tokens: int, temperature: float,
                 top_p: float, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback_model = fallback_model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

    @property
    def chat_completions_url(self) -> str:
        return self.base_url + ("/chat/completions" if self.base_url.endswith("/v1") else "/v1/chat/completions")

    def _post(self, payload: Dict) -> Dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.chat_completions_url, data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
                "api-key": self.api_key,
            }, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qwen HTTP {exc.code}: {detail[:1000]}") from exc

    def _request_json(self, messages: List[Dict], max_tokens: int | None = None) -> Dict:
        last_error = None
        for model in dict.fromkeys((self.model, self.fallback_model)):
            payload = {
                "model": model,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": max_tokens or self.max_tokens,
                "response_format": {"type": "json_object"},
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if "dashscope.aliyuncs.com" in self.base_url:
                payload["enable_thinking"] = False
                payload.pop("response_format", None)
            if "gpt-oss" in model.lower():
                payload["reasoning_effort"] = "low"
            try:
                data = self._post(payload)
                response_message = data["choices"][0]["message"]
                content = response_message.get("content") or ""
                if not content:
                    logger.warning(
                        "chat_policy_reasoning_fallback",
                        model=model,
                        finish_reason=data["choices"][0].get("finish_reason"),
                        reasoning=str(response_message.get("reasoning", ""))[:4000],
                    )
                    retry_payload = dict(payload)
                    retry_payload["messages"] = messages + [{
                        "role": "user",
                        "content": (
                            "Your previous attempt stopped in internal reasoning. Emit the "
                            "final JSON object now. It must contain tool_name, arguments, "
                            "and reasoning, with no text outside the JSON."
                        ),
                    }]
                    retry_payload["temperature"] = 0.0
                    retry_data = self._post(retry_payload)
                    retry_message = retry_data["choices"][0]["message"]
                    retry_content = retry_message.get("content") or ""
                    if retry_content:
                        retry_parsed = json_repair.loads(retry_content)
                        if isinstance(retry_parsed, dict) and retry_parsed:
                            logger.info("chat_policy_final_json_retry_succeeded", model=model)
                            return retry_parsed
                    # gpt-oss can stop after a short internal decision such as
                    # "Let's search" without emitting its final JSON. Return a
                    # marker object so action() deterministically selects the
                    # current intent's next retrieval query instead of failing
                    # the episode or sending malformed Harmony tokens.
                    return {
                        "_reasoning_only": True,
                        "reasoning": str(response_message.get("reasoning", "")),
                    }
                parsed = json_repair.loads(content)
                if isinstance(parsed, dict):
                    return parsed
                retry_payload = dict(payload)
                retry_payload["messages"] = messages + [{
                    "role": "user",
                    "content": (
                        "The previous response was not a JSON object. Emit the final JSON "
                        "object now with tool_name, arguments, and reasoning, and no prose."
                    ),
                }]
                retry_payload["temperature"] = 0.0
                retry_data = self._post(retry_payload)
                retry_content = retry_data["choices"][0]["message"].get("content") or ""
                if retry_content:
                    retry_parsed = json_repair.loads(retry_content)
                    if isinstance(retry_parsed, dict) and retry_parsed:
                        logger.info("chat_policy_non_object_retry_succeeded", model=model)
                logger.warning(
                    "qwen_policy_non_object_response",
                    model=model,
                    parsed_type=type(parsed).__name__,
                    content_preview=repr(content[:1000]),
                )
                raise ValueError("policy response is not an object")
            except Exception as exc:
                last_error = exc
                logger.warning("qwen_policy_retry", model=model, error=str(exc)[:240])
                # The local gpt-oss Harmony parser can intermittently reject a
                # valid multi-turn tool history with "unexpected tokens
                # remaining in message header". This is not a policy decision
                # failure, so let action() take its deterministic retrieval
                # fallback instead of failing the entire benchmark episode.
                if "unexpected tokens remaining in message header" in str(exc):
                    return {
                        "_reasoning_only": True,
                        "reasoning": "local gpt-oss Harmony history parse fallback",
                    }
        raise RuntimeError(f"all Qwen policy models failed: {last_error}")

    @staticmethod
    def _to_tokens(env: HarnessSearchEnv, tool_name: str,
                   arguments: Dict) -> TokensWithLogprobs:
        message = (
            Message.from_role_and_content(Role.ASSISTANT, json.dumps(arguments))
            .with_channel("commentary")
            .with_recipient("functions." + tool_name)
            .with_content_type("<|constrain|>json")
        )
        tokens = env.enc.render_conversation(Conversation.from_messages([message]))
        return TokensWithLogprobs(tokens=tokens, maybe_logprobs=None)

    def _context(self, env: HarnessSearchEnv, required_tool: str | None = None) -> List[Dict]:
        schemas = []
        for name, tool in env.policy_toolset().tools.items():
            if isinstance(tool, UserTextTool):
                continue
            schema = tool.tool_schema
            schemas.append({
                "name": name,
                "description": schema.description,
                "parameters": schema.parameters,
                "required": schema.required,
            })
        recent = env._result_summaries[-3:]
        instruction = (
            "You are a retrieval policy. Choose exactly one available tool. Return one "
            "JSON object only with this shape: "
            "{tool_name:string, arguments:object, reasoning:string}. Do not emit a native "
            "tool call, Harmony headers, analysis-only output, or prose outside the JSON. "
            "Use search tools to gather evidence, curate relevant documents at checkpoints, "
            "and end only when the curated evidence is sufficient."
        )
        if required_tool:
            instruction += f" You must choose {required_tool}."
        elif env.harness.retrieval_required:
            instruction += (
                " The environment is at a mandatory retrieval stage. You must choose "
                "search, grep_corpus, fan_out_search, or read now; do not "
                "choose curate, redirect, or end."
            )
        user_payload = {
            "original_query": env.wm.query,
            "working_memory": env.wm.to_text(),
            "recent_observations": recent,
            "available_tools": schemas,
            "action_constraint": env.harness.action_hint(),
            "summary_auditor_feedback": env.wm.audit_feedback,
        }
        if required_tool == "curate":
            uncurated_ids = [
                doc_id for doc_id in env.wm.pool_ids
                if doc_id not in env.wm.curated_ids
            ]
            reviewed_ids = getattr(env, "_policy_curation_reviewed_ids", set())
            if not isinstance(reviewed_ids, set):
                reviewed_ids = set(reviewed_ids)
            unreviewed_ids = [
                doc_id for doc_id in uncurated_ids if doc_id not in reviewed_ids
            ]
            if not unreviewed_ids and uncurated_ids:
                reviewed_ids.clear()
                unreviewed_ids = list(uncurated_ids)
            judged_ids = [
                doc_id for doc_id in unreviewed_ids
                if doc_id in env.memory_operator.relevance_annotations
            ]
            candidate_ids = []
            for doc_id in judged_ids + unreviewed_ids:
                if doc_id not in candidate_ids:
                    candidate_ids.append(doc_id)
            candidate_ids = candidate_ids[:60]
            reviewed_ids.update(candidate_ids)
            env._policy_curation_reviewed_ids = reviewed_ids
            env._last_curation_candidate_ids = list(candidate_ids)
            candidates = []
            for doc_id in candidate_ids:
                store = env.wm.doc_store.get(doc_id, {})
                candidates.append({
                    "doc_id": doc_id,
                    "relevance_annotation": env.memory_operator.relevance_annotations.get(doc_id, ""),
                    "snippet": (
                        store.get("full_text") or store.get("snippet") or ""
                    )[:500],
                })
            instruction += (
                " This is multi-document evidence curation: the selected documents jointly "
                "answer the query. NEVER require one document to satisfy every constraint. "
                "Review every curation_candidate and put every document that plausibly "
                "supports even one original-query constraint, entity bridge, or source clue "
                "in arguments.add_ids. Usually select "
                "10-25 IDs when that many useful candidates exist, with a hard maximum of 30; "
                "do not stop after finding only the single best document. Include complementary "
                "and borderline evidence because incomplete evidence can still be curated. "
                "Leave add_ids empty only when every candidate is unrelated. Do not invent IDs."
            )
            user_payload["curation_candidates"] = candidates
            user_payload["relevance_judge_recommended_ids"] = [
                doc_id for doc_id in candidate_ids
                if doc_id in env.memory_operator.relevance_annotations
            ]
            if env.summary_auditor.latest:
                user_payload["summary_auditor_feedback"] = env.summary_auditor.latest
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    async def action(self, env: HarnessSearchEnv) -> TokensWithLogprobs:
        result = await asyncio.wait_for(
            asyncio.to_thread(self._request_json, self._context(env)),
            timeout=self.timeout + 30,
        )
        tool_name = str(
            result.get("tool_name") or result.get("tool") or result.get("name")
            or (result.get("action") if isinstance(result.get("action"), str) else "")
            or ""
        ).strip()
        function = result.get("function")
        if isinstance(function, dict):
            tool_name = tool_name or str(function.get("name", "")).strip()
        nested_tool = None
        for alias in ("tool_use", "tool_call", "action"):
            value = result.get(alias)
            if isinstance(value, dict):
                nested_tool = value
                tool_name = tool_name or str(
                    value.get("tool_name") or value.get("tool")
                    or value.get("name") or value.get("action") or ""
                ).strip()
                break
            if isinstance(value, str) and value:
                tool_name = tool_name or value.strip()
        known_tools = dict(env._build_full_toolset().tools)
        for legacy, canonical in LEGACY_TOOL_NAMES.items():
            if canonical in known_tools:
                known_tools[legacy] = known_tools[canonical]
        if tool_name not in known_tools:
            normalized = re.sub(r"_\d+$", "", tool_name)
            if normalized in known_tools:
                tool_name = normalized
        if not tool_name:
            wrapped = [
                key for key in result
                if key in known_tools or re.sub(r"_\d+$", "", key) in known_tools
            ]
            if len(wrapped) == 1:
                tool_name = re.sub(r"_\d+$", "", wrapped[0])
        arguments = result.get("arguments") or result.get("parameters")
        if arguments is None and isinstance(function, dict):
            arguments = function.get("arguments")
        if arguments is None and isinstance(nested_tool, dict):
            arguments = nested_tool.get("arguments") or nested_tool.get("parameters")
        if arguments is None and tool_name in result and isinstance(result[tool_name], dict):
            arguments = result[tool_name]
        if arguments is None:
            for key, value in result.items():
                if re.sub(r"_\d+$", "", key) == tool_name and isinstance(value, dict):
                    arguments = value
                    break
        if tool_name not in known_tools and "search" in env.harness.allowed_tools:
            tool_name = "search"
            try:
                intent = json.loads(env.wm.current_intent)
                directions = intent.get("searchable_directions", [])
                index = min(env.harness.searches_since_curate, max(len(directions) - 1, 0))
                fallback_query = str(directions[index]) if directions else env.wm.query
            except Exception:
                fallback_query = env.wm.query
            arguments = {"query": fallback_query}
            logger.warning(
                "qwen_policy_retrieval_fallback", qid=env.query_id,
                response_keys=list(result)[:12],
            )
        if not isinstance(arguments, dict):
            arguments = result if any(key in result for key in ("query", "queries", "doc_ids", "add_ids")) else {}
        return self._to_tokens(env, canonical_tool_name(tool_name), arguments)

    async def forced_tool_call(self, env: HarnessSearchEnv,
                               tool_name: str) -> TokensWithLogprobs:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                self._request_json, self._context(env, tool_name), 1200
            ),
            timeout=self.timeout + 30,
        )
        candidate_ids = list(
            getattr(env, "_last_curation_candidate_ids", [])
        )
        raw_arguments = result.get("arguments", {})
        raw_add_ids = (
            raw_arguments.get("add_ids", [])
            if isinstance(raw_arguments, dict) else []
        )
        explicit_candidate_adds = {
            str(doc_id) for doc_id in raw_add_ids
            if str(doc_id) in set(candidate_ids)
        }
        needs_compact_curate = (
            tool_name == "curate"
            and (
                result.get("_reasoning_only")
                or (
                    len(candidate_ids) >= 10
                    and len(explicit_candidate_adds) < 10
                )
            )
        )
        if needs_compact_curate:
            # A full working-memory prompt occasionally leaves gpt-oss in a
            # one-line reasoning-only state (for example, "Need curate"). Give
            # the same policy model one compact selection pass. The runtime
            # still does not choose IDs: the model must explicitly return them.
            if not candidate_ids:
                candidate_ids = [
                    doc_id for doc_id in env.wm.pool_ids
                    if doc_id not in env.wm.curated_ids
                ][:60]
            compact_candidates = []
            for doc_id in candidate_ids:
                store = env.wm.doc_store.get(doc_id, {})
                compact_candidates.append({
                    "doc_id": doc_id,
                    "intent": env.memory_operator.relevance_annotations.get(doc_id, ""),
                    "snippet": (
                        store.get("full_text") or store.get("snippet") or ""
                    )[:220],
                })
            compact_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are the curation policy. Return JSON only as "
                        "{tool_name:'curate',arguments:{add_ids:[],remove_ids:[]},reasoning:''}. "
                        "The documents form a joint multi-hop evidence set; never require one "
                        "document to satisfy the entire query. Select every candidate that "
                        "plausibly supports any one original-query "
                        "constraint, usually 10-25 and at most 30. Candidates with a non-empty "
                        "Relevance Judge intent are recommended: include them unless that intent "
                        "clearly contradicts the original query. Your previous selection was "
                        "too narrow; when at least 10 candidates are plausibly related, explicitly "
                        "select at least 10 IDs. Favor evidence recall and independent supporting "
                        "sources. Do not invent IDs."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "original_query": env.wm.query,
                        "current_intent": env.wm.current_intent,
                        "summary_auditor_feedback": env.summary_auditor.latest,
                        "relevance_judge_recommended_ids": [
                            doc_id for doc_id in candidate_ids
                            if doc_id in env.memory_operator.relevance_annotations
                        ],
                        "candidates": compact_candidates,
                    }, ensure_ascii=False),
                },
            ]
            compact_result = await asyncio.to_thread(
                self._request_json, compact_messages, 2400
            )
            logger.info(
                "forced_chat_policy_compact_retry",
                qid=env.query_id,
                result=json.dumps(compact_result, ensure_ascii=False)[:4000],
            )
            if not compact_result.get("_reasoning_only"):
                result = compact_result
            else:
                result["reasoning"] = (
                    str(result.get("reasoning", "")) + " "
                    + str(compact_result.get("reasoning", ""))
                ).strip()
        logger.info(
            "forced_chat_policy_result",
            qid=env.query_id,
            required_tool=tool_name,
            result=json.dumps(result, ensure_ascii=False)[:4000],
        )
        function = result.get("function")
        arguments = result.get("arguments") or result.get("parameters")
        if arguments is None and isinstance(function, dict):
            arguments = function.get("arguments")
        if arguments is None:
            for alias in ("tool_use", "tool_call", "action"):
                nested = result.get(alias)
                if isinstance(nested, dict):
                    arguments = nested.get("arguments") or nested.get("parameters")
                    if arguments is not None:
                        break
        if arguments is None and tool_name in result and isinstance(result[tool_name], dict):
            arguments = result[tool_name]
        if arguments is None:
            arguments = result
        if not isinstance(arguments, dict):
            arguments = {}
        if tool_name == "curate":
            def clean_ids(value, valid_ids, limit):
                if not isinstance(value, list):
                    value = [value] if value else []
                cleaned = []
                for raw_id in value:
                    doc_id = env.wm._normalize_id(str(raw_id).strip())
                    if doc_id in valid_ids and doc_id not in cleaned:
                        cleaned.append(doc_id)
                    if len(cleaned) >= limit:
                        break
                return cleaned
            raw_add_ids = arguments.get("add_ids", [])
            # gpt-oss occasionally makes an explicit curation decision in its
            # reasoning channel but omits the final JSON. Recover only IDs that
            # the policy itself named and that are present in the pool; never
            # fill from the candidate list automatically.
            if not raw_add_ids and result.get("_reasoning_only"):
                reasoning = str(result.get("reasoning", ""))
                raw_add_ids = re.findall(r"(?<!\d)\d{2,}(?!\d)", reasoning)
            policy_ids = clean_ids(raw_add_ids, set(env.wm.pool_ids), 30)
            # Match the original Harness-Search contract: curate exactly the policy
            # selection, with no top-K pool floor or read-document auto-add.
            arguments["add_ids"] = policy_ids
            importance = arguments.get("importance", {})
            if not isinstance(importance, dict):
                importance = {}
            arguments["importance"] = {
                doc_id: importance[doc_id]
                for doc_id in policy_ids if doc_id in importance
            }
            arguments["remove_ids"] = clean_ids(
                arguments.get("remove_ids", []), set(env.wm.curated_ids), 5
            )
        # Ignore an incorrect tool selection but retain its arguments; the JSON
        # schema shown in context gives the model the required parameter names.
        return self._to_tokens(env, tool_name, arguments)
