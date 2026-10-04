"""Shared JSON transport; contains no memory or termination decisions."""
from typing import Any, Dict, Optional
import json
import structlog
from harness_search.model_settings import (
    INTENT_MODEL_API_KEY, INTENT_MODEL_BASE_URL, INTENT_MODEL_HEADER_API_KEY,
    INTENT_MODEL_TIMEOUT, RELEVANCE_JUDGE_FALLBACK_NAME,
    SUMMARY_AUDITOR_MODEL_NAME, SUMMARY_AUDITOR_FALLBACK_NAME,
)

logger = structlog.get_logger(__name__)

class JudgeClient:
    def __init__(self, query_id: str):
        self.query_id = query_id

    def complete(self, model: str, system: str, user: str) -> Dict[str, Any]:
        """Call a v3 judge with deterministic JSON output and one model fallback."""
        if not INTENT_MODEL_API_KEY:
            raise RuntimeError("INTENT_MODEL_API_KEY is not configured")
        from openai import OpenAI
        client = OpenAI(
            base_url=INTENT_MODEL_BASE_URL,
            api_key=INTENT_MODEL_API_KEY,
            default_headers={"api-key": INTENT_MODEL_HEADER_API_KEY},
            timeout=INTENT_MODEL_TIMEOUT,
        )
        last_error: Optional[Exception] = None
        fallback_model = (
            SUMMARY_AUDITOR_FALLBACK_NAME
            if model == SUMMARY_AUDITOR_MODEL_NAME
            else RELEVANCE_JUDGE_FALLBACK_NAME
        )
        for candidate in dict.fromkeys((model, fallback_model)):
            try:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                response = client.chat.completions.create(
                    model=candidate,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1600,
                    response_format={"type": "json_object"},
                    extra_body=(
                        {"reasoning_effort": "low"}
                        if "gpt-oss" in candidate.lower()
                        else ({"chat_template_kwargs": {"enable_thinking": False}}
                              if "qwen" in candidate.lower() else None)
                    ),
                )
                message = response.choices[0].message
                content = message.content or ""
                if not content:
                    logger.warning(
                        "judge_reasoning_fallback",
                        qid=self.query_id,
                        model=candidate,
                        reasoning=str(getattr(message, "reasoning", ""))[:2000],
                    )
                    retry = client.chat.completions.create(
                        model=candidate,
                        messages=messages + [{
                            "role": "user",
                            "content": (
                                "Your previous attempt stopped in internal reasoning. "
                                "Emit the requested final JSON object now, with no text "
                                "outside the JSON."
                            ),
                        }],
                        temperature=0.0,
                        max_tokens=2000,
                        response_format={"type": "json_object"},
                        extra_body=(
                            {"reasoning_effort": "low"}
                            if "gpt-oss" in candidate.lower()
                            else ({"chat_template_kwargs": {"enable_thinking": False}}
                                  if "qwen" in candidate.lower() else None)
                        ),
                    )
                    content = retry.choices[0].message.content or ""
                if not content:
                    raise ValueError("judge returned reasoning only without final JSON")
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("judge response is not a JSON object")
                return parsed
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "judge_model_retry", qid=self.query_id,
                    model=candidate, error=str(exc)[:200],
                )
        raise RuntimeError(f"all judge models failed: {last_error}")
