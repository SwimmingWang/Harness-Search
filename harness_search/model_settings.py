"""Model settings for Memory Operator, Relevance Judge and Summary Auditor.

Legacy environment variable names remain valid for existing launch configurations.
"""

import os

ABLATE_REVIEW_DOCS_UNAVAILABLE = os.environ.get("ABLATE_REVIEW_DOCS_UNAVAILABLE", "0") == "1"
INTENT_MODEL_BASE_URL = os.environ.get(
    "INTENT_MODEL_BASE_URL",
    os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8000/v1"),
).rstrip("/")
# The relevance judge and intent planner intentionally share one model.
# Legacy INTENT_MODEL_* names remain accepted as configuration fallbacks.
RELEVANCE_JUDGE_MODEL_NAME = os.environ.get(
    "RELEVANCE_JUDGE_MODEL_NAME",
    os.environ.get("REFERENCE_JUDGE_MODEL_NAME", os.environ.get("INTENT_MODEL_NAME", "Qwen/Qwen3.6-27B")),
)
RELEVANCE_JUDGE_FALLBACK_NAME = os.environ.get(
    "RELEVANCE_JUDGE_FALLBACK_NAME",
    os.environ.get("REFERENCE_JUDGE_FALLBACK_NAME", os.environ.get("INTENT_MODEL_FALLBACK_NAME", "Qwen/Qwen3.6-27B-FP8")),
)
INTENT_MODEL_NAME = RELEVANCE_JUDGE_MODEL_NAME  # Backward-compatible alias.
INTENT_MODEL_FALLBACK_NAME = RELEVANCE_JUDGE_FALLBACK_NAME
SUMMARY_AUDITOR_MODEL_NAME = os.environ.get(
    "SUMMARY_AUDITOR_MODEL_NAME", os.environ.get("SUMMARY_JUDGE_MODEL_NAME", "Qwen/Qwen3.6-27B")
)
SUMMARY_AUDITOR_FALLBACK_NAME = os.environ.get(
    "SUMMARY_AUDITOR_FALLBACK_NAME", os.environ.get("SUMMARY_JUDGE_FALLBACK_NAME", SUMMARY_AUDITOR_MODEL_NAME)
)
V3_JUDGES_ENABLED = os.environ.get("V3_JUDGES_ENABLED", "1") == "1"
SUMMARY_MIN_RETRIEVAL_CALLS = int(
    os.environ.get("SUMMARY_MIN_RETRIEVAL_CALLS", "5")
)
INTENT_MODEL_TIMEOUT = float(os.environ.get("INTENT_MODEL_TIMEOUT", "120"))
INTENT_MODEL_MAX_TOKENS = int(os.environ.get("INTENT_MODEL_MAX_TOKENS", "2600"))
INTENT_MODEL_RETRY_MAX_TOKENS = int(
    os.environ.get("INTENT_MODEL_RETRY_MAX_TOKENS", "1800")
)
INTENT_MODEL_API_KEY = os.environ.get("INTENT_MODEL_API_KEY", "").strip()
INTENT_MODEL_HEADER_API_KEY = os.environ.get(
    "INTENT_MODEL_HEADER_API_KEY", INTENT_MODEL_API_KEY
).strip()
