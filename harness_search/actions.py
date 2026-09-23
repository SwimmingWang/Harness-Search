"""Method action names and compatibility with previously served checkpoints."""

LEGACY_TOOL_NAMES = {
    "search_corpus": "search",
    "read_document": "read",
    "revise_intent": "redirect",
    "end_search": "end",
}


def canonical_tool_name(name: str) -> str:
    return LEGACY_TOOL_NAMES.get(name, name)


# Concrete search/read variants remain available to preserve retrieval behavior.
METHOD_ACTIONS = {
    "search": "search",
    "fan_out_search": "search",
    "grep_corpus": "search",
    "read": "read",
    "review_docs": "read",
    "redirect": "redirect",
    "curate": "curate",
    "end": "end",
}
