"""Tools over the document bundle supplied with each LongSeal example."""

import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import PrivateAttr

from harness_search.tools import (
    GREP_CORPUS_SCHEMA,
    READ_SCHEMA,
    SEARCH_SCHEMA,
    GrepCorpusToolCallMetadata,
    PruneChunksTool,
    SearchToolCallMetadata,
    Tool,
    ToolCallMetadata,
    ToolSet,
)


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class LongSealSearchTool(Tool):
    _documents: List[dict] = PrivateAttr()

    def __init__(self, documents: List[dict]):
        super().__init__(tool_schema=SEARCH_SCHEMA)
        self._documents = documents

    def __call__(self, params: Dict[Any, Any], overrides=None):
        query_terms = _terms(str(params["query"]))
        ignored = set((overrides or {}).get("ignore_ids", []))
        ranked = []
        for position, doc in enumerate(self._documents):
            doc_id = str(doc["url"])
            if doc_id in ignored:
                continue
            title = str(doc.get("title") or "")
            body = str(doc.get("text") or "")
            score = 3 * len(query_terms & _terms(title)) + len(query_terms & _terms(body[:12000]))
            ranked.append((score, -position, doc_id, title, body))
        ranked.sort(reverse=True)
        selected = ranked[:10]
        output = "\n".join(
            f"# DOCUMENT ID: {doc_id}\n# {title}\n{body[:2048]}"
            for _, _, doc_id, title, body in selected
        ) or "No results found"
        ids = [item[2] for item in selected]
        return output, SearchToolCallMetadata(returned_chunk_ids=ids)


class LongSealReadTool(Tool):
    _by_id: Dict[str, dict] = PrivateAttr()

    def __init__(self, documents: List[dict]):
        super().__init__(tool_schema=READ_SCHEMA)
        self._by_id = {str(doc["url"]): doc for doc in documents}

    def __call__(self, params: Dict[Any, Any], overrides=None):
        doc_id = str(params.get("doc_id", params.get("id", "")))
        doc = self._by_id.get(doc_id)
        if doc is None:
            raise ValueError(f"Document not found: {doc_id}")
        return f"# {doc.get('title', '')}\n{doc.get('text', '')}", None


class LongSealGrepTool(Tool):
    _documents: List[dict] = PrivateAttr()

    def __init__(self, documents: List[dict]):
        super().__init__(tool_schema=GREP_CORPUS_SCHEMA)
        self._documents = documents

    def __call__(self, params: Dict[Any, Any], overrides=None):
        pattern = str(params["pattern"])
        try:
            expression = re.compile(pattern, re.IGNORECASE)
        except re.error:
            expression = re.compile(re.escape(pattern), re.IGNORECASE)
        matches = [doc for doc in self._documents if expression.search(str(doc.get("text", "")))][:5]
        ids = [str(doc["url"]) for doc in matches]
        output = "\n".join(
            f"# DOCUMENT ID: {doc['url']}\n# {doc.get('title', '')}\n{str(doc.get('text', ''))[:2048]}"
            for doc in matches
        ) or "No results found"
        return output, GrepCorpusToolCallMetadata(returned_chunk_ids=ids)


def create_longseal_toolset(documents: List[dict]) -> Tuple[ToolSet, Tool]:
    search = LongSealSearchTool(documents)
    toolset = ToolSet(name="longseal_query_toolset")
    toolset.add_tool(search)
    toolset.add_tool(LongSealReadTool(documents))
    toolset.add_tool(LongSealGrepTool(documents))
    toolset.add_tool(PruneChunksTool())
    return toolset, search
