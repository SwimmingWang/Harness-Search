# Architecture and execution semantics

| Method term | Implementation | Responsibility |
|---|---|---|
| Retrieval Policy | `harness_search/retrieval_policy.py` | `RetrievalPolicy` with chat and vLLM adapters; proposes tools and document selections |
| Memory Operator | `harness_search/memory_operator.py`, with an internal annotation helper in `harness_search/relevance_judge.py` | Per-query stateful agent; annotates documents, updates intent, and exclusively commits the candidate pool, curated evidence and progress records |
| Summary Auditor | `harness_search/summary_auditor.py` | Audits a detached view of curated evidence; cannot mutate working memory |
| Harness | `runtime/environment.py` | `HarnessSearchEnv` enforces action checkpoints, budgets, termination and rewards |

The three agent roles are Retrieval Policy, Memory Operator, and Summary Auditor.
The harness enforces their execution protocol. Memory Operator is a stateful
agent with two internal model-assisted tasks: document annotation and intent
updates. The names Intent Planner and Relevance Judge refer to these tasks and
their implementation helpers, not additional agents with separate authority.

For newly encountered documents, Memory Operator invokes its annotation helper
in batches of up to five documents and records their query-relative support
annotations. These recommendations do not automatically curate evidence. For
initialization or an accepted `redirect` request, it constructs a structured
intent update; policy reasoning and up to five policy-selected documents ground
the update. Validation and state commitment are implemented in code.

Memory Operator accepts the policy's selected operations, commits retrieval
observations, and validates curation. It has no independent retrieval loop and
cannot approve termination. The two internal operations share one fixed System prompt and select their task
through the User message operation field. Each call retains its task-specific
inputs and invocation time; document annotation and intent updating remain
separate model calls.
The policy and harness access `MemoryView`; mutable fields returned by this view
are detached copies, so callers cannot directly change committed memory.

The canonical actions are `search`, `read`, `redirect`, `curate`, and `end`.
`fan_out_search` and `grep_corpus` remain concrete search variants; `review_docs`
is a memory-only read variant. Previously served checkpoints may still emit
`search_corpus`, `read_document`, `revise_intent`, or `end_search`: input aliases
resolve these names to the canonical actions without advertising duplicate tools.

The current execution rules are preserved:

- Initial intent construction and each successful redirect require retrieval next.
- The stage search counter triggers mandatory curation at its configured limit;
  both curation and successful redirection reset that counter. Document reads do
  not consume the search limit, and fan-out counts as one stage operation.
- Curation requires a redirect next, except when terminal processing takes priority.
- Summary Auditor runs at attempted termination and at the turn limit, not after
  ordinary curation. It retains the original three verdicts: `answer_ready`,
  `revise_summary`, and `search_more`.
- Budget exhaustion can terminate with insufficient evidence; early termination
  can be blocked up to eight times; auditor errors retain the original terminal
  fail-open behavior. These are operational exits, not evidence certifications.
- Memory Operator's intent-update and document-annotation tasks share the
  configured model and fixed System prompt, with separate request contexts and
  operation-specific User payloads.
  Grouping them under one agent does not add or merge model calls.

## Suggested Method wording

The Memory Operator is a stateful agent with exclusive authority to commit
working-memory updates. It performs two internal model-assisted tasks:
query-relative annotation of newly retrieved documents and structured intent
updates at initialization and redirection. For a policy-proposed redirection,
the operator uses the policy's rationale, selected documents, and existing search
state to construct the intent update. Its commit interface validates identifiers,
evidence capacity, and update structure before writing the corresponding state.
The Retrieval Policy retains authority over the next retrieval operation, while
the Summary Auditor independently evaluates evidence sufficiency for termination.

## Configuration and compatibility

New configuration names are `RELEVANCE_JUDGE_*`, `SUMMARY_AUDITOR_*`,
`HARNESS_SEARCH_QUERY_DATA_ROOT`, and `V8D_REDIRECT_TOOL`. Their previous names
(`REFERENCE_JUDGE_*`, `SUMMARY_JUDGE_*`, `INTENT_CURATOR_QUERY_DATA_ROOT`, and
`V8D_REVISE_INTENT_TOOL`) remain accepted as fallbacks. New result records use
`relevance_judge_calls`, `summary_auditor_calls`, and `latest_audit`; historical
outputs retain their original field names.
