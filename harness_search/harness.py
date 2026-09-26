"""Propose-Commit-Audit loop (paper Section 3 and Algorithm 1).

This controller knows neither model protocols nor benchmark labels. Policy
adapters submit one proposal; only MemoryOperator commits state, and only an
accepted SummaryAuditor verdict permits evidence-gated termination.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from harness_search.actions import METHOD_ACTIONS, canonical_tool_name
from harness_search.contracts import ActionProposal, HarnessBudget, OperationResult

if TYPE_CHECKING:
    from harness_search.memory_operator import MemoryOperator
    from harness_search.summary_auditor import SummaryAuditor


class HarnessSearch:
    def __init__(self, memory: MemoryOperator, auditor: SummaryAuditor,
                 budget: HarnessBudget | None = None, *, audit_enabled: bool = True):
        self.memory = memory
        self.auditor = auditor
        self.budget = budget or HarnessBudget()
        self.audit_enabled = audit_enabled
        self.memory.configure(evidence_capacity=self.budget.evidence_capacity)
        self.turns = 0
        self.search_calls = 0
        self.searches_since_curate = 0
        self.curate_calls = 0
        self.redirect_calls = 0
        self.audit_rejections = 0
        self.last_action = None
        self.redirect_required = False
        self.retrieval_required = False
        self.termination_reason = None

    @property
    def done(self) -> bool:
        return self.termination_reason is not None

    @property
    def allowed_actions(self) -> frozenset[str]:
        if self.done:
            return frozenset()
        if self.redirect_required:
            return frozenset({"redirect"})
        if self.searches_since_curate >= self.budget.searches_per_curation:
            return frozenset({"curate"})
        if self.last_action == "curate":
            return frozenset({"redirect", "end"})
        # Appendix E.2: a committed redirect is followed by retrieval.
        if self.retrieval_required:
            return frozenset({"search", "read"})
        return frozenset(METHOD_ACTIONS.values())

    @property
    def allowed_tools(self) -> frozenset[str]:
        return frozenset(name for name, action in METHOD_ACTIONS.items()
                         if action in self.allowed_actions)

    @property
    def required_action(self) -> str | None:
        actions = self.allowed_actions
        return next(iter(actions)) if len(actions) == 1 else None

    def action_hint(self) -> str:
        return (
            f"Allowed next tools: {', '.join(sorted(self.allowed_tools)) or '(none)'}. "
            "Choose exactly one tool. "
            f"Policy turns: {self.turns}/{self.budget.max_turns}; "
            f"searches since curate: {self.searches_since_curate}/"
            f"{self.budget.searches_per_curation}."
        )

    def initialize(self) -> str:
        if self.turns:
            raise RuntimeError("Create a new HarnessSearch for each episode")
        result = self.memory.initialize_direction()
        self.retrieval_required = result.committed
        return result.text

    def reject(self, detail: str) -> OperationResult:
        """Malformed policy responses also consume T_max, preventing retry loops."""
        if self.done:
            raise RuntimeError("The episode has already ended")
        return self._finish_turn("invalid", detail, False)

    def step(self, proposal: ActionProposal) -> OperationResult:
        if self.done:
            raise RuntimeError("The episode has already ended")
        name = canonical_tool_name(proposal.name)
        if name not in self.allowed_tools:
            return self._finish_turn(name, f"Rejected inadmissible action: {name}.", False)
        if not isinstance(proposal.arguments, dict):
            # Copy any Mapping supplied by a caller; never mutate policy arguments.
            try:
                params = dict(proposal.arguments)
            except (TypeError, ValueError):
                return self._finish_turn(name, "Arguments must be an object.", False)
        else:
            params = copy.deepcopy(proposal.arguments)

        if name == "end":
            return self._audit_end()

        result = self.memory.execute(ActionProposal(name, params))
        action = METHOD_ACTIONS[name]
        if result.committed:
            if action == "search":
                self.search_calls += 1
                self.searches_since_curate += 1
            if action in {"search", "read"}:
                self.retrieval_required = False
            elif action == "curate":
                self.curate_calls += 1
                self.searches_since_curate = 0
            elif action == "redirect":
                self.redirect_calls += 1
                self.redirect_required = False
                self.retrieval_required = True
                # Redirect is not curation: it cannot reset S_max.
            self.last_action = action
        return self._finish_turn(name, result.text, result.committed, result.metadata)

    def _audit_end(self) -> OperationResult:
        if not self.audit_enabled:
            self.termination_reason = "audit_disabled"
            return self._finish_turn("end", "Termination audit disabled by configuration.", False)
        verdict = self.auditor.audit(
            self.memory.evidence_snapshot(), self.search_calls, terminal=True,
        )
        self.memory.commit_audit(verdict)
        ready = verdict.get("verdict") == "answer_ready"
        if ready:
            self.termination_reason = "answer_ready"
        else:
            self.audit_rejections += 1
            self.redirect_required = True
            self.retrieval_required = False
            # Algorithm 1 explicitly opens a fresh search stage after rejection.
            self.searches_since_curate = 0
        self.last_action = "end"
        text = (
            f"[SUMMARY AUDITOR: {verdict.get('verdict', 'search_more').upper()}] "
            f"{verdict.get('reason', '')}\nSummary: {verdict.get('summary', '')}\n"
            f"Missing target: {verdict.get('next_intent', '')}"
        )
        return self._finish_turn("end", text, ready, audit=verdict)

    def _finish_turn(self, name, text, committed, metadata=None, audit=None):
        self.turns += 1
        self.memory.advance_turn()
        if not self.done and self.turns >= self.budget.max_turns:
            # Algorithm 1 returns C_t at budget exhaustion without claiming sufficiency.
            self.termination_reason = "max_turns"
        if not self.done:
            text += "\n\n" + self.action_hint()
        return OperationResult(
            action=name, text=text, committed=committed, metadata=metadata,
            episode_done=self.done, termination_reason=self.termination_reason,
            audit=copy.deepcopy(audit),
        )
