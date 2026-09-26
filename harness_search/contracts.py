"""Provider-independent contracts for the three Harness-Search authorities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class HarnessBudget:
    """T_max, S_max and B in Section 3 and Algorithm 1."""

    max_turns: int = 40
    searches_per_curation: int = 5
    evidence_capacity: int = 60

    def __post_init__(self):
        for name in ("max_turns", "searches_per_curation", "evidence_capacity"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ActionProposal:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitResult:
    text: str
    committed: bool = True
    metadata: Any = None


@dataclass(frozen=True)
class EvidenceDocument:
    doc_id: str
    text: str
    intent: str = ""


@dataclass(frozen=True)
class EvidenceSnapshot:
    """The auditor can see only q, I_t and selected C_t, never the whole pool."""

    query: str
    current_intent: str
    documents: tuple[EvidenceDocument, ...]


@dataclass(frozen=True)
class OperationResult:
    action: str
    text: str
    committed: bool
    metadata: Any = None
    episode_done: bool = False
    termination_reason: str | None = None
    audit: Mapping[str, Any] | None = None
