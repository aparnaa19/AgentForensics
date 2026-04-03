"""
Data models for AgentForensics.

All models are Pydantic v2 BaseModel subclasses and are fully
serialisable to/from JSON via model_dump_json() / model_validate_json().
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class InjectionDetectedError(Exception):
    """
    Raised by raise_handler when an injection signal meets the threshold.
    Carries the triggering signal and the session ID so callers can catch
    and handle it gracefully.
    """
    def __init__(self, signal: "InjectionSignal", session_id: str) -> None:
        self.signal = signal
        self.session_id = session_id
        super().__init__(
            f"Injection detected in session {session_id} "
            f"at turn {signal.turn_index} (score {signal.score:.2f}, "
            f"heuristics={signal.matched_heuristics})"
        )


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None  # raw tool_calls from API response


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]
    result: str | None = None  # populated after tool executes


class TraceEvent(BaseModel):
    event_id: str               # uuid4
    session_id: str             # groups all events in one agent run
    turn_index: int             # 0-based turn number in conversation
    timestamp: datetime
    messages_in: list[Message]  # full context window sent to LLM
    message_out: Message        # LLM response
    tool_calls: list[ToolCall]  # tool calls made this turn (may be empty)
    model: str                  # e.g. "gpt-4o", "claude-opus-4-6"
    external_sources: list[str] # URLs/file paths that contributed to context
    prev_hash: str              # SHA-256 of previous event (empty string for first)
    event_hash: str             # SHA-256 of this event's canonical JSON


class InjectionSignal(BaseModel):
    turn_index: int
    score: float                # 0.0–1.0 injection likelihood
    matched_heuristics: list[str]
    source: str | None          # which external_source is suspected
    evidence_snippet: str       # the suspicious text, max 500 chars
    campaign_match: dict | None = None  # populated by AlertManager when a known fingerprint matches


class BehaviorDiff(BaseModel):
    pivot_turn: int             # turn where behavior shifted
    before_actions: list[str]   # tool names called before pivot
    after_actions: list[str]    # tool names called after pivot
    new_actions: list[str]      # actions that only appear after pivot
    anomaly_score: float        # 0.0–1.0


class Fingerprint(BaseModel):
    fingerprint_id: str                  # uuid4
    first_seen: datetime
    last_seen: datetime
    session_ids: list[str]               # all sessions where this pattern appeared
    hit_count: int
    representative_snippet: str          # the first evidence text that created this fingerprint
    vector: list[float]                  # the embedding - stored as JSON array
    campaign_name: str | None = None     # optional human label


class ForensicReport(BaseModel):
    session_id: str
    generated_at: datetime
    total_turns: int
    chain_valid: bool           # did all hashes verify?
    injection_signals: list[InjectionSignal]
    behavior_diffs: list[BehaviorDiff]
    verdict: Literal["clean", "suspicious", "compromised"]
    confidence: float
    summary: str                # 2–3 sentence human-readable summary
    fingerprint_matches: list[dict] = Field(default_factory=list)  # matched attack fingerprints
