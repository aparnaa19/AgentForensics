"""
Tests for store.py.

Required by spec:
  - Insert two events, tamper with the first event's hash,
    confirm verify_chain() returns False.
"""
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone

import pytest

# Point trace dir at a temp directory so tests don't pollute ~/.agentforensics
@pytest.fixture(autouse=True)
def isolated_trace_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AF_TRACE_DIR", str(tmp_path))
    yield tmp_path


from agentforensics.models import Message, ToolCall, TraceEvent
from agentforensics.store import (
    append_event,
    compute_event_hash,
    get_events,
    verify_chain,
    _db_path,
)


def _make_event(session_id: str, turn_index: int, prev_hash: str) -> TraceEvent:
    msg_in = Message(role="user", content=f"Hello turn {turn_index}")
    msg_out = Message(role="assistant", content=f"Reply {turn_index}")
    event = TraceEvent(
        event_id=str(uuid.uuid4()),
        session_id=session_id,
        turn_index=turn_index,
        timestamp=datetime.now(timezone.utc),
        messages_in=[msg_in],
        message_out=msg_out,
        tool_calls=[],
        model="gpt-4o",
        external_sources=[],
        prev_hash=prev_hash,
        event_hash="",  # will be replaced
    )
    event = event.model_copy(update={"event_hash": compute_event_hash(event)})
    return event


# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------

def test_append_and_retrieve():
    sid = str(uuid.uuid4())
    e0 = _make_event(sid, 0, "")
    e1 = _make_event(sid, 1, e0.event_hash)

    append_event(e0)
    append_event(e1)

    events = get_events(sid)
    assert len(events) == 2
    assert events[0].turn_index == 0
    assert events[1].turn_index == 1
    assert events[0].session_id == sid


# ---------------------------------------------------------------------------
# Valid chain verifies correctly
# ---------------------------------------------------------------------------

def test_verify_chain_valid():
    sid = str(uuid.uuid4())
    e0 = _make_event(sid, 0, "")
    e1 = _make_event(sid, 1, e0.event_hash)

    append_event(e0)
    append_event(e1)

    assert verify_chain(sid) is True


# ---------------------------------------------------------------------------
# Spec-required: tamper with first event's hash → verify_chain returns False
# ---------------------------------------------------------------------------

def test_verify_chain_detects_tampering():
    sid = str(uuid.uuid4())
    e0 = _make_event(sid, 0, "")
    e1 = _make_event(sid, 1, e0.event_hash)

    append_event(e0)
    append_event(e1)

    # Directly tamper with the stored hash of the first event
    db = str(_db_path(sid))
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE events SET event_hash = ? WHERE turn_index = 0",
        ("deadbeef" * 8,),  # 64-char fake hash
    )
    conn.commit()
    conn.close()

    assert verify_chain(sid) is False


# ---------------------------------------------------------------------------
# Append-only: duplicate event_id must raise
# ---------------------------------------------------------------------------

def test_append_only_no_duplicate():
    sid = str(uuid.uuid4())
    e0 = _make_event(sid, 0, "")
    append_event(e0)

    with pytest.raises(Exception):
        append_event(e0)  # same event_id → PRIMARY KEY conflict


# ---------------------------------------------------------------------------
# Empty session chain is valid
# ---------------------------------------------------------------------------

def test_verify_chain_empty_session():
    # A session that has never had events written is valid (trivially)
    sid = str(uuid.uuid4())
    # We need a DB to exist; create it by connecting
    from agentforensics.store import _get_conn
    conn = _get_conn(sid)
    conn.close()
    assert verify_chain(sid) is True


# ---------------------------------------------------------------------------
# Payload round-trip fidelity
# ---------------------------------------------------------------------------

def test_event_payload_roundtrip():
    sid = str(uuid.uuid4())
    tc = ToolCall(id="tc1", name="search", arguments={"q": "test"}, result="found")
    msg_in = Message(role="user", content="Search for test")
    msg_out = Message(role="assistant", content="Sure", tool_calls=[{"id": "tc1"}])
    e0 = TraceEvent(
        event_id=str(uuid.uuid4()),
        session_id=sid,
        turn_index=0,
        timestamp=datetime.now(timezone.utc),
        messages_in=[msg_in],
        message_out=msg_out,
        tool_calls=[tc],
        model="claude-opus-4-6",
        external_sources=["https://example.com"],
        prev_hash="",
        event_hash="",
    )
    e0 = e0.model_copy(update={"event_hash": compute_event_hash(e0)})
    append_event(e0)

    retrieved = get_events(sid)[0]
    assert retrieved.tool_calls[0].name == "search"
    assert retrieved.external_sources == ["https://example.com"]
    assert retrieved.model == "claude-opus-4-6"
