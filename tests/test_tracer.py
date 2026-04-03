"""
Tests for tracer.py.

Spec requirements:
  - Wrapping a real OpenAI client call (mocked) writes a TraceEvent to the
    store with the correct session_id.
  - The wrapper must never alter the API response returned to the caller.
"""
import json
import types
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from agentforensics.models import Message, ToolCall
from agentforensics.store import get_events
from agentforensics.tracer import Tracer, trace, _extract_external_sources


# ---------------------------------------------------------------------------
# Isolate trace directory
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_trace_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AF_TRACE_DIR", str(tmp_path))
    yield tmp_path


# ---------------------------------------------------------------------------
# Helpers — fake API response objects
# ---------------------------------------------------------------------------

def _make_openai_response(content="Hello!", model="gpt-4o", tool_calls=None):
    """Build a minimal object that looks like an openai.ChatCompletion response."""
    msg = MagicMock()
    msg.role = "assistant"
    msg.content = content
    msg.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = msg

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    # Make it behave like the real object (not a Mock subclass) for identity check
    response.__class__.__name__ = "ChatCompletion"
    return response


def _make_openai_client(response=None):
    """Return a fake openai.OpenAI client whose module looks like 'openai'."""
    if response is None:
        response = _make_openai_response()

    # Create a fake module path so trace() routes to OpenAI proxy
    FakeClient = type("OpenAI", (), {"__module__": "openai"})
    client = FakeClient()

    # Wire up client.chat.completions.create
    completions = MagicMock()
    completions.create = MagicMock(return_value=response)
    chat = MagicMock()
    chat.completions = completions
    client.chat = chat

    return client, response


def _make_anthropic_response(text="Bonjour!", model="claude-opus-4-6"):
    """Build a minimal object that looks like an anthropic.Message response."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    response = MagicMock()
    response.content = [block]
    response.role = "assistant"
    response.model = model
    response.__class__.__name__ = "Message"
    return response


def _make_anthropic_client(response=None):
    """Return a fake anthropic.Anthropic client."""
    if response is None:
        response = _make_anthropic_response()

    FakeClient = type("Anthropic", (), {"__module__": "anthropic"})
    client = FakeClient()

    messages_api = MagicMock()
    messages_api.create = MagicMock(return_value=response)
    client.messages = messages_api

    return client, response


# ---------------------------------------------------------------------------
# Spec-required: TraceEvent written with correct session_id
# ---------------------------------------------------------------------------

def test_openai_trace_writes_event_with_session_id():
    client, _ = _make_openai_client()
    wrapped = trace(client)

    session_id = wrapped._af_session_id
    assert session_id  # non-empty

    wrapped.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi there"}],
    )

    events = get_events(session_id)
    assert len(events) == 1
    assert events[0].session_id == session_id


# ---------------------------------------------------------------------------
# Spec-required: wrapper must NOT alter the API response
# ---------------------------------------------------------------------------

def test_openai_response_unchanged():
    fake_response = _make_openai_response("Exact response text")
    client, _ = _make_openai_client(response=fake_response)
    wrapped = trace(client)

    result = wrapped.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result is fake_response  # identity check — must be the exact same object


def test_anthropic_response_unchanged():
    fake_response = _make_anthropic_response("Exact Anthropic text")
    client, _ = _make_anthropic_client(response=fake_response)
    wrapped = trace(client)

    result = wrapped.messages.create(
        model="claude-opus-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "Bonjour"}],
    )

    assert result is fake_response


# ---------------------------------------------------------------------------
# session_id exposed as _af_session_id
# ---------------------------------------------------------------------------

def test_explicit_session_id_preserved():
    sid = "my-custom-session-42"
    client, _ = _make_openai_client()
    wrapped = trace(client, session_id=sid)
    assert wrapped._af_session_id == sid


def test_auto_session_id_is_uuid():
    client, _ = _make_openai_client()
    wrapped = trace(client)
    # Must be parseable as a UUID
    uuid.UUID(wrapped._af_session_id)


# ---------------------------------------------------------------------------
# TraceEvent content is correct
# ---------------------------------------------------------------------------

def test_event_captures_model_and_messages():
    client, _ = _make_openai_client(_make_openai_response(model="gpt-4o"))
    wrapped = trace(client)

    wrapped.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ],
    )

    event = get_events(wrapped._af_session_id)[0]
    assert event.model == "gpt-4o"
    assert event.turn_index == 0
    assert len(event.messages_in) == 2
    assert event.messages_in[0].role == "system"
    assert event.messages_in[1].role == "user"
    assert event.message_out.role == "assistant"
    assert event.message_out.content == "Hello!"


# ---------------------------------------------------------------------------
# Multiple turns increment turn_index
# ---------------------------------------------------------------------------

def test_multiple_turns_increment_index():
    client, _ = _make_openai_client()
    wrapped = trace(client)

    for i in range(3):
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Turn {i}"}],
        )

    events = get_events(wrapped._af_session_id)
    assert [e.turn_index for e in events] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Hash chain integrity after tracing
# ---------------------------------------------------------------------------

def test_traced_events_form_valid_chain():
    from agentforensics.store import verify_chain

    client, _ = _make_openai_client()
    wrapped = trace(client)

    for _ in range(3):
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "ping"}],
        )

    assert verify_chain(wrapped._af_session_id) is True


# ---------------------------------------------------------------------------
# External source extraction
# ---------------------------------------------------------------------------

def test_external_sources_extracted_from_urls():
    client, _ = _make_openai_client()
    wrapped = trace(client)

    wrapped.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Check https://example.com/data and summarise."},
        ],
    )

    event = get_events(wrapped._af_session_id)[0]
    assert "https://example.com/data" in event.external_sources


# ---------------------------------------------------------------------------
# Anthropic proxy: system prompt becomes a system Message
# ---------------------------------------------------------------------------

def test_anthropic_system_becomes_message():
    client, _ = _make_anthropic_client()
    wrapped = trace(client)

    wrapped.messages.create(
        model="claude-opus-4-6",
        max_tokens=100,
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Hello"}],
    )

    event = get_events(wrapped._af_session_id)[0]
    assert event.messages_in[0].role == "system"
    assert event.messages_in[0].content == "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Unsupported client raises
# ---------------------------------------------------------------------------

def test_unsupported_client_raises():
    class WeirdClient:
        pass

    with pytest.raises(ValueError, match="Unsupported client type"):
        trace(WeirdClient())


# ---------------------------------------------------------------------------
# Tracer direct API
# ---------------------------------------------------------------------------

def test_tracer_direct_record():
    sid = str(uuid.uuid4())
    tracer = Tracer(session_id=sid)

    msg_in = Message(role="user", content="Direct call")
    msg_out = Message(role="assistant", content="Direct reply")

    event = tracer.record(
        messages_in=[msg_in],
        message_out=msg_out,
        tool_calls=[],
        model="gpt-4o",
    )

    assert event.session_id == sid
    assert event.turn_index == 0

    events = get_events(sid)
    assert len(events) == 1
    assert events[0].event_id == event.event_id


# ---------------------------------------------------------------------------
# _extract_external_sources unit tests
# ---------------------------------------------------------------------------

def test_extract_urls():
    msgs = [Message(role="user", content="See https://api.example.com/v1/data for details")]
    sources = _extract_external_sources(msgs)
    assert "https://api.example.com/v1/data" in sources


def test_extract_ignores_assistant_messages():
    msgs = [Message(role="assistant", content="Go to https://evil.com")]
    sources = _extract_external_sources(msgs)
    assert sources == []


def test_extract_deduplicates():
    url = "https://example.com"
    msgs = [
        Message(role="user", content=f"First {url}"),
        Message(role="tool", content=f"Second {url}", tool_call_id="t1"),
    ]
    sources = _extract_external_sources(msgs)
    assert sources.count(url) == 1
