"""
Tests for agentforensics/integrations/ — LangChain and AutoGen.

Neither langchain nor pyautogen is required to be installed.
Fake versions of both are injected into sys.modules per-fixture,
and the integration modules are reloaded so their import guards
re-evaluate with the fakes in place.
"""
import importlib
import sys
import types
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentforensics.models import Message, ToolCall, TraceEvent
from agentforensics.store import get_events


# ---------------------------------------------------------------------------
# Shared infra
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AF_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("AF_DISABLE_ML", "true")
    monkeypatch.delenv("AF_ALERT_WEBHOOK", raising=False)
    import agentforensics.classifier as clf
    clf._ml_load_attempted = False
    clf._ml_available = False
    clf._ml_pipeline = None
    yield
    clf._ml_load_attempted = False
    clf._ml_available = False


# ---------------------------------------------------------------------------
# LangChain fake module injection
# ---------------------------------------------------------------------------

class _FakeBaseCallbackHandler:
    """Minimal stand-in for langchain_core.callbacks.base.BaseCallbackHandler."""
    def __init__(self, *args, **kwargs):
        pass


def _make_lc_message(class_name: str, content: str, **attrs):
    """Create a fake LangChain BaseMessage."""
    cls = type(class_name, (), {"content": content, "additional_kwargs": {}, **attrs})
    return cls()


def _make_lc_generation(text: str, tool_calls: list | None = None):
    """Create a fake LangChain ChatGeneration."""
    ai_msg = _make_lc_message("AIMessage", text)
    if tool_calls:
        ai_msg.tool_calls = tool_calls
        ai_msg.additional_kwargs = {"tool_calls": tool_calls}
    else:
        ai_msg.tool_calls = []
        ai_msg.additional_kwargs = {}

    gen = MagicMock()
    gen.text = text
    gen.message = ai_msg
    return gen


def _make_lc_result(text: str, tool_calls: list | None = None):
    """Create a fake LangChain LLMResult."""
    result = MagicMock()
    result.generations = [[_make_lc_generation(text, tool_calls)]]
    return result


@pytest.fixture
def langchain_mod(monkeypatch):
    """
    Inject a fake langchain_core into sys.modules and reload
    langchain_handler so _LANGCHAIN_AVAILABLE becomes True.
    """
    # Build fake module tree
    fake_base_mod = types.ModuleType("langchain_core.callbacks.base")
    fake_base_mod.BaseCallbackHandler = _FakeBaseCallbackHandler

    fake_callbacks_mod = types.ModuleType("langchain_core.callbacks")
    fake_callbacks_mod.base = fake_base_mod

    fake_core_mod = types.ModuleType("langchain_core")
    fake_core_mod.callbacks = fake_callbacks_mod

    monkeypatch.setitem(sys.modules, "langchain_core", fake_core_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks", fake_callbacks_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks.base", fake_base_mod)

    # Force reload so _LANGCHAIN_AVAILABLE re-evaluates to True
    import agentforensics.integrations.langchain_handler as lh
    importlib.reload(lh)
    yield lh

    # Restore: remove fakes and reload back to "not available" state
    for key in ["langchain_core", "langchain_core.callbacks", "langchain_core.callbacks.base"]:
        sys.modules.pop(key, None)
    importlib.reload(lh)


# ---------------------------------------------------------------------------
# LangChain: ImportError when not installed
# ---------------------------------------------------------------------------

class TestLangChainImportGuard:
    def test_raises_import_error_without_langchain(self):
        """ForensicsCallbackHandler must raise ImportError if langchain absent."""
        # Ensure langchain_core is NOT in sys.modules
        for key in list(sys.modules):
            if key.startswith("langchain"):
                sys.modules.pop(key, None)

        import agentforensics.integrations.langchain_handler as lh
        importlib.reload(lh)

        with pytest.raises(ImportError, match="pip install langchain"):
            lh.ForensicsCallbackHandler()

    def test_error_message_is_exact(self):
        for key in list(sys.modules):
            if key.startswith("langchain"):
                sys.modules.pop(key, None)

        import agentforensics.integrations.langchain_handler as lh
        importlib.reload(lh)

        try:
            lh.ForensicsCallbackHandler()
        except ImportError as exc:
            assert "pip install langchain" in str(exc)


# ---------------------------------------------------------------------------
# LangChain: ForensicsCallbackHandler behaviour
# ---------------------------------------------------------------------------

class TestForensicsCallbackHandler:
    def test_session_id_auto_generated(self, langchain_mod):
        h = langchain_mod.ForensicsCallbackHandler()
        uuid.UUID(h.session_id)  # must be a valid UUID

    def test_explicit_session_id_preserved(self, langchain_mod):
        sid = "my-lc-session"
        h = langchain_mod.ForensicsCallbackHandler(session_id=sid)
        assert h.session_id == sid

    def test_on_chat_model_start_then_end_writes_event(self, langchain_mod):
        h = langchain_mod.ForensicsCallbackHandler()
        run_id = uuid.uuid4()

        serialized = {"kwargs": {"model_name": "gpt-4o-mini"}, "id": ["ChatOpenAI"]}
        human = _make_lc_message("HumanMessage", "What is 2+2?")
        sys_msg = _make_lc_message("SystemMessage", "You are helpful.")
        messages = [[sys_msg, human]]

        h.on_chat_model_start(serialized, messages, run_id=run_id)
        response = _make_lc_result("The answer is 4.")
        h.on_llm_end(response, run_id=run_id)

        events = get_events(h.session_id)
        assert len(events) == 1
        evt = events[0]
        assert evt.model == "gpt-4o-mini"
        assert evt.messages_in[0].role == "system"
        assert evt.messages_in[1].role == "user"
        assert evt.messages_in[1].content == "What is 2+2?"
        assert evt.message_out.content == "The answer is 4."

    def test_on_llm_start_then_end_writes_event(self, langchain_mod):
        """Plain LLM path (on_llm_start with prompts)."""
        h = langchain_mod.ForensicsCallbackHandler()
        run_id = uuid.uuid4()

        serialized = {"name": "openai", "kwargs": {}}
        h.on_llm_start(serialized, ["Hello from user"], run_id=run_id)
        h.on_llm_end(_make_lc_result("Hello back!"), run_id=run_id)

        events = get_events(h.session_id)
        assert len(events) == 1
        assert events[0].messages_in[0].role == "user"
        assert events[0].messages_in[0].content == "Hello from user"

    def test_multiple_turns_increment_turn_index(self, langchain_mod):
        h = langchain_mod.ForensicsCallbackHandler()
        serialized = {"kwargs": {"model_name": "gpt-4o"}}

        for i in range(3):
            rid = uuid.uuid4()
            h.on_chat_model_start(serialized, [[_make_lc_message("HumanMessage", f"Turn {i}")]], run_id=rid)
            h.on_llm_end(_make_lc_result(f"Reply {i}"), run_id=rid)

        events = get_events(h.session_id)
        assert [e.turn_index for e in events] == [0, 1, 2]

    def test_tool_calls_extracted_from_generation(self, langchain_mod):
        h = langchain_mod.ForensicsCallbackHandler()
        run_id = uuid.uuid4()

        raw_tool_calls = [
            {"name": "search", "args": {"query": "python"}, "id": "tc-001"}
        ]
        serialized = {"kwargs": {"model_name": "gpt-4o"}}
        h.on_chat_model_start(serialized, [[_make_lc_message("HumanMessage", "Search for python")]], run_id=run_id)
        h.on_llm_end(_make_lc_result("", tool_calls=raw_tool_calls), run_id=run_id)

        events = get_events(h.session_id)
        assert len(events[0].tool_calls) == 1
        assert events[0].tool_calls[0].name == "search"
        assert events[0].tool_calls[0].id == "tc-001"

    def test_openai_wire_format_tool_calls(self, langchain_mod):
        h = langchain_mod.ForensicsCallbackHandler()
        run_id = uuid.uuid4()

        raw_tool_calls = [
            {"id": "call_abc", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "London"}'}}
        ]
        serialized = {"kwargs": {"model_name": "gpt-4o"}}
        h.on_chat_model_start(serialized, [[_make_lc_message("HumanMessage", "Weather?")]], run_id=run_id)
        h.on_llm_end(_make_lc_result("", tool_calls=raw_tool_calls), run_id=run_id)

        events = get_events(h.session_id)
        tc = events[0].tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "London"}

    def test_on_llm_error_cleans_up_and_warns(self, langchain_mod):
        h = langchain_mod.ForensicsCallbackHandler()
        run_id = uuid.uuid4()

        serialized = {"kwargs": {"model_name": "gpt-4o"}}
        h.on_chat_model_start(serialized, [[_make_lc_message("HumanMessage", "Hi")]], run_id=run_id)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h.on_llm_error(RuntimeError("API timeout"), run_id=run_id)

        assert run_id not in h._pending
        assert any("API timeout" in str(warning.message) for warning in w)
        # No event written after error
        assert get_events(h.session_id) == []

    def test_on_tool_error_cleans_up_and_warns(self, langchain_mod):
        h = langchain_mod.ForensicsCallbackHandler()
        run_id = uuid.uuid4()

        h.on_tool_start({"name": "search"}, "query", run_id=run_id)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h.on_tool_error(RuntimeError("Tool failed"), run_id=run_id)

        assert run_id not in h._pending_tools
        assert any("Tool failed" in str(warning.message) for warning in w)

    def test_on_llm_end_without_start_is_noop(self, langchain_mod):
        """on_llm_end called without a matching on_llm_start must not crash."""
        h = langchain_mod.ForensicsCallbackHandler()
        h.on_llm_end(_make_lc_result("orphan"), run_id=uuid.uuid4())
        assert get_events(h.session_id) == []

    def test_hash_chain_valid_after_multiple_turns(self, langchain_mod):
        from agentforensics.store import verify_chain
        h = langchain_mod.ForensicsCallbackHandler()
        serialized = {"kwargs": {"model_name": "gpt-4o"}}

        for i in range(4):
            rid = uuid.uuid4()
            h.on_chat_model_start(serialized, [[_make_lc_message("HumanMessage", f"msg {i}")]], run_id=rid)
            h.on_llm_end(_make_lc_result(f"reply {i}"), run_id=rid)

        assert verify_chain(h.session_id) is True

    def test_model_name_from_invocation_params(self, langchain_mod):
        h = langchain_mod.ForensicsCallbackHandler()
        run_id = uuid.uuid4()

        # invocation_params in kwargs (common in older LangChain versions)
        h.on_llm_start(
            {"name": "openai"},
            ["hello"],
            run_id=run_id,
            invocation_params={"model_name": "gpt-3.5-turbo"},
        )
        h.on_llm_end(_make_lc_result("hi"), run_id=run_id)
        assert get_events(h.session_id)[0].model == "gpt-3.5-turbo"

    def test_on_injection_registration(self, langchain_mod):
        h = langchain_mod.ForensicsCallbackHandler()
        fired = []
        h.on_injection(lambda s, e: fired.append(s.score))
        # Verify delegation reaches the tracer's alert manager
        assert len(h._tracer._alert_manager._handlers) >= 1


# ---------------------------------------------------------------------------
# AutoGen fake module injection
# ---------------------------------------------------------------------------

class _FakeConversableAgent:
    """Minimal stand-in for autogen.ConversableAgent."""
    def __init__(self, name="agent", llm_config=None):
        self.name = name
        self.llm_config = {"model": "gpt-4o-mini"} if llm_config is None else llm_config

    def generate_reply(self, messages=None, sender=None, **kwargs):
        return "I am the original reply."


@pytest.fixture
def autogen_mod(monkeypatch):
    """
    Inject a fake autogen into sys.modules and reload autogen_tracer
    so _AUTOGEN_AVAILABLE becomes True.
    """
    fake_ag_mod = types.ModuleType("autogen")
    fake_ag_mod.ConversableAgent = _FakeConversableAgent

    monkeypatch.setitem(sys.modules, "autogen", fake_ag_mod)

    import agentforensics.integrations.autogen_tracer as at
    importlib.reload(at)
    yield at

    sys.modules.pop("autogen", None)
    importlib.reload(at)


# ---------------------------------------------------------------------------
# AutoGen: ImportError when not installed
# ---------------------------------------------------------------------------

class TestAutoGenImportGuard:
    def test_raises_import_error_without_autogen(self):
        sys.modules.pop("autogen", None)

        import agentforensics.integrations.autogen_tracer as at
        importlib.reload(at)

        agent = _FakeConversableAgent()
        with pytest.raises(ImportError, match="pip install autogen"):
            at.patch_autogen(agent)

    def test_error_message_is_exact(self):
        sys.modules.pop("autogen", None)

        import agentforensics.integrations.autogen_tracer as at
        importlib.reload(at)

        try:
            at.patch_autogen(_FakeConversableAgent())
        except ImportError as exc:
            assert "pip install autogen" in str(exc)


# ---------------------------------------------------------------------------
# AutoGen: patch_autogen behaviour
# ---------------------------------------------------------------------------

class TestPatchAutogen:
    def test_returns_tracer(self, autogen_mod):
        from agentforensics.tracer import Tracer
        agent = _FakeConversableAgent()
        tracer = autogen_mod.patch_autogen(agent)
        assert isinstance(tracer, Tracer)

    def test_session_id_exposed_on_agent(self, autogen_mod):
        agent = _FakeConversableAgent()
        tracer = autogen_mod.patch_autogen(agent)
        assert agent._af_session_id == tracer.session_id
        uuid.UUID(agent._af_session_id)  # valid UUID

    def test_explicit_session_id(self, autogen_mod):
        agent = _FakeConversableAgent()
        tracer = autogen_mod.patch_autogen(agent, session_id="ag-custom-001")
        assert tracer.session_id == "ag-custom-001"
        assert agent._af_session_id == "ag-custom-001"

    def test_generate_reply_returns_unchanged(self, autogen_mod):
        """Zero-behaviour-change: the original reply must be returned intact."""
        agent = _FakeConversableAgent()
        autogen_mod.patch_autogen(agent)

        result = agent.generate_reply(
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert result == "I am the original reply."

    def test_trace_event_written_to_store(self, autogen_mod):
        agent = _FakeConversableAgent(llm_config={"model": "gpt-4o-mini"})
        tracer = autogen_mod.patch_autogen(agent)

        agent.generate_reply(messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ])

        events = get_events(tracer.session_id)
        assert len(events) == 1
        evt = events[0]
        assert evt.turn_index == 0
        assert evt.model == "gpt-4o-mini"
        assert evt.messages_in[0].role == "user"
        assert "France" in evt.messages_in[0].content
        assert evt.message_out.content == "I am the original reply."

    def test_multiple_replies_increment_turn_index(self, autogen_mod):
        agent = _FakeConversableAgent()
        tracer = autogen_mod.patch_autogen(agent)

        for i in range(3):
            agent.generate_reply(messages=[{"role": "user", "content": f"msg {i}"}])

        events = get_events(tracer.session_id)
        assert [e.turn_index for e in events] == [0, 1, 2]

    def test_model_from_config_list(self, autogen_mod):
        agent = _FakeConversableAgent(llm_config={
            "config_list": [{"model": "claude-opus-4-6"}]
        })
        tracer = autogen_mod.patch_autogen(agent)
        agent.generate_reply(messages=[{"role": "user", "content": "hi"}])

        events = get_events(tracer.session_id)
        assert events[0].model == "claude-opus-4-6"

    def test_dict_reply_content_extracted(self, autogen_mod):
        agent = _FakeConversableAgent()
        agent.generate_reply = lambda messages=None, sender=None, **kw: {
            "role": "assistant", "content": "Dict reply content"
        }
        tracer = autogen_mod.patch_autogen(agent)
        agent.generate_reply(messages=[{"role": "user", "content": "hi"}])

        events = get_events(tracer.session_id)
        assert events[0].message_out.content == "Dict reply content"

    def test_none_reply_handled_gracefully(self, autogen_mod):
        agent = _FakeConversableAgent()
        agent.generate_reply = lambda messages=None, sender=None, **kw: None
        tracer = autogen_mod.patch_autogen(agent)
        result = agent.generate_reply(messages=[{"role": "user", "content": "hi"}])

        assert result is None
        events = get_events(tracer.session_id)
        assert events[0].message_out.content == ""

    def test_tool_calls_in_dict_reply(self, autogen_mod):
        agent = _FakeConversableAgent()
        agent.generate_reply = lambda messages=None, sender=None, **kw: {
            "content": None,
            "tool_calls": [{
                "id": "tc-99",
                "type": "function",
                "function": {"name": "send_email", "arguments": '{"to": "a@b.com"}'},
            }],
        }
        tracer = autogen_mod.patch_autogen(agent)
        agent.generate_reply(messages=[{"role": "user", "content": "send it"}])

        events = get_events(tracer.session_id)
        assert len(events[0].tool_calls) == 1
        assert events[0].tool_calls[0].name == "send_email"
        assert events[0].tool_calls[0].arguments == {"to": "a@b.com"}

    def test_hash_chain_valid_after_patched_calls(self, autogen_mod):
        from agentforensics.store import verify_chain
        agent = _FakeConversableAgent()
        tracer = autogen_mod.patch_autogen(agent)

        for i in range(4):
            agent.generate_reply(messages=[{"role": "user", "content": f"turn {i}"}])

        assert verify_chain(tracer.session_id) is True

    def test_other_agents_unaffected(self, autogen_mod):
        """Patching one instance must not affect other instances of the class."""
        agent_a = _FakeConversableAgent(name="a")
        agent_b = _FakeConversableAgent(name="b")

        autogen_mod.patch_autogen(agent_a)

        # agent_b still has the original generate_reply
        result = agent_b.generate_reply(messages=[{"role": "user", "content": "hi"}])
        assert result == "I am the original reply."
        assert not hasattr(agent_b, "_af_session_id")

    def test_type_error_if_no_generate_reply(self, autogen_mod):
        class BrokenAgent:
            pass

        with pytest.raises(TypeError, match="generate_reply"):
            autogen_mod.patch_autogen(BrokenAgent())

    def test_no_llm_config_uses_unknown_model(self, autogen_mod):
        agent = _FakeConversableAgent(llm_config=False)
        tracer = autogen_mod.patch_autogen(agent)
        agent.generate_reply(messages=[{"role": "user", "content": "hi"}])
        assert get_events(tracer.session_id)[0].model == "unknown"

    def test_message_roles_preserved(self, autogen_mod):
        agent = _FakeConversableAgent()
        tracer = autogen_mod.patch_autogen(agent)
        agent.generate_reply(messages=[
            {"role": "system",    "content": "You are helpful."},
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ])

        events = get_events(tracer.session_id)
        roles = [m.role for m in events[0].messages_in]
        assert roles == ["system", "user", "assistant"]
