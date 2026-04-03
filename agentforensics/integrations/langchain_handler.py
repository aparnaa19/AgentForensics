"""
LangChain integration for agentforensics.

Provides ForensicsCallbackHandler - a LangChain BaseCallbackHandler that
records every LLM turn as a tamper-evident TraceEvent.

Requirements

    pip install langchain langchain-openai

Usage

    from agentforensics.integrations.langchain_handler import ForensicsCallbackHandler
    from langchain_openai import ChatOpenAI

    handler = ForensicsCallbackHandler()
    llm = ChatOpenAI(callbacks=[handler])
    llm.invoke("Hello")
    print(handler.session_id)
"""
import json
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any

 
# Optional dependency guard
 

try:
    from langchain_core.callbacks.base import BaseCallbackHandler as _LCBase
    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LCBase = object  # type: ignore[assignment,misc]
    _LANGCHAIN_AVAILABLE = False

from agentforensics.models import Message, ToolCall
from agentforensics.tracer import Tracer


 
# Helpers


def _extract_model_name(serialized: dict, kwargs: dict) -> str:
    """Pull the model name out of LangChain's serialized dict."""
    # Prefer explicit model_name in kwargs (e.g. invocation_params)
    inv = kwargs.get("invocation_params") or {}
    for key in ("model_name", "model"):
        if inv.get(key):
            return inv[key]
    # Fall back to serialized kwargs
    skw = serialized.get("kwargs") or {}
    for key in ("model_name", "model"):
        if skw.get(key):
            return skw[key]
    # Last resort: last segment of the id list (e.g. "ChatOpenAI")
    ids = serialized.get("id") or []
    if isinstance(ids, list) and ids:
        return ids[-1]
    return serialized.get("name", "unknown")


def _lc_msg_to_message(lc_msg: Any) -> Message:
    """Convert a LangChain BaseMessage to agentforensics Message."""
    _ROLE_MAP = {
        "HumanMessage":    "user",
        "AIMessage":       "assistant",
        "SystemMessage":   "system",
        "ToolMessage":     "tool",
        "FunctionMessage": "tool",
        "ChatMessage":     "user",
    }
    class_name = type(lc_msg).__name__
    role = _ROLE_MAP.get(class_name, "user")

    content = lc_msg.content
    if not isinstance(content, str):
        content = json.dumps(content) if isinstance(content, (list, dict)) else str(content)

    tool_call_id: str | None = getattr(lc_msg, "tool_call_id", None)
    return Message(role=role, content=content, tool_call_id=tool_call_id)


def _extract_tool_calls_from_generation(gen: Any) -> list[ToolCall]:
    """
    Extract ToolCall objects from a LangChain ChatGeneration.

    Handles both the modern LangChain tool_calls format (list of dicts with
    "name"/"args"/"id") and the OpenAI legacy format (list of dicts with
    "function"/"id"/"type").
    """
    ai_msg = getattr(gen, "message", None)
    if ai_msg is None:
        return []

    # Modern LangChain (>= 0.1): ai_msg.tool_calls is a list[dict]
    raw: list = getattr(ai_msg, "tool_calls", None) or []

    # Legacy: additional_kwargs["tool_calls"] (OpenAI wire format)
    if not raw:
        ak = getattr(ai_msg, "additional_kwargs", {}) or {}
        raw = ak.get("tool_calls") or []

    calls: list[ToolCall] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        if "function" in tc:
            # OpenAI wire format
            func = tc["function"]
            name = func.get("name", "unknown")
            raw_args = func.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": raw_args}
            calls.append(ToolCall(id=tc.get("id", str(uuid.uuid4())), name=name, arguments=args))
        elif "name" in tc:
            # Modern LangChain format: {"name": ..., "args": {...}, "id": ...}
            args = tc.get("args") or {}
            calls.append(ToolCall(id=tc.get("id", str(uuid.uuid4())), name=tc["name"], arguments=args))

    return calls


# Handler

class ForensicsCallbackHandler(_LCBase):
    """
    LangChain callback handler that records every LLM turn to agentforensics.

    Parameters
    
    session_id  : str, optional
        Reuse an existing session or let one be auto-generated.
    agent_name  : str, optional
        Human-readable label stored in session metadata.
    """

    def __init__(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "Install langchain to use this integration: pip install langchain"
            )
        super().__init__()
        self._tracer = Tracer(session_id=session_id, agent_name=agent_name)
        # run_id (UUID) → {messages_in, timestamp, model}
        self._pending: dict[Any, dict] = {}
        # run_id (UUID) → partial tool info
        self._pending_tools: dict[Any, dict] = {}

    # Public

    @property
    def session_id(self) -> str:
        return self._tracer.session_id

    def on_injection(self, handler: Any) -> None:
        """Proxy to register an alert handler on the underlying Tracer."""
        self._tracer.on_injection(handler)

    # LLM lifecycle

    def on_llm_start(
        self,
        serialized: dict,
        prompts: list[str],
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Plain LLM (completion-style) - prompts are raw strings."""
        messages_in = [Message(role="user", content=p) for p in prompts]
        self._pending[run_id] = {
            "messages_in": messages_in,
            "timestamp": datetime.now(timezone.utc),
            "model": _extract_model_name(serialized, kwargs),
        }

    def on_chat_model_start(
        self,
        serialized: dict,
        messages: list[list[Any]],
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Chat model - messages are lists of BaseMessage objects."""
        batch = messages[0] if messages else []
        messages_in = [_lc_msg_to_message(m) for m in batch]
        self._pending[run_id] = {
            "messages_in": messages_in,
            "timestamp": datetime.now(timezone.utc),
            "model": _extract_model_name(serialized, kwargs),
        }

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """LLM finished - build and store the TraceEvent."""
        pending = self._pending.pop(run_id, None)
        if pending is None:
            return

        gens = response.generations
        if not gens or not gens[0]:
            return
        gen = gens[0][0]

        text = gen.text if hasattr(gen, "text") else str(gen)
        tool_calls = _extract_tool_calls_from_generation(gen)
        message_out = Message(role="assistant", content=text)

        self._tracer.record(
            messages_in=pending["messages_in"],
            message_out=message_out,
            tool_calls=tool_calls,
            model=pending["model"],
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """LLM raised an error - clean up pending state, warn, never crash."""
        self._pending.pop(run_id, None)
        warnings.warn(
            f"[agentforensics] LLM error in session {self._tracer.session_id}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Tool lifecycle

    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Tool is about to execute - store partial ToolCall."""
        self._pending_tools[run_id] = {
            "id": str(run_id) if run_id is not None else str(uuid.uuid4()),
            "name": serialized.get("name", "unknown_tool"),
            "arguments": {"input": input_str},
        }

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Tool finished - discard pending state (result is in next LLM messages)."""
        self._pending_tools.pop(run_id, None)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Tool raised an error - clean up, warn, never crash."""
        self._pending_tools.pop(run_id, None)
        warnings.warn(
            f"[agentforensics] Tool error in session {self._tracer.session_id}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
