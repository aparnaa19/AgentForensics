"""
AutoGen integration for agentforensics.

Provides patch_autogen(agent) - monkey-patches a ConversableAgent so every
generate_reply() call is recorded as a tamper-evident TraceEvent.

Requirements

    pip install autogen

Usage

    from autogen import ConversableAgent
    from agentforensics.integrations.autogen_tracer import patch_autogen

    agent = ConversableAgent(name="assistant", llm_config={"model": "gpt-4o-mini"})
    patch_autogen(agent)

    # agent.generate_reply(...) now records to agentforensics automatically.
    print(agent._af_session_id)
"""
import json
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any


# Optional dependency guard


try:
    from autogen import ConversableAgent as _AGBase  # noqa: F401
    _AUTOGEN_AVAILABLE = True
except ImportError:
    _AUTOGEN_AVAILABLE = False

from agentforensics.models import Message, ToolCall
from agentforensics.tracer import Tracer


 
# Message conversion helpers
 

_AG_ROLE_MAP = {
    "user":      "user",
    "assistant": "assistant",
    "system":    "system",
    "tool":      "tool",
    "function":  "tool",
}


def _ag_msg_to_message(raw: dict) -> Message:
    """Convert an AutoGen message dict to agentforensics Message."""
    role = _AG_ROLE_MAP.get(raw.get("role", "user"), "user")
    content = raw.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content)
    return Message(
        role=role,
        content=content,
        tool_call_id=raw.get("tool_call_id"),
        tool_calls=raw.get("tool_calls"),
    )


def _extract_model_from_agent(agent: Any) -> str:
    """Best-effort extraction of model name from an AutoGen agent."""
    llm_config = getattr(agent, "llm_config", None)
    if not llm_config or not isinstance(llm_config, dict):
        return "unknown"
    # config_list takes priority (standard AutoGen pattern)
    config_list = llm_config.get("config_list") or []
    if config_list and isinstance(config_list[0], dict):
        model = config_list[0].get("model", "")
        if model:
            return model
    # Flat llm_config fallback
    return llm_config.get("model", "unknown")


def _extract_tool_calls_from_reply(reply: dict) -> list[ToolCall]:
    """Extract ToolCall objects from an AutoGen reply dict if it contains tool calls."""
    raw_tcs = reply.get("tool_calls") or []
    calls: list[ToolCall] = []
    for tc in raw_tcs:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        name = func.get("name", "unknown")
        raw_args = func.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except (json.JSONDecodeError, TypeError):
            args = {"_raw": raw_args}
        calls.append(ToolCall(
            id=tc.get("id", str(uuid.uuid4())),
            name=name,
            arguments=args,
        ))
    return calls


 
# Public API
 

def patch_autogen(
    agent: Any,
    session_id: str | None = None,
    agent_name: str | None = None,
) -> Tracer:
    """
    Monkey-patch *agent* so every generate_reply() call is traced.

    Parameters
    
    agent       : ConversableAgent (or any object with generate_reply)
    session_id  : str, optional  - reuse an existing session
    agent_name  : str, optional  - label for session metadata

    Returns
    
    Tracer
        The underlying Tracer instance (exposes .session_id and .on_injection).

    Raises
    
    ImportError  if pyautogen is not installed.
    TypeError    if the agent does not have a generate_reply method.
    """
    if not _AUTOGEN_AVAILABLE:
        raise ImportError(
            "Install autogen to use this integration: pip install autogen"
        )
    if not hasattr(agent, "generate_reply"):
        raise TypeError(
            f"patch_autogen expects a ConversableAgent with generate_reply, "
            f"got {type(agent).__name__}"
        )

    label = agent_name or getattr(agent, "name", None)
    tracer = Tracer(session_id=session_id, agent_name=label)
    original_generate_reply = agent.generate_reply

    def _patched_generate_reply(
        messages: list[dict] | None = None,
        sender: Any = None,
        **kwargs: Any,
    ) -> Any:
        msgs_in: list[Message] = [_ag_msg_to_message(m) for m in (messages or [])]
        model = _extract_model_from_agent(agent)

        # Call the real generate_reply - zero behaviour change.
        reply = original_generate_reply(messages=messages, sender=sender, **kwargs)

        # Build message_out from the reply (str | dict | None)
        if isinstance(reply, str):
            message_out = Message(role="assistant", content=reply)
            tool_calls: list[ToolCall] = []
        elif isinstance(reply, dict):
            content = reply.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content)
            message_out = Message(role="assistant", content=content)
            tool_calls = _extract_tool_calls_from_reply(reply)
        elif reply is None:
            message_out = Message(role="assistant", content="")
            tool_calls = []
        else:
            message_out = Message(role="assistant", content=str(reply))
            tool_calls = []

        try:
            tracer.record(
                messages_in=msgs_in,
                message_out=message_out,
                tool_calls=tool_calls,
                model=model,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"[agentforensics] Failed to record AutoGen turn: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

        return reply  # unchanged

    # Replace the instance method (not the class method) so other agents
    # of the same class are unaffected.
    agent.generate_reply = _patched_generate_reply
    agent._af_session_id = tracer.session_id
    agent._af_tracer = tracer

    return tracer
