"""
Instrumentation layer - wraps OpenAI and Anthropic API clients so every LLM
call is recorded as a tamper-evident TraceEvent without altering API behaviour.
"""
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from .models import Message, ToolCall, TraceEvent
from .store import append_event, compute_event_hash, get_events, update_session_signal_score, store_signal
from .alerting import AlertManager, AlertHandler, _auto_register_webhook


# External-source extraction helpers


_URL_RE = re.compile(r"https?://[^\s<>\"{}|\\^'\[\]]+")
# Unix absolute paths, Windows absolute paths, or ./relative paths
_PATH_RE = re.compile(
    r"(?:^|[\s(\"'])(/[a-zA-Z0-9_./ \-]{3,}|[A-Za-z]:\\[^\s]{2,}|\./[^\s]{2,})"
)


def _extract_external_sources(messages: list[Message]) -> list[str]:
    """Scan user and tool messages for URLs and file paths."""
    seen: dict[str, None] = {}
    for msg in messages:
        if msg.role in ("user", "tool") and msg.content:
            for url in _URL_RE.findall(msg.content):
                seen[url] = None
            for path_match in _PATH_RE.findall(msg.content):
                seen[path_match.strip()] = None
    return list(seen)



# Tracer - core recording class


class Tracer:
    """
    Low-level recording primitive.  Use directly for custom integrations:

        tracer = Tracer(session_id="my-session")
        tracer.record(messages_in=..., message_out=..., tool_calls=..., model=...)
    """

    def __init__(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())
        self.agent_name = agent_name
        self._turn_index: int | None = None  # lazy-initialised from store
        self._prev_hash: str | None = None   # lazy-initialised from store
        self._alert_manager: AlertManager = AlertManager()
        _auto_register_webhook(self._alert_manager)  # honour AF_ALERT_WEBHOOK

 
    # Internal state bootstrap


    def _bootstrap(self) -> None:
        """Load turn_index and prev_hash from the existing store (if any)."""
        events = get_events(self.session_id)
        if events:
            last = events[-1]
            self._turn_index = last.turn_index + 1
            self._prev_hash = last.event_hash
        else:
            self._turn_index = 0
            self._prev_hash = ""

   
    # Public API
   

    def record(
        self,
        messages_in: list[Message],
        message_out: Message,
        tool_calls: list[ToolCall],
        model: str,
        external_sources: list[str] | None = None,
    ) -> TraceEvent:
        """Build a TraceEvent, hash it, persist it, and return it."""
        if self._turn_index is None:
            self._bootstrap()

        if external_sources is None:
            external_sources = _extract_external_sources(messages_in)

        # Build event without final hash so we can compute it.
        event = TraceEvent(
            event_id=str(uuid.uuid4()),
            session_id=self.session_id,
            turn_index=self._turn_index,
            timestamp=datetime.now(timezone.utc),
            messages_in=messages_in,
            message_out=message_out,
            tool_calls=tool_calls,
            model=model,
            external_sources=external_sources,
            prev_hash=self._prev_hash,
            event_hash="",  # placeholder - replaced below
        )
        event = event.model_copy(update={"event_hash": compute_event_hash(event)})

        append_event(event, agent_name=self.agent_name)

        # Advance state before alerting so turn_index is consistent if the
        # raise_handler throws and the caller inspects the store.
        self._prev_hash = event.event_hash
        self._turn_index += 1

        # Real-time injection detection
        # Import lazily to avoid a circular import at module load time.
        from .classifier import classify, classify_window, _threshold
        threshold = _threshold()
        signals   = classify(event)

        # Stage 5: sliding window - only runs when per-turn check found nothing,
        # so it catches multi-turn attacks without creating duplicate alerts.
        above = [s for s in signals if s.score >= threshold]
        if not above:
            signals += classify_window(event)

        for signal in signals:
            if signal.score >= threshold:
                store_signal(self.session_id, signal)
                self._alert_manager.fire(signal, event)
                update_session_signal_score(self.session_id, signal.score)
        # InjectionDetectedError from raise_handler propagates to caller here.

        return event

    def on_injection(self, handler: AlertHandler) -> None:
        """Register a callable to be invoked when an injection is detected.

        The handler receives ''(signal: InjectionSignal, event: TraceEvent)''.
        """
        self._alert_manager.register(handler)



# Message conversion helpers


def _content_to_str(content: Any) -> str:
    """Normalise an API content value (str or list of parts) to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", json.dumps(part)))
            elif hasattr(part, "text"):
                parts.append(part.text)
            else:
                parts.append(str(part))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _dict_to_message(d: dict) -> Message:
    """Convert a raw OpenAI-style message dict to a Message model."""
    role = d.get("role", "user")
    content = _content_to_str(d.get("content", ""))
    return Message(
        role=role,
        content=content,
        tool_call_id=d.get("tool_call_id"),
        tool_calls=d.get("tool_calls"),
    )



# OpenAI proxy

def _parse_openai_response(response: Any) -> tuple[Message, list[ToolCall]]:
    """Extract message_out and tool_calls from an OpenAI chat completion response."""
    choice = response.choices[0]
    msg = choice.message

    content = _content_to_str(msg.content)
    raw_tool_calls = None
    parsed_tool_calls: list[ToolCall] = []

    if msg.tool_calls:
        raw_tool_calls = []
        for tc in msg.tool_calls:
            raw_tool_calls.append({"id": tc.id, "type": "function", "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }})
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {"_raw": tc.function.arguments}
            parsed_tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=arguments,
            ))

    message_out = Message(
        role=msg.role,
        content=content,
        tool_calls=raw_tool_calls,
    )
    return message_out, parsed_tool_calls


class _OpenAICompletionsProxy:
    """Proxy for client.chat.completions - intercepts .create()."""

    def __init__(self, real_completions: Any, tracer: Tracer) -> None:
        object.__setattr__(self, "_real", real_completions)
        object.__setattr__(self, "_tracer", tracer)

    def create(self, **kwargs: Any) -> Any:
        real = object.__getattribute__(self, "_real")
        tracer = object.__getattribute__(self, "_tracer")

        raw_messages: list[dict] = kwargs.get("messages", [])
        model: str = kwargs.get("model", "unknown")

        messages_in = [_dict_to_message(m) for m in raw_messages]

        # Call the real API - zero behaviour change.
        response = real.create(**kwargs)

        message_out, tool_calls = _parse_openai_response(response)

        tracer.record(
            messages_in=messages_in,
            message_out=message_out,
            tool_calls=tool_calls,
            model=model,
        )

        return response  # unchanged

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)


class _OpenAIChatProxy:
    """Proxy for client.chat - exposes .completions as an intercepted proxy."""

    def __init__(self, real_chat: Any, tracer: Tracer) -> None:
        object.__setattr__(self, "_real", real_chat)
        object.__setattr__(self, "_completions", _OpenAICompletionsProxy(
            real_chat.completions, tracer
        ))

    @property
    def completions(self) -> _OpenAICompletionsProxy:
        return object.__getattribute__(self, "_completions")

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)


class _OpenAIProxy:
    """Top-level proxy for an openai.OpenAI client."""

    def __init__(self, real_client: Any, tracer: Tracer) -> None:
        object.__setattr__(self, "_real", real_client)
        object.__setattr__(self, "_tracer", tracer)
        object.__setattr__(self, "_af_session_id", tracer.session_id)
        object.__setattr__(self, "chat", _OpenAIChatProxy(real_client.chat, tracer))

    def on_injection(self, handler: AlertHandler) -> None:
        """Register a handler called when an injection signal is detected."""
        object.__getattribute__(self, "_tracer").on_injection(handler)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)



# Anthropic proxy

def _parse_anthropic_response(response: Any) -> tuple[Message, list[ToolCall]]:
    """Extract message_out and tool_calls from an Anthropic messages response."""
    text_parts: list[str] = []
    raw_tool_calls: list[dict] = []
    parsed_tool_calls: list[ToolCall] = []

    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            raw_tool_calls.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
            parsed_tool_calls.append(ToolCall(
                id=block.id,
                name=block.name,
                arguments=block.input if isinstance(block.input, dict) else {},
            ))

    message_out = Message(
        role=getattr(response, "role", "assistant"),
        content="\n".join(text_parts),
        tool_calls=raw_tool_calls if raw_tool_calls else None,
    )
    return message_out, parsed_tool_calls


class _AnthropicMessagesProxy:
    """Proxy for client.messages - intercepts .create()."""

    def __init__(self, real_messages: Any, tracer: Tracer) -> None:
        object.__setattr__(self, "_real", real_messages)
        object.__setattr__(self, "_tracer", tracer)

    def create(self, **kwargs: Any) -> Any:
        real = object.__getattribute__(self, "_real")
        tracer = object.__getattribute__(self, "_tracer")

        raw_messages: list[dict] = kwargs.get("messages", [])
        model: str = kwargs.get("model", "unknown")
        system: str | None = kwargs.get("system")

        messages_in: list[Message] = []
        if system:
            messages_in.append(Message(role="system", content=system))
        messages_in.extend(_dict_to_message(m) for m in raw_messages)

        # Call the real API - zero behaviour change.
        response = real.create(**kwargs)

        message_out, tool_calls = _parse_anthropic_response(response)

        tracer.record(
            messages_in=messages_in,
            message_out=message_out,
            tool_calls=tool_calls,
            model=model,
        )

        return response  # unchanged

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)


class _AnthropicProxy:
    """Top-level proxy for an anthropic.Anthropic client."""

    def __init__(self, real_client: Any, tracer: Tracer) -> None:
        object.__setattr__(self, "_real", real_client)
        object.__setattr__(self, "_tracer", tracer)
        object.__setattr__(self, "_af_session_id", tracer.session_id)
        object.__setattr__(self, "messages", _AnthropicMessagesProxy(
            real_client.messages, tracer
        ))

    def on_injection(self, handler: AlertHandler) -> None:
        """Register a handler called when an injection signal is detected."""
        object.__getattribute__(self, "_tracer").on_injection(handler)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)



# Public entry point


def trace(
    client: Any,
    session_id: str | None = None,
    agent_name: str | None = None,
) -> Any:
    """
    Wrap an OpenAI or Anthropic client so every API call is recorded.

    Returns a proxy that is API-identical to the original client.
    The session ID is accessible as ''client._af_session_id''.
    """
    tracer = Tracer(session_id=session_id, agent_name=agent_name)
    module = type(client).__module__

    if module.startswith("openai"):
        return _OpenAIProxy(client, tracer)
    if module.startswith("anthropic"):
        return _AnthropicProxy(client, tracer)

    # Fallback: guess from attribute presence
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        return _OpenAIProxy(client, tracer)
    if hasattr(client, "messages") and callable(getattr(client.messages, "create", None)):
        return _AnthropicProxy(client, tracer)

    raise ValueError(
        f"Unsupported client type '{type(client).__name__}'. "
        "Pass an openai.OpenAI or anthropic.Anthropic instance."
    )
