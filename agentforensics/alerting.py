"""
Real-time alerting layer for agentforensics.

AlertManager collects handler callables and fires them synchronously whenever
the tracer detects an injection signal that meets the threshold.

Built-in handlers
console_handler   - prints a red Rich warning to the terminal
webhook_handler   - factory that returns a handler POSTing JSON to a URL
raise_handler     - raises InjectionDetectedError to hard-stop the agent

Usage
    from agentforensics import trace
    from agentforensics.alerting import console_handler, raise_handler

    client = trace(openai.OpenAI())
    client.on_injection(console_handler)
    client.on_injection(raise_handler)          # hard-stop on detection
"""
import json
import os
import urllib.request
import urllib.error
from typing import Callable

from .models import InjectionDetectedError, InjectionSignal, TraceEvent


# Type alias for handler callables
AlertHandler = Callable[["InjectionSignal", "TraceEvent"], None]


# Process-wide global hooks (e.g. SSE broadcast registered by the dashboard)

_global_fire_hooks: list[AlertHandler] = []


def register_global_hook(handler: AlertHandler) -> None:
    """Register a handler called by *every* AlertManager.fire() in this process.

    Idempotent - registering the same callable twice is a no-op.
    """
    if handler not in _global_fire_hooks:
        _global_fire_hooks.append(handler)


def unregister_global_hook(handler: AlertHandler) -> None:
    """Remove a handler previously registered with register_global_hook()."""
    try:
        _global_fire_hooks.remove(handler)
    except ValueError:
        pass


# AlertManager

class AlertManager:
    """Holds a list of alert handlers and fires them on detected injections."""

    def __init__(self) -> None:
        self._handlers: list[AlertHandler] = []

    def register(self, handler: AlertHandler) -> None:
        """Add a handler to the notification list."""
        self._handlers.append(handler)

    def fire(self, signal: InjectionSignal, event: TraceEvent) -> None:
        """
        Call every registered handler in registration order, then call any
        process-wide global hooks (e.g. the dashboard SSE broadcaster).

        Fingerprinting runs first so that all handlers receive an enriched
        signal with campaign_match populated when a known attack pattern is
        recognised.  InjectionDetectedError is always re-raised.
        """
        # Fingerprint (before handlers so campaign_match is in payload) 
        enriched = signal
        if os.environ.get("AF_DISABLE_FINGERPRINTING", "").lower() not in ("1", "true"):
            try:
                from .fingerprinter import FingerprintStore
                fp, matched = FingerprintStore().add_or_match(signal, event.session_id)
                if matched:
                    enriched = signal.model_copy(update={
                        "campaign_match": {
                            "fingerprint_id": fp.fingerprint_id,
                            "hit_count":      fp.hit_count,
                            "campaign_name":  fp.campaign_name,
                        }
                    })
            except Exception as exc:  # noqa: BLE001
                import warnings
                warnings.warn(
                    f"[agentforensics] Fingerprinting failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        for handler in self._handlers:
            try:
                handler(enriched, event)
            except InjectionDetectedError:
                # re-raise hard-stop exceptions - do not swallow them
                raise
            except Exception as exc:  # noqa: BLE001
                # Broken handler must not crash the agent
                import warnings
                warnings.warn(
                    f"Alert handler {handler!r} raised an unexpected error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        for hook in list(_global_fire_hooks):
            try:
                hook(enriched, event)
            except InjectionDetectedError:
                raise
            except Exception as exc:  # noqa: BLE001
                import warnings
                warnings.warn(
                    f"Global alert hook {hook!r} raised an unexpected error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )


# Built-in handlers


def console_handler(signal: InjectionSignal, event: TraceEvent) -> None:
    """Print a red warning to the terminal using Rich."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        _con = Console(stderr=True)
        body = (
            f"[bold]Session:[/bold]  {event.session_id}\n"
            f"[bold]Turn:[/bold]     {signal.turn_index}\n"
            f"[bold]Score:[/bold]    {signal.score:.3f}\n"
            f"[bold]Rules:[/bold]    {', '.join(signal.matched_heuristics) or 'ML only'}\n"
            f"[bold]Source:[/bold]   {signal.source or 'unknown'}\n"
            f"[bold]Snippet:[/bold]  {signal.evidence_snippet[:120]!r}"
        )
        _con.print(Panel(body, title="[red bold][!!] INJECTION DETECTED[/red bold]",
                         border_style="red"))
    except Exception:  # noqa: BLE001  - never crash the agent for a print
        print(
            f"\n[INJECTION DETECTED] session={event.session_id} "
            f"turn={signal.turn_index} score={signal.score:.3f}\n"
        )


def webhook_handler(url: str) -> AlertHandler:
    """
    Factory - returns a handler that HTTP POSTs the signal as JSON to ''url''.

    The POST body is a JSON object with all InjectionSignal fields plus
    ''session_id'', ''model'', ''event_id'', and ''timestamp'' from the event.
    """
    def _handler(signal: InjectionSignal, event: TraceEvent) -> None:
        payload = {
            **signal.model_dump(),
            "session_id":  event.session_id,
            "event_id":    event.event_id,
            "model":       event.model,
            "timestamp":   event.timestamp.isoformat(),
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except urllib.error.URLError as exc:
            import warnings
            warnings.warn(
                f"webhook_handler: POST to {url!r} failed — {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return _handler


def raise_handler(signal: InjectionSignal, event: TraceEvent) -> None:
    """
    Raise InjectionDetectedError to immediately halt agent execution.
    Catch this in your agent loop to handle the incident gracefully.
    """
    raise InjectionDetectedError(signal=signal, session_id=event.session_id)


# Auto-register AF_ALERT_WEBHOOK if set


def _auto_register_webhook(manager: AlertManager) -> None:
    """Register a webhook handler from the AF_ALERT_WEBHOOK env var if set."""
    url = os.environ.get("AF_ALERT_WEBHOOK", "").strip()
    if url:
        manager.register(webhook_handler(url))
