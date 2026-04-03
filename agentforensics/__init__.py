"""
AgentForensics
Real-time prompt injection detection and forensics for LLM agents.

Quickstart:

    import agentforensics
    import openai

    client = agentforensics.trace(openai.OpenAI())
    # Use `client` exactly like openai.OpenAI() - all calls are monitored.

Environment variables:

    AF_INJECT_THRESHOLD  Detection threshold 0–1 (default: 0.25)
    AF_DISABLE_ML        Set to "true" to skip the ML classifier
    AF_TRACE_DIR         Directory for trace databases (default: .af_traces)
    AF_ALERT_WEBHOOK     Optional webhook URL for injection alerts
"""

from .tracer import Tracer, trace
from .alerting import AlertManager, console_handler, raise_handler, webhook_handler
from .models import InjectionDetectedError

__all__ = [
    "trace",
    "Tracer",
    "AlertManager",
    "console_handler",
    "webhook_handler",
    "raise_handler",
    "InjectionDetectedError",
]
