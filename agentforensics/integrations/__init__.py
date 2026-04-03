"""
Optional framework integrations for agentforensics.

Available integrations

ForensicsCallbackHandler  - LangChain callback handler
    Requires: pip install langchain langchain-openai

patch_autogen             - AutoGen ConversableAgent tracer
    Requires: pip install pyautogen

Neither framework needs to be installed to import this package - the
ImportError is raised only when you try to instantiate / call the integration.
"""

__all__ = ["ForensicsCallbackHandler", "patch_autogen"]


def __getattr__(name: str):  # lazy re-export
    if name == "ForensicsCallbackHandler":
        from .langchain_handler import ForensicsCallbackHandler
        return ForensicsCallbackHandler
    if name == "patch_autogen":
        from .autogen_tracer import patch_autogen
        return patch_autogen
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
