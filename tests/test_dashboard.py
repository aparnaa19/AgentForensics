"""
Tests for the agentforensics web dashboard API.

Requires FastAPI and httpx:
    pip install "agentforensics[dashboard]"
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed — pip install agentforensics[dashboard]")
httpx   = pytest.importorskip("httpx",   reason="httpx not installed — pip install agentforensics[dashboard]")

from fastapi.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(session_id: str, turn_index: int = 0, prev_hash: str = ""):
    from agentforensics.models import Message, TraceEvent
    from agentforensics.store import compute_event_hash

    ev = TraceEvent(
        event_id=str(uuid.uuid4()),
        session_id=session_id,
        turn_index=turn_index,
        messages_in=[Message(role="user", content="hello")],
        message_out=Message(role="assistant", content="hi"),
        tool_calls=[],
        model="gpt-4o-mini",
        external_sources=[],
        prev_hash=prev_hash,
        event_hash="",
        timestamp=datetime.now(timezone.utc),
    )
    return ev.model_copy(update={"event_hash": compute_event_hash(ev)})


def _seed_session(session_id: str | None = None, turns: int = 1) -> str:
    """Write one or more turns to the store and return the session_id."""
    from agentforensics.store import append_event

    sid = session_id or f"test-{uuid.uuid4().hex[:8]}"
    prev = ""
    for i in range(turns):
        ev = _make_event(sid, turn_index=i, prev_hash=prev)
        append_event(ev)
        prev = ev.event_hash
    return sid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_global_hooks():
    """Ensure global hooks left by create_app() don't leak between tests."""
    from agentforensics.alerting import _global_fire_hooks
    from agentforensics.dashboard.server import _sse_push
    yield
    # Remove the hook registered by create_app() so the list stays clean.
    try:
        _global_fire_hooks.remove(_sse_push)
    except ValueError:
        pass


@pytest.fixture()
def trace_dir(tmp_path, monkeypatch):
    """Isolated trace directory."""
    monkeypatch.setenv("AF_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("AF_DISABLE_ML", "true")
    return tmp_path


@pytest.fixture()
def client(trace_dir):
    """TestClient backed by a fresh app instance."""
    from agentforensics.dashboard import server as srv
    srv._sse_clients.clear()
    app = srv.create_app()
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# GET /api/sessions
# ---------------------------------------------------------------------------

class TestGetSessions:
    def test_empty_returns_empty_list(self, client, trace_dir):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_lists_seeded_session(self, client, trace_dir):
        sid = _seed_session()
        data = client.get("/api/sessions").json()
        assert len(data) == 1
        assert data[0]["session_id"] == sid

    def test_response_has_required_fields(self, client, trace_dir):
        _seed_session()
        entry = client.get("/api/sessions").json()[0]
        for field in ("session_id", "started_at", "turn_count", "model", "verdict"):
            assert field in entry, f"missing field: {field}"

    def test_verdict_is_valid_value(self, client, trace_dir):
        _seed_session()
        entry = client.get("/api/sessions").json()[0]
        assert entry["verdict"] in {"clean", "suspicious", "compromised"}

    def test_clean_session_has_verdict_clean(self, client, trace_dir):
        _seed_session()
        entry = client.get("/api/sessions").json()[0]
        assert entry["verdict"] == "clean"

    def test_sorted_most_recent_first(self, client, trace_dir):
        sid_a = _seed_session()
        time.sleep(0.01)          # ensure distinct timestamps
        sid_b = _seed_session()
        data = client.get("/api/sessions").json()
        ids = [d["session_id"] for d in data]
        assert ids[0] == sid_b    # most recent first

    def test_turn_count_reflects_stored_turns(self, client, trace_dir):
        sid = _seed_session(turns=3)
        data = client.get("/api/sessions").json()
        entry = next(d for d in data if d["session_id"] == sid)
        assert entry["turn_count"] == 3

    def test_multiple_sessions_all_listed(self, client, trace_dir):
        ids = {_seed_session() for _ in range(3)}
        data = client.get("/api/sessions").json()
        listed = {d["session_id"] for d in data}
        assert ids == listed


# ---------------------------------------------------------------------------
# GET /api/sessions/{session_id}
# ---------------------------------------------------------------------------

class TestGetSessionDetail:
    def test_returns_200_with_forensic_report(self, client, trace_dir):
        sid = _seed_session()
        resp = client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200

    def test_forensic_report_required_fields(self, client, trace_dir):
        sid = _seed_session()
        data = client.get(f"/api/sessions/{sid}").json()
        for field in (
            "session_id", "verdict", "confidence", "chain_valid",
            "total_turns", "injection_signals", "behavior_diffs", "summary",
        ):
            assert field in data, f"missing ForensicReport field: {field}"

    def test_session_id_matches(self, client, trace_dir):
        sid = _seed_session()
        data = client.get(f"/api/sessions/{sid}").json()
        assert data["session_id"] == sid

    def test_chain_valid_true_for_clean_session(self, client, trace_dir):
        sid = _seed_session()
        data = client.get(f"/api/sessions/{sid}").json()
        assert data["chain_valid"] is True

    def test_total_turns_correct(self, client, trace_dir):
        sid = _seed_session(turns=2)
        data = client.get(f"/api/sessions/{sid}").json()
        assert data["total_turns"] == 2

    def test_404_for_unknown_session(self, client, trace_dir):
        resp = client.get("/api/sessions/no-such-session-xyz")
        assert resp.status_code == 404

    def test_404_response_has_detail(self, client, trace_dir):
        resp = client.get("/api/sessions/no-such-session-xyz")
        assert "detail" in resp.json()


# ---------------------------------------------------------------------------
# GET /api/sessions/{session_id}/report
# ---------------------------------------------------------------------------

class TestGetSessionReport:
    def test_returns_200(self, client, trace_dir):
        sid = _seed_session()
        resp = client.get(f"/api/sessions/{sid}/report")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client, trace_dir):
        sid = _seed_session()
        resp = client.get(f"/api/sessions/{sid}/report")
        assert "text/html" in resp.headers.get("content-type", "")

    def test_html_is_complete_document(self, client, trace_dir):
        sid = _seed_session()
        html = client.get(f"/api/sessions/{sid}/report").text
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "AgentForensics" in html

    def test_404_for_unknown_session(self, client, trace_dir):
        resp = client.get("/api/sessions/no-such-session-xyz/report")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/sessions/{session_id}
# ---------------------------------------------------------------------------

class TestDeleteSession:
    def test_delete_returns_200(self, client, trace_dir):
        sid = _seed_session()
        resp = client.delete(f"/api/sessions/{sid}")
        assert resp.status_code == 200

    def test_delete_response_body(self, client, trace_dir):
        sid = _seed_session()
        resp = client.delete(f"/api/sessions/{sid}")
        assert resp.json() == {"deleted": sid}

    def test_session_no_longer_in_list_after_delete(self, client, trace_dir):
        sid = _seed_session()
        client.delete(f"/api/sessions/{sid}")
        ids = [d["session_id"] for d in client.get("/api/sessions").json()]
        assert sid not in ids

    def test_detail_404_after_delete(self, client, trace_dir):
        sid = _seed_session()
        client.delete(f"/api/sessions/{sid}")
        assert client.get(f"/api/sessions/{sid}").status_code == 404

    def test_delete_404_for_unknown(self, client, trace_dir):
        resp = client.delete("/api/sessions/no-such-session-xyz")
        assert resp.status_code == 404

    def test_other_sessions_unaffected_by_delete(self, client, trace_dir):
        sid_a = _seed_session()
        sid_b = _seed_session()
        client.delete(f"/api/sessions/{sid_a}")
        ids = [d["session_id"] for d in client.get("/api/sessions").json()]
        assert sid_b in ids


# ---------------------------------------------------------------------------
# GET / (dashboard SPA)
# ---------------------------------------------------------------------------

class TestDashboardUI:
    def test_root_returns_200(self, client, trace_dir):
        assert client.get("/").status_code == 200

    def test_root_is_html(self, client, trace_dir):
        resp = client.get("/")
        assert "text/html" in resp.headers.get("content-type", "")

    def test_root_contains_app_name(self, client, trace_dir):
        resp = client.get("/")
        assert "AgentForensics" in resp.text

    def test_root_contains_session_list_element(self, client, trace_dir):
        resp = client.get("/")
        assert "session-list" in resp.text

    def test_root_contains_live_monitor_toggle(self, client, trace_dir):
        resp = client.get("/")
        assert "Live Monitor" in resp.text


# ---------------------------------------------------------------------------
# Import guard — ImportError without fastapi
# ---------------------------------------------------------------------------

class TestImportGuard:
    def test_create_app_raises_without_fastapi(self, monkeypatch):
        import sys
        import importlib

        # Temporarily hide fastapi
        real = sys.modules.pop("fastapi", None)
        import agentforensics.dashboard.server as srv
        orig_flag = srv._FASTAPI_AVAILABLE
        srv._FASTAPI_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="pip install"):
                srv.create_app()
        finally:
            srv._FASTAPI_AVAILABLE = orig_flag
            if real is not None:
                sys.modules["fastapi"] = real


# ---------------------------------------------------------------------------
# SSE global hook wiring
# ---------------------------------------------------------------------------

class TestSSEWiring:
    def test_sse_push_registered_as_global_hook(self, trace_dir):
        from agentforensics.alerting import _global_fire_hooks
        from agentforensics.dashboard.server import _sse_push, create_app
        from agentforensics.alerting import unregister_global_hook

        unregister_global_hook(_sse_push)          # ensure clean slate
        assert _sse_push not in _global_fire_hooks

        create_app()
        assert _sse_push in _global_fire_hooks

    def test_sse_push_idempotent_on_double_create(self, trace_dir):
        from agentforensics.alerting import _global_fire_hooks
        from agentforensics.dashboard.server import _sse_push, create_app
        from agentforensics.alerting import unregister_global_hook

        unregister_global_hook(_sse_push)
        create_app()
        create_app()                                # second call must not double-register
        assert _global_fire_hooks.count(_sse_push) == 1

    def test_sse_push_noop_outside_event_loop(self, trace_dir):
        """_sse_push must not raise when called from a sync context."""
        from agentforensics.dashboard.server import _sse_push
        from agentforensics.models import InjectionSignal, TraceEvent, Message

        sig = InjectionSignal(
            turn_index=0, score=0.9,
            matched_heuristics=["H01"],
            source=None, evidence_snippet="ignore",
        )
        ev = TraceEvent(
            event_id="x", session_id="s", turn_index=0,
            messages_in=[Message(role="user", content="hi")],
            message_out=Message(role="assistant", content="ok"),
            tool_calls=[], model="m", external_sources=[],
            prev_hash="", event_hash="",
            timestamp=datetime.now(timezone.utc),
        )
        _sse_push(sig, ev)    # must not raise

    def test_alert_manager_fire_calls_global_hook(self, trace_dir):
        """Firing an AlertManager invokes _global_fire_hooks."""
        from agentforensics.alerting import AlertManager, _global_fire_hooks
        from agentforensics.models import InjectionSignal, TraceEvent, Message

        fired: list = []

        def capture(signal, event):
            fired.append((signal, event))

        _global_fire_hooks.append(capture)
        try:
            manager = AlertManager()
            sig = InjectionSignal(
                turn_index=0, score=0.9,
                matched_heuristics=["H01"],
                source=None, evidence_snippet="test",
            )
            ev = TraceEvent(
                event_id="y", session_id="s2", turn_index=0,
                messages_in=[Message(role="user", content="x")],
                message_out=Message(role="assistant", content="y"),
                tool_calls=[], model="m", external_sources=[],
                prev_hash="", event_hash="",
                timestamp=datetime.now(timezone.utc),
            )
            manager.fire(sig, ev)
            assert len(fired) == 1
            assert fired[0][0] is sig
        finally:
            try:
                _global_fire_hooks.remove(capture)
            except ValueError:
                pass
