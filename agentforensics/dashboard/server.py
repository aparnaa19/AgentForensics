"""
FastAPI dashboard server for agentforensics.

Routes
GET  /                               single-page dashboard HTML
GET  /api/sessions                   list sessions (most-recent first) + pre-computed verdict
GET  /api/sessions/{id}              full ForensicReport as JSON
GET  /api/sessions/{id}/report       self-contained HTML report
DELETE /api/sessions/{id}            delete a session
GET  /api/live                       Server-Sent Events; pushes injection alerts in real time

The SSE endpoint is fed by registering _sse_push as a process-wide global hook
in agentforensics.alerting, so *any* AlertManager.fire() call in the current
process automatically notifies all connected browser clients.

Requirements
    pip install agentforensics[dashboard]
    # i.e.: fastapi>=0.100, uvicorn>=0.23
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


# Optional dependency guard


try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, StreamingResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from ..store import list_sessions as _store_list_sessions
from ..store import get_events, get_signals, delete_session
from ..reporter import generate as _reporter_generate
from ..reporter import _determine_verdict
from ..classifier import classify
from ..differ import diff as _behavior_diff
from ..alerting import register_global_hook, unregister_global_hook


# Module-level SSE client registry
# One asyncio.Queue per connected /api/live client.


_sse_clients: list[asyncio.Queue] = []

_UI_PATH = Path(__file__).parent / "ui.html"



# SSE broadcast handler (registered as a global alert hook)


def _sse_push(signal: Any, event: Any) -> None:
    """Sync alert handler  serialises the signal and enqueues it for every
    connected SSE client.  Safe to call from any thread; uses
    call_soon_threadsafe so the asyncio queue is touched only from its loop."""
    # Invalidate cached verdict so the next load reflects the new signal.
    _invalidate_verdict_cache(event.session_id)
    payload = json.dumps({
        "session_id":        event.session_id,
        "turn_index":        signal.turn_index,
        "score":             signal.score,
        "matched_heuristics": list(signal.matched_heuristics),
        "source":            signal.source,
        "evidence_snippet":  signal.evidence_snippet,
    })
    try:
        loop = asyncio.get_running_loop()
        for q in list(_sse_clients):
            loop.call_soon_threadsafe(q.put_nowait, payload)
    except RuntimeError:
        # No running event loop in this thread (e.g. sync test context).
        # Silently drop  SSE clients will miss this event, which is acceptable.
        pass



# Report cache
# Keyed by (session_id, output_format) → (last_event_hash, rendered_string).
# Invalidated when a new SSE alert fires for that session.
# Prevents re-running expensive ML inference on every report click.


_verdict_cache: dict[str, tuple[str, str]] = {}
_report_cache:  dict[tuple[str, str], tuple[str, str]] = {}


def _invalidate_verdict_cache(session_id: str) -> None:
    _verdict_cache.pop(session_id, None)
    _report_cache.pop((session_id, "json"), None)
    _report_cache.pop((session_id, "html"), None)



# Session augmentation helper


def _build_session_entry(s: dict) -> dict:
    """Add verdict and top injection source to a store session summary dict.

    Reads the stored max_signal_score and signals written by the Tracer at
    detection time  no ML inference needed on page load.
    """
    sid = s["session_id"]
    score = s.get("max_signal_score") or 0.0
    if score >= 0.75:
        verdict = "compromised"
    elif score >= 0.25:
        verdict = "suspicious"
    else:
        verdict = "clean"

    # Pull top source from stored signals  instant, no ML.
    top_source: str | None = None
    try:
        signals = get_signals(sid)
        if signals:
            top = max(signals, key=lambda sig: sig.score)
            top_source = top.source
    except Exception:
        pass

    return {**s, "verdict": verdict, "top_source": top_source}



# App factory


def create_app() -> "FastAPI":
    """Build and return the FastAPI application.

    Also registers _sse_push as a process-wide alert hook so that injection
    signals fired by *any* Tracer in the process reach connected SSE clients.

    Raises
    ------
    ImportError  if fastapi is not installed.
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "Install dashboard dependencies: pip install 'agentforensics[dashboard]'"
        )

    register_global_hook(_sse_push)

    app = FastAPI(title="AgentForensics Dashboard", version="0.1.0")

    # GET /

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_UI_PATH.read_text(encoding="utf-8"))

    # GET /api/sessions 

    @app.get("/api/sessions")
    async def api_sessions() -> list:
        sessions = _store_list_sessions()
        sessions.sort(key=lambda s: s.get("started_at", ""), reverse=True)
        return [_build_session_entry(s) for s in sessions]

    # GET /api/sessions/{session_id} 

    @app.get("/api/sessions/{session_id}")
    async def api_session_detail(session_id: str) -> dict:
        events = get_events(session_id)
        if not events:
            raise HTTPException(
                status_code=404, detail=f"Session '{session_id}' not found"
            )
        last_hash = events[-1].event_hash
        cache_key = (session_id, "json")
        cached = _report_cache.get(cache_key)
        if cached and cached[0] == last_hash:
            return json.loads(cached[1])
        result = _reporter_generate(session_id, output="json")
        _report_cache[cache_key] = (last_hash, result)
        return json.loads(result)

    # GET /api/sessions/{session_id}/report 

    @app.get("/api/sessions/{session_id}/report", response_class=HTMLResponse)
    async def api_session_report(
        session_id: str,
        autoprint: bool = Query(default=False),
    ) -> HTMLResponse:
        events = get_events(session_id)
        if not events:
            raise HTTPException(
                status_code=404, detail=f"Session '{session_id}' not found"
            )
        last_hash = events[-1].event_hash
        cache_key = (session_id, "html")
        cached = _report_cache.get(cache_key)
        if cached and cached[0] == last_hash:
            html = cached[1]
        else:
            html = _reporter_generate(session_id, output="html")
            _report_cache[cache_key] = (last_hash, html)
        if autoprint:
            html = html.replace(
                "</body>",
                "<script>window.onload=function(){window.print();}</script></body>",
                1,
            )
        return HTMLResponse(html)

    # DELETE /api/sessions/{session_id} 

    @app.delete("/api/sessions/{session_id}")
    async def api_delete_session(session_id: str) -> dict:
        if not get_events(session_id):
            raise HTTPException(
                status_code=404, detail=f"Session '{session_id}' not found"
            )
        delete_session(session_id)
        return {"deleted": session_id}

    # GET /api/fingerprints 

    @app.get("/api/fingerprints")
    async def api_fingerprints() -> list:
        from ..fingerprinter import FingerprintStore
        fps = FingerprintStore().list_fingerprints()
        return [fp.model_dump(mode="json") for fp in fps]

    # GET /api/fingerprints/campaigns 

    @app.get("/api/fingerprints/campaigns")
    async def api_fingerprint_campaigns() -> list:
        from ..fingerprinter import FingerprintStore
        return FingerprintStore().get_campaigns()

    # POST /api/fingerprints/{fingerprint_id}/label

    @app.post("/api/fingerprints/{fingerprint_id}/label")
    async def api_label_fingerprint(fingerprint_id: str, body: dict) -> dict:
        from ..fingerprinter import FingerprintStore
        store = FingerprintStore()
        fps = store.list_fingerprints()
        match = next((f for f in fps if f.fingerprint_id == fingerprint_id), None)
        if match is None:
            raise HTTPException(
                status_code=404,
                detail=f"Fingerprint '{fingerprint_id}' not found",
            )
        name = body.get("name", "") or ""
        store.label_campaign(fingerprint_id, name)
        return {"ok": True, "fingerprint_id": fingerprint_id, "campaign_name": name}

    # POST /api/sessions/{session_id}/false-alarm 

    @app.post("/api/sessions/{session_id}/false-alarm")
    async def api_false_alarm(session_id: str) -> dict:
        """Mark all fingerprints associated with this session as false alarms.

        Future signals whose content closely matches these fingerprints will
        only be flagged if they clear the high-confidence threshold (0.70),
        preventing repeated false positives from the same safe source.
        """
        if not get_events(session_id):
            raise HTTPException(
                status_code=404, detail=f"Session '{session_id}' not found"
            )
        from ..fingerprinter import FingerprintStore
        store = FingerprintStore()
        matches = store.get_matches_for_session(session_id)
        marked = []
        for m in matches:
            store.mark_false_alarm(m["fingerprint_id"])
            marked.append(m["fingerprint_id"])
        return {"ok": True, "session_id": session_id, "marked_fingerprints": marked}

    # GET /api/false-alarms 

    @app.get("/api/false-alarms")
    async def api_false_alarms() -> list:
        """Return all fingerprints marked as false alarms."""
        from ..fingerprinter import FingerprintStore
        store = FingerprintStore()
        rows = store.get_approved()
        return rows

    # DELETE /api/false-alarms/{fingerprint_id} 

    @app.delete("/api/false-alarms/{fingerprint_id}")
    async def api_delete_false_alarm(fingerprint_id: str) -> dict:
        """Remove a false alarm marking so the content is scanned again."""
        from ..fingerprinter import FingerprintStore
        store = FingerprintStore()
        store.unmark_false_alarm(fingerprint_id)
        return {"ok": True, "fingerprint_id": fingerprint_id}

    # POST /api/fingerprints/deduplicate 
    # Merges fingerprints whose representative_snippet vectors are above the
    # similarity threshold into a single record (highest hit_count survives).

    @app.post("/api/fingerprints/deduplicate")
    async def api_deduplicate_fingerprints() -> dict:
        from ..fingerprinter import FingerprintStore, _embed, _cosine
        import json as _json
        store = FingerprintStore()
        fps   = store.list_fingerprints()
        if len(fps) < 2:
            return {"ok": True, "merged": 0, "remaining": len(fps)}

        # Build similarity groups greedily (similar to union-find)
        parent = {fp.fingerprint_id: fp.fingerprint_id for fp in fps}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        fp_list = list(fps)
        for i in range(len(fp_list)):
            for j in range(i + 1, len(fp_list)):
                a, b = fp_list[i], fp_list[j]
                if _cosine(a.vector, b.vector) >= store._threshold:
                    ra, rb = find(a.fingerprint_id), find(b.fingerprint_id)
                    if ra != rb:
                        # Keep the one with more hits as root
                        a_hits = next(f.hit_count for f in fps if f.fingerprint_id == ra)
                        b_hits = next(f.hit_count for f in fps if f.fingerprint_id == rb)
                        if b_hits > a_hits:
                            parent[ra] = rb
                        else:
                            parent[rb] = ra

        # Group by root
        groups: dict[str, list] = {}
        for fp in fp_list:
            root = find(fp.fingerprint_id)
            groups.setdefault(root, []).append(fp)

        merged_count = 0
        conn = store._get_conn()
        try:
            for root_id, members in groups.items():
                if len(members) == 1:
                    continue
                # Merge all into root: sum hits, union session_ids, keep earliest first_seen / latest last_seen
                root_fp = next(f for f in members if f.fingerprint_id == root_id)
                all_sessions: list[str] = list(root_fp.session_ids)
                total_hits = root_fp.hit_count
                first_seen = root_fp.first_seen
                last_seen  = root_fp.last_seen
                for fp in members:
                    if fp.fingerprint_id == root_id:
                        continue
                    total_hits += fp.hit_count
                    for s in fp.session_ids:
                        if s not in all_sessions:
                            all_sessions.append(s)
                    if fp.first_seen < first_seen:
                        first_seen = fp.first_seen
                    if fp.last_seen > last_seen:
                        last_seen = fp.last_seen
                    conn.execute("DELETE FROM fingerprints WHERE fingerprint_id = ?", (fp.fingerprint_id,))
                    merged_count += 1
                conn.execute(
                    """UPDATE fingerprints
                       SET hit_count   = ?,
                           session_ids = ?,
                           first_seen  = ?,
                           last_seen   = ?
                       WHERE fingerprint_id = ?""",
                    (total_hits, _json.dumps(all_sessions),
                     first_seen.isoformat(), last_seen.isoformat(), root_id),
                )
            conn.commit()
        finally:
            conn.close()

        remaining = len(fps) - merged_count
        return {"ok": True, "merged": merged_count, "remaining": remaining}

    # POST /api/scan
    # Lightweight endpoint for the VS Code extension to scan arbitrary text
    # without creating a full session.  Runs Stage 1 (heuristics) + Stage 3
    # (instruction boundary) only  no ML, responds in milliseconds.

    @app.post("/api/scan")
    async def api_scan(body: dict) -> dict:
        import uuid
        from datetime import timezone
        from datetime import datetime as dt
        text     = body.get("text", "")
        source   = body.get("source", "vscode-editor")   # file path or label
        if not text or not text.strip():
            return {"score": 0.0, "verdict": "clean", "matched_heuristics": [], "evidence_snippet": ""}
        from ..classifier import _run_heuristics
        from ..semantic import instruction_boundary_score
        h_score, h_rules, h_evidence = _run_heuristics(text)
        ib_score, ib_rules, ib_evidence = instruction_boundary_score(text)
        all_rules = h_rules + ib_rules
        final_score = min(1.0, h_score + ib_score)
        if final_score >= 0.75:
            verdict = "compromised"
        elif final_score >= 0.25:
            verdict = "suspicious"
        else:
            verdict = "clean"

        result = {
            "score":              final_score,
            "verdict":            verdict,
            "matched_heuristics": all_rules,
            "evidence_snippet":   (h_evidence or ib_evidence or "")[:500],
        }

        # Log suspicious/compromised scans as a session so they appear in the dashboard
        if verdict != "clean":
            from ..models import TraceEvent, Message, InjectionSignal
            from ..store import append_event, store_signal, compute_event_hash, update_session_signal_score
            session_id = str(uuid.uuid4())
            now        = dt.now(timezone.utc)
            dummy_msg  = Message(role="user", content=f"[VS Code file scan] {source}")
            event = TraceEvent(
                event_id         = str(uuid.uuid4()),
                session_id       = session_id,
                turn_index       = 0,
                timestamp        = now,
                messages_in      = [dummy_msg],
                message_out      = Message(role="assistant", content="[scan result]"),
                tool_calls       = [],
                model            = "vscode-scan",
                external_sources = [source],
                prev_hash        = "",
                event_hash       = "",
            )
            event.event_hash = compute_event_hash(event)
            append_event(event, agent_name="vscode", notes=f"File scan: {source}")

            signal = InjectionSignal(
                turn_index       = 0,
                score            = final_score,
                matched_heuristics = all_rules,
                source           = source,
                evidence_snippet = (h_evidence or ib_evidence or "")[:500],
            )
            store_signal(session_id, signal)
            update_session_signal_score(session_id, final_score)

            # Create / update fingerprint so it appears in Campaigns
            try:
                from ..fingerprinter import FingerprintStore
                FingerprintStore().add_or_match(signal, session_id)
            except Exception:
                pass

            # Push SSE alert to dashboard live view
            _sse_push(signal, event)

            result["session_id"] = session_id

        return result

    # GET /api/live (SSE) 

    @app.get("/api/live")
    async def api_live() -> StreamingResponse:
        queue: asyncio.Queue = asyncio.Queue()
        _sse_clients.append(queue)

        async def event_stream():
            try:
                while True:
                    try:
                        data = await asyncio.wait_for(queue.get(), timeout=30.0)
                        yield f"data: {data}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                try:
                    _sse_clients.remove(queue)
                except ValueError:
                    pass

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":    "no-cache",
                "X-Accel-Buffering": "no",
                "Connection":       "keep-alive",
            },
        )

    return app



# Server runner (used by CLI)


def run_server(host: str = "127.0.0.1", port: int = 7890) -> None:
    """Start the dashboard with uvicorn.  Blocks until interrupted.

    Raises
    ImportError  if fastapi or uvicorn is not installed.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "Install dashboard dependencies: pip install 'agentforensics[dashboard]'"
        )
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_config=None)
