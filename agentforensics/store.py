"""
Trace storage - SQLite-backed persistence for AgentForensics sessions.

Each session is stored in its own SQLite database file under AF_TRACE_DIR
(default: ~/.agentforensics/traces/<session_id>.db).

Schema

    events       - one row per LLM turn; includes SHA-256 hash chain fields
    session_meta - one row per session; tracks model, agent name, peak score
    signals      - one row per detected InjectionSignal; avoids re-running ML

Public API

    append_event(event, agent_name, notes)
    get_events(session_id) -> list[TraceEvent]
    compute_event_hash(event) -> str
    verify_chain(session_id) -> bool
    store_signal(session_id, signal)
    get_signals(session_id) -> list[InjectionSignal]
    update_session_signal_score(session_id, score)
    list_sessions() -> list[dict]
    delete_session(session_id)
"""
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

from .models import InjectionSignal, TraceEvent


def _sanitize_session_id(session_id: str) -> str:
    """Strip any characters that are not safe for use in file names."""
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)
    if not sanitized:
        raise ValueError(f"session_id {session_id!r} is empty after sanitization")
    return sanitized


def _trace_dir() -> Path:
    raw = os.environ.get("AF_TRACE_DIR", "")
    if raw:
        return Path(raw)
    return Path.home() / ".agentforensics" / "traces"


def _db_path(session_id: str) -> Path:
    safe = _sanitize_session_id(session_id)
    directory = _trace_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{safe}.db"


def _get_conn(session_id: str) -> sqlite3.Connection:
    path = _db_path(session_id)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            event_id    TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            turn_index  INTEGER NOT NULL,
            timestamp   TEXT NOT NULL,
            payload     TEXT NOT NULL,
            event_hash  TEXT NOT NULL,
            prev_hash   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS session_meta (
            session_id      TEXT PRIMARY KEY,
            started_at      TEXT NOT NULL,
            model           TEXT,
            agent_name      TEXT,
            notes           TEXT,
            max_signal_score REAL
        );

        CREATE TABLE IF NOT EXISTS signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            turn_index  INTEGER NOT NULL,
            payload     TEXT NOT NULL
        );
    """)
    # Migrate existing DBs that predate max_signal_score column.
    try:
        conn.execute("ALTER TABLE session_meta ADD COLUMN max_signal_score REAL")
        conn.commit()
    except Exception:
        pass  # column already exists
    conn.commit()


def store_signal(session_id: str, signal: "InjectionSignal") -> None:
    """Persist an InjectionSignal at detection time so the reporter never re-runs ML."""
    conn = _get_conn(session_id)
    try:
        conn.execute(
            "INSERT INTO signals (session_id, turn_index, payload) VALUES (?, ?, ?)",
            (session_id, signal.turn_index, signal.model_dump_json()),
        )
        conn.commit()
    finally:
        conn.close()


def get_signals(session_id: str) -> list["InjectionSignal"]:
    """Return all stored InjectionSignals for a session ordered by turn_index."""
    from .models import InjectionSignal
    conn = _get_conn(session_id)
    try:
        rows = conn.execute(
            "SELECT payload FROM signals WHERE session_id = ? ORDER BY turn_index ASC",
            (session_id,),
        ).fetchall()
        return [InjectionSignal.model_validate_json(r["payload"]) for r in rows]
    finally:
        conn.close()


def update_session_signal_score(session_id: str, score: float) -> None:
    """Persist the highest injection signal score seen so far for a session.

    Called by the Tracer when an alert fires so the dashboard can show the
    verdict without re-running expensive ML inference on every page load.
    """
    conn = _get_conn(session_id)
    try:
        row = conn.execute(
            "SELECT max_signal_score FROM session_meta WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        current = row["max_signal_score"] if row and row["max_signal_score"] is not None else 0.0
        if score > current:
            conn.execute(
                "UPDATE session_meta SET max_signal_score = ? WHERE session_id = ?",
                (score, session_id),
            )
            conn.commit()
    finally:
        conn.close()


def _canonical_json(event: TraceEvent) -> str:
    """Return deterministic JSON of the event excluding event_hash (to avoid circularity)."""
    data = event.model_dump(mode="json")
    data.pop("event_hash", None)
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


def compute_event_hash(event: TraceEvent) -> str:
    """Compute SHA-256 over prev_hash + canonical JSON of the event (minus event_hash)."""
    canonical = _canonical_json(event)
    raw = event.prev_hash + canonical
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def append_event(event: TraceEvent, agent_name: str | None = None, notes: str | None = None) -> None:
    """Insert a TraceEvent into its session DB. Raises if event_id already exists."""
    conn = _get_conn(event.session_id)
    try:
        payload = event.model_dump_json()
        conn.execute(
            "INSERT INTO events (event_id, session_id, turn_index, timestamp, payload, event_hash, prev_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.session_id,
                event.turn_index,
                event.timestamp.isoformat(),
                payload,
                event.event_hash,
                event.prev_hash,
            ),
        )
        # Upsert session_meta only if not already present (preserve original started_at).
        conn.execute(
            "INSERT OR IGNORE INTO session_meta (session_id, started_at, model, agent_name, notes) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                event.session_id,
                event.timestamp.isoformat(),
                event.model,
                agent_name,
                notes,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_events(session_id: str) -> list[TraceEvent]:
    """Return all TraceEvents for a session ordered by turn_index."""
    conn = _get_conn(session_id)
    try:
        rows = conn.execute(
            "SELECT payload FROM events WHERE session_id = ? ORDER BY turn_index ASC",
            (session_id,),
        ).fetchall()
        return [TraceEvent.model_validate_json(row["payload"]) for row in rows]
    finally:
        conn.close()


def verify_chain(session_id: str) -> bool:
    """
    Re-derive every event_hash from scratch and confirm it matches the stored column.

    Reads (payload, event_hash, prev_hash) columns directly so that tampering
    of any of these three fields is caught independently.
    """
    conn = _get_conn(session_id)
    try:
        rows = conn.execute(
            "SELECT payload, event_hash, prev_hash FROM events "
            "WHERE session_id = ? ORDER BY turn_index ASC",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return True

    expected_prev = ""
    for row in rows:
        stored_hash: str = row["event_hash"]
        stored_prev: str = row["prev_hash"]

        # 1. The prev_hash column must match what we expect from the previous event.
        if stored_prev != expected_prev:
            return False

        # 2. Recompute from the payload; use the stored prev_hash so the
        #    canonical JSON matches what was originally hashed.
        event = TraceEvent.model_validate_json(row["payload"])
        candidate = event.model_copy(update={"prev_hash": stored_prev, "event_hash": ""})
        recomputed = compute_event_hash(candidate)

        # 3. Recomputed hash must match the stored event_hash column.
        if recomputed != stored_hash:
            return False

        expected_prev = stored_hash

    return True


def list_sessions() -> list[dict]:
    """Return summary dicts for all sessions found in the trace directory."""
    trace_dir = _trace_dir()
    if not trace_dir.exists():
        return []

    results = []
    for db_file in sorted(trace_dir.glob("*.db")):
        try:
            conn = sqlite3.connect(str(db_file))
            conn.row_factory = sqlite3.Row
            meta_row = conn.execute("SELECT * FROM session_meta LIMIT 1").fetchone()
            count_row = conn.execute("SELECT COUNT(*) as cnt FROM events").fetchone()
            conn.close()

            if meta_row:
                results.append({
                    "session_id":       meta_row["session_id"],
                    "started_at":       meta_row["started_at"],
                    "model":            meta_row["model"],
                    "agent_name":       meta_row["agent_name"],
                    "turn_count":       count_row["cnt"] if count_row else 0,
                    "max_signal_score": meta_row["max_signal_score"],
                })
        except Exception:
            # Skip corrupt or unreadable DB files
            continue

    return results


def delete_session(session_id: str) -> None:
    """Remove a session DB file entirely (used by CLI clear command)."""
    path = _db_path(session_id)
    if path.exists():
        path.unlink()
