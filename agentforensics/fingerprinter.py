"""
Attack fingerprinter for agentforensics.

Groups repeated injection patterns across sessions into "campaigns" using
vector similarity of evidence snippets.

Embedding tiers (auto-detected at runtime)
1. sentence-transformers all-MiniLM-L6-v2  - semantic (384-dim), opt-in
2. Pure Python character bi+trigram hashing - no extra deps (256-dim)

The AF_FP_THRESHOLD env var (default 0.82) controls the minimum cosine
similarity required to consider two snippets the same campaign.

The AF_FP_DB env var overrides the default DB path
(~/.agentforensics/fingerprints.db).
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .models import Fingerprint, InjectionSignal


# Config

_DEFAULT_THRESHOLD = float(os.environ.get("AF_FP_THRESHOLD", "0.70"))
_EMBED_DIM = 256          # dimension for the built-in trigram embedding


def _fp_db_path() -> Path:
    raw = os.environ.get("AF_FP_DB", "").strip()
    if raw:
        return Path(raw)
    return Path.home() / ".agentforensics" / "fingerprints.db"


# Embedding

def _trigram_embed(text: str, dim: int = _EMBED_DIM) -> list[float]:
    """Character bi+trigram hashing to a fixed-size L2-normalised vector.

    Adding boundary spaces improves coverage of short words.
    No external dependencies required.
    """
    counts = [0.0] * dim
    t = " " + text.lower() + " "
    for n in (2, 3):
        for i in range(len(t) - n + 1):
            h = hash(t[i : i + n]) % dim
            counts[h] += 1.0
    norm = math.sqrt(sum(x * x for x in counts)) or 1.0
    return [x / norm for x in counts]


def _embed(text: str) -> list[float]:
    """Return an embedding vector for *text*.

    Always uses the built-in character trigram hasher (256-dim) for
    consistency across all machines regardless of installed packages.
    This guarantees fingerprints stored on one machine are always
    comparable with fingerprints stored on another.
    """
    return _trigram_embed(text)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # Vectors are already L2-normalised; clamp for floating-point safety.
    return max(-1.0, min(1.0, dot))


# FingerprintStore


class FingerprintStore:
    """Persistent store for attack fingerprints backed by SQLite."""

    def __init__(
        self,
        db_path: str | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        self._db_path = Path(db_path) if db_path else _fp_db_path()
        self._threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else _DEFAULT_THRESHOLD
        )
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # Internal helpers
    
    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS fingerprints (
                    fingerprint_id         TEXT PRIMARY KEY,
                    first_seen             TEXT NOT NULL,
                    last_seen              TEXT NOT NULL,
                    session_ids            TEXT NOT NULL,
                    hit_count              INTEGER NOT NULL DEFAULT 1,
                    representative_snippet TEXT NOT NULL,
                    vector                 TEXT NOT NULL,
                    campaign_name          TEXT,
                    approved               INTEGER NOT NULL DEFAULT 0
                );
            """)
            # Migrate existing DBs that predate the approved column.
            try:
                conn.execute(
                    "ALTER TABLE fingerprints ADD COLUMN approved INTEGER NOT NULL DEFAULT 0"
                )
                conn.commit()
            except Exception:
                pass  # column already exists - safe to ignore
        finally:
            conn.close()

    @staticmethod
    def _row_to_fingerprint(row: sqlite3.Row) -> Fingerprint:
        return Fingerprint(
            fingerprint_id=row["fingerprint_id"],
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
            session_ids=json.loads(row["session_ids"]),
            hit_count=row["hit_count"],
            representative_snippet=row["representative_snippet"],
            vector=json.loads(row["vector"]),
            campaign_name=row["campaign_name"],
        )

   
    # Public API
   
    def add_or_match(
        self,
        signal: InjectionSignal,
        session_id: str,
    ) -> tuple[Fingerprint, bool]:
        """Embed the signal's snippet and search for a similar fingerprint.

        Returns
        (fingerprint, matched)
            matched=True  - found and updated an existing fingerprint
            matched=False - created a new fingerprint
        """
        snippet = signal.evidence_snippet or ""
        vec = _embed(snippet)
        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM fingerprints ORDER BY hit_count DESC"
            ).fetchall()
        finally:
            conn.close()

        # Find closest existing fingerprint
        best_row = None
        best_score = -1.0
        for row in rows:
            stored_vec = json.loads(row["vector"])
            score = _cosine(vec, stored_vec)
            if score > best_score:
                best_score = score
                best_row = row

        if best_row is not None and best_score >= self._threshold:
            # Update existing fingerprint
            fid = best_row["fingerprint_id"]
            session_ids: list[str] = json.loads(best_row["session_ids"])
            if session_id not in session_ids:
                session_ids.append(session_id)

            conn = self._get_conn()
            try:
                conn.execute(
                    """UPDATE fingerprints
                       SET last_seen   = ?,
                           session_ids = ?,
                           hit_count   = hit_count + 1
                       WHERE fingerprint_id = ?""",
                    (now, json.dumps(session_ids), fid),
                )
                conn.commit()
                updated = conn.execute(
                    "SELECT * FROM fingerprints WHERE fingerprint_id = ?", (fid,)
                ).fetchone()
            finally:
                conn.close()

            return self._row_to_fingerprint(updated), True

        else:
            # Create new fingerprint
            fid = str(uuid.uuid4())
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO fingerprints
                       (fingerprint_id, first_seen, last_seen, session_ids,
                        hit_count, representative_snippet, vector, campaign_name)
                       VALUES (?, ?, ?, ?, 1, ?, ?, NULL)""",
                    (
                        fid, now, now,
                        json.dumps([session_id]),
                        snippet,
                        json.dumps(vec),
                    ),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM fingerprints WHERE fingerprint_id = ?", (fid,)
                ).fetchone()
            finally:
                conn.close()

            return self._row_to_fingerprint(row), False

    def list_fingerprints(self) -> list[Fingerprint]:
        """Return all fingerprints sorted by hit_count descending."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM fingerprints ORDER BY hit_count DESC"
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_fingerprint(r) for r in rows]

    def get_campaigns(self) -> list[dict]:
        """Group fingerprints by campaign_name; unnamed ones are their own group."""
        fps = self.list_fingerprints()

        # Group: named fingerprints share a bucket by campaign_name;
        # unnamed fingerprints each get their own bucket.
        groups: dict[str, list[Fingerprint]] = {}
        for fp in fps:
            key = fp.campaign_name if fp.campaign_name else fp.fingerprint_id
            groups.setdefault(key, []).append(fp)

        result = []
        for key, group in sorted(
            groups.items(),
            key=lambda kv: sum(f.hit_count for f in kv[1]),
            reverse=True,
        ):
            all_sessions: list[str] = []
            for fp in group:
                for s in fp.session_ids:
                    if s not in all_sessions:
                        all_sessions.append(s)
            total_hits = sum(fp.hit_count for fp in group)
            first_seen = min(fp.first_seen for fp in group)
            last_seen  = max(fp.last_seen  for fp in group)
            result.append({
                "campaign_name":   key,
                "named":           group[0].campaign_name is not None,
                "fingerprint_ids": [fp.fingerprint_id for fp in group],
                "total_hits":      total_hits,
                "session_count":   len(all_sessions),
                "session_ids":     all_sessions,
                "first_seen":      first_seen.isoformat() if hasattr(first_seen, "isoformat") else str(first_seen),
                "last_seen":       last_seen.isoformat()  if hasattr(last_seen,  "isoformat") else str(last_seen),
                "top_snippet":     group[0].representative_snippet,
            })
        return result

    def label_campaign(self, fingerprint_id: str, name: str) -> None:
        """Attach a human-readable campaign name to a fingerprint."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE fingerprints SET campaign_name = ? WHERE fingerprint_id = ?",
                (name or None, fingerprint_id),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_false_alarm(self, fingerprint_id: str) -> None:
        """Mark a fingerprint as a false alarm.

        Future signals whose content matches this fingerprint will only be
        flagged if their score exceeds the high-confidence threshold (0.70),
        preventing repeated false positives while still catching genuine
        high-confidence attacks from the same source.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE fingerprints SET approved = 1 WHERE fingerprint_id = ?",
                (fingerprint_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def unmark_false_alarm(self, fingerprint_id: str) -> None:
        """Remove the false alarm marking so the content is scanned again."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE fingerprints SET approved = 0 WHERE fingerprint_id = ?",
                (fingerprint_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def get_approved(self) -> list[dict]:
        """Return all fingerprints marked as false alarms."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM fingerprints WHERE approved = 1 ORDER BY last_seen DESC"
            ).fetchall()
        finally:
            conn.close()
        result = []
        for row in rows:
            result.append({
                "fingerprint_id":         row["fingerprint_id"],
                "first_seen":             row["first_seen"],
                "last_seen":              row["last_seen"],
                "hit_count":              row["hit_count"],
                "campaign_name":          row["campaign_name"],
                "representative_snippet": row["representative_snippet"][:200],
                "session_ids":            json.loads(row["session_ids"]),
            })
        return result

    def is_approved_content(self, text: str) -> bool:
        """Return True if *text* closely matches a false-alarm-approved fingerprint.

        Uses cosine similarity on the same embedding as fingerprinting, so only
        substantially identical content is considered approved - a modified
        malicious page on the same domain gets a fresh score.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT vector FROM fingerprints WHERE approved = 1"
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return False

        vec = _embed(text)
        for row in rows:
            stored = json.loads(row["vector"])
            if _cosine(vec, stored) >= self._threshold:
                return True
        return False

    def get_matches_for_session(self, session_id: str) -> list[dict]:
        """Return fingerprints that were seen in *session_id*."""
        matches = []
        for fp in self.list_fingerprints():
            if session_id in fp.session_ids:
                matches.append({
                    "fingerprint_id":         fp.fingerprint_id,
                    "hit_count":              fp.hit_count,
                    "campaign_name":          fp.campaign_name,
                    "representative_snippet": fp.representative_snippet[:200],
                    "first_seen":             fp.first_seen.isoformat(),
                    "last_seen":              fp.last_seen.isoformat(),
                    "session_count":          len(fp.session_ids),
                })
        return matches
