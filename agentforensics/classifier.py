"""
Injection classifier - four-stage detection.

Stage 1: fast regex/string heuristics (always available).
Stage 2: fine-tuned distilbert-base-uncased ML classifier (optional).
Stage 3: instruction boundary detection - rephrased command patterns.
Stage 4: semantic similarity - LLM response vs original question.
Stage 5: sliding window - scores last N turns together to catch
         multi-turn attacks spread across multiple conversations.

Public API
classify(event: TraceEvent) -> list[InjectionSignal]
classify_window(event: TraceEvent) -> list[InjectionSignal]
"""
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from .models import InjectionSignal, Message, TraceEvent

# Configuration

def _threshold() -> float:
    raw = os.environ.get("AF_INJECT_THRESHOLD", "0.25")
    try:
        return float(raw)
    except ValueError:
        return 0.25


def _window_size() -> int:
    """Number of turns to include in the sliding window check (0 = disabled)."""
    try:
        return max(0, int(os.environ.get("AF_WINDOW_SIZE", "3")))
    except ValueError:
        return 3


# Signals from approved (false-alarm) content must clear this higher bar.
_FALSE_ALARM_THRESHOLD = 0.70


def _is_approved_content(text: str) -> bool:
    """Check if text matches a false-alarm-approved fingerprint."""
    try:
        from .fingerprinter import FingerprintStore
        return FingerprintStore().is_approved_content(text)
    except Exception:
        return False


def _model_path() -> Path:
    raw = os.environ.get("AF_MODEL_PATH", "")
    if raw:
        return Path(raw)
    return Path(__file__).parent / "model"


def _ml_disabled() -> bool:
    return os.environ.get("AF_DISABLE_ML", "false").lower() in ("1", "true", "yes")



# Stage 1 — heuristic rules

class _Rule(NamedTuple):
    rule_id: str
    pattern: re.Pattern
    weight: float
    description: str


def _compile(rule_id: str, pattern: str, weight: float, description: str) -> _Rule:
    return _Rule(rule_id, re.compile(pattern, re.IGNORECASE | re.DOTALL), weight, description)


_HEURISTIC_RULES: list[_Rule] = [
    _compile(
        "H01",
        r"ignore\s+(previous|all|prior)\s+instructions",
        0.35,
        "Classic override",
    ),
    _compile(
        "H02",
        r"(you\s+are\s+now\b|your\s+new\s+(role|persona|instructions)\b)",
        0.30,
        "Role hijack",
    ),
    _compile(
        "H03",
        r"do\s+not\s+(tell|inform|mention)\s+the\s+user",
        0.25,
        "Concealment directive",
    ),
    _compile(
        "H04",
        r"(\[\[.{0,200}?\]\]|<\|[a-zA-Z_]{1,30}\|>|\[INST\]|</?(s|INST|SYS)>)",
        0.15,
        "Encoded/token injection markers",
    ),
    # H05 is evaluated with a dedicated function (see _check_h05)
    _compile(
        "H06",
        r"(?m)^(send|forward|email|post|upload|exfiltrate|leak|steal|"
        r"ignore|disregard|pretend|act\s+as|respond\s+as|output|print|"
        r"reveal|expose|dump|return)\s+\w",
        0.20,
        "Imperative instruction in tool output",
    ),
    _compile(
        "H07",
        r"(system\s+prompt|context\s+window|your\s+instructions|"
        r"initial\s+(instructions?|prompt)|original\s+(instructions?|prompt)|"
        r"base\s+(instructions?|prompt)|hidden\s+instructions?)",
        0.25,
        "Reference to system prompt / context window",
    ),
    _compile(
        "H08",
        r"(send|forward|email|post|upload)\b.{0,80}"
        r"(secret|password|passwd|token|api[_\s]?key|private[_\s]?key|credential)",
        0.40,
        "Exfiltration setup: send/forward near secret/credential keywords",
    ),
]

# H05 — language-switch detection (separate logic, same weight: 0.20)
_H05_WEIGHT = 0.20
_H05_ID = "H05"
_H05_DESC = "Sudden language switch mid-document (evasion attempt)"

# Script ranges for non-Latin alphabetic characters
_NON_LATIN_SCRIPTS = frozenset([
    "Arabic", "Hebrew", "Devanagari", "Bengali", "Gujarati",
    "Gurmukhi", "Kannada", "Malayalam", "Oriya", "Tamil", "Telugu",
    "Thai", "Georgian", "Armenian", "Ethiopic", "Tibetan",
    "CJK Unified Ideographs", "Hiragana", "Katakana", "Hangul Syllables",
    "Cyrillic",
])


def _char_script(ch: str) -> str:
    try:
        name = unicodedata.name(ch, "")
        # Return the first two words of the Unicode name as a rough script marker
        return " ".join(name.split()[:3])
    except Exception:
        return ""


def _check_h05(text: str) -> bool:
    """
    Return True if the text looks predominantly ASCII/Latin but contains a
    non-trivial block of characters from a different script - a common evasion
    tactic where attackers embed instructions in a second language.
    """
    if not text:
        return False
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) < 20:
        return False

    non_latin_count = sum(
        1 for c in alpha_chars
        if ord(c) > 0x024F  # above extended Latin block
    )
    ratio = non_latin_count / len(alpha_chars)
    # Flag whenever both scripts are meaningfully present (5-95%).
    # A fully non-Latin document is fine; a fully Latin document is fine;
    # a mix in either direction is suspicious.
    return 0.05 <= ratio <= 0.95


def _run_heuristics(text: str) -> tuple[float, list[str], str]:
    """
    Run all heuristic rules against 'text'.

    Returns (score, matched_rule_ids, evidence_snippet).
    """
    matched: list[str] = []
    total_weight = 0.0
    snippets: list[str] = []

    for rule in _HEURISTIC_RULES:
        m = rule.pattern.search(text)
        if m:
            matched.append(rule.rule_id)
            total_weight += rule.weight
            # Grab context around the match for the evidence snippet
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            snippets.append(text[start:end].strip())

    if _check_h05(text):
        matched.append(_H05_ID)
        total_weight += _H05_WEIGHT
        # No match object; just use the beginning of the text as evidence
        snippets.append(text[:100].strip())

    score = min(1.0, total_weight)
    evidence = " … ".join(snippets)[:500]
    return score, matched, evidence


# Stage 2 - ML classifier (lazy-loaded)

_ml_pipeline = None      # transformers pipeline, loaded on demand
_ml_load_attempted = False
_ml_available = False


def _try_load_ml() -> bool:
    """
    Attempt to load the fine-tuned classifier from disk.  Sets module-level
    flags so the load is only attempted once.  Returns True if successful.
    """
    global _ml_pipeline, _ml_load_attempted, _ml_available
    if _ml_load_attempted:
        return _ml_available
    _ml_load_attempted = True

    if _ml_disabled():
        warnings.warn(
            "AF_DISABLE_ML=true — using heuristics-only injection detection.",
            UserWarning,
            stacklevel=3,
        )
        _ml_available = False
        return False

    model_dir = _model_path()
    if not model_dir.exists() or not any(model_dir.iterdir()):
        warnings.warn(
            f"No trained classifier found at '{model_dir}'. "
            "Running in heuristics-only mode. "
            "Train a model with: python scripts/train_classifier.py",
            UserWarning,
            stacklevel=3,
        )
        _ml_available = False
        return False

    try:
        from transformers import pipeline as hf_pipeline  # type: ignore
        _ml_pipeline = hf_pipeline(
            "text-classification",
            model=str(model_dir),
            tokenizer=str(model_dir),
            truncation=True,
            max_length=512,
        )
        _ml_available = True
        return True
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Failed to load ML classifier ({exc}). "
            "Running in heuristics-only mode.",
            UserWarning,
            stacklevel=3,
        )
        _ml_available = False
        return False


def _ml_score(text: str) -> float:
    """
    Return injection probability from the ML model (0.0–1.0).
    Caller must have verified _ml_available is True.
    """
    result = _ml_pipeline(text[:2000])  # guard extremely long inputs
    # HuggingFace pipeline returns [{"label": "LABEL_1", "score": 0.98}]
    if isinstance(result, list) and result:
        item = result[0]
        label: str = item.get("label", "LABEL_0")
        score: float = item.get("score", 0.0)
        # LABEL_1 → injection; LABEL_0 → clean
        if label.endswith("1") or label.lower() in ("injection", "positive", "malicious"):
            return score
        return 1.0 - score
    return 0.0


# External-source message identification

_URL_RE = re.compile(r"https?://[^\s<>\"{}|\\^`\[\]]+")
_PATH_RE = re.compile(
    r"(?:^|[\s(\"'])(/[a-zA-Z0-9_./ \-]{3,}|[A-Za-z]:\\[^\s]{2,}|\./[^\s]{2,})"
)


def _first_source_in_text(text: str, external_sources: list[str]) -> str | None:
    """Return the first external_source string that appears in `text`, or None."""
    for src in external_sources:
        if src in text:
            return src
    # Fallback: any URL present in the text
    m = _URL_RE.search(text)
    if m:
        return m.group(0)
    return None


def _is_external_message(msg: Message, external_sources: list[str]) -> bool:
    """
    True if the message plausibly carries content from an external source:
    all tool messages qualify; user messages qualify only when they contain
    a URL or file path.
    """
    if msg.role == "tool":
        return True
    if msg.role == "user":
        return bool(_URL_RE.search(msg.content) or _PATH_RE.search(msg.content))
    return False


# Public API

def classify(event: TraceEvent) -> list[InjectionSignal]:
    """
    Analyse every external-source message in 'event' and return a list of
    InjectionSignals for those whose final score meets the threshold.

    Stage 1 - Heuristic regex rules             (always)
    Stage 2 - ML injection classifier           (when model available)
    Stage 3 - Instruction boundary detection    (always)
    Stage 4 - Semantic similarity check         (when model available)

    Per-message score:
        ML available  : 0.30 * h_score + 0.40 * ml_score + 0.30 * ib_score
        Heuristics only: 0.50 * h_score + 0.50 * ib_score

    Stage 4 produces a separate event-level signal when the LLM response
    is semantically unrelated to the user's question (possible hijack).
    """
    from .semantic import instruction_boundary_score, semantic_injection_score

    threshold = _threshold()
    ml_ready  = _try_load_ml()
    signals: list[InjectionSignal] = []

    for msg in event.messages_in:
        if not _is_external_message(msg, event.external_sources):
            continue

        text = msg.content
        if not text:
            continue

        # Stage 1: Heuristics
        h_score, matched_rules, evidence = _run_heuristics(text)

        # Stage 3: Instruction Boundary
        ib_score, ib_rules, ib_evidence = instruction_boundary_score(text)
        matched_rules = matched_rules + ib_rules
        if ib_evidence and not evidence:
            evidence = ib_evidence

        # Stage 2: ML classifier
        if ml_ready:
            m_score     = _ml_score(text)
            final_score = round(0.30 * h_score + 0.40 * m_score + 0.30 * ib_score, 4)
        else:
            final_score = round(0.50 * h_score + 0.50 * ib_score, 4)

        if final_score < threshold:
            continue

        # False-alarm check: if content matches an approved fingerprint,
        # only flag when score clears the high-confidence bar (0.70).
        if final_score < _FALSE_ALARM_THRESHOLD and _is_approved_content(text):
            continue

        source = _first_source_in_text(text, event.external_sources)

        signals.append(InjectionSignal(
            turn_index=event.turn_index,
            score=final_score,
            matched_heuristics=matched_rules if matched_rules else ["ML only"],
            source=source,
            evidence_snippet=evidence[:500],
        ))

    # Stage 4: Semantic similarity (event-level)
    sem_signal = _check_semantic_similarity(event, threshold, semantic_injection_score)
    if sem_signal:
        signals.append(sem_signal)

    return signals


def _check_semantic_similarity(event: TraceEvent, threshold: float, sem_fn) -> InjectionSignal | None:
    """
    Compare the user's question to the LLM's response.
    Returns an InjectionSignal if the response is semantically unrelated.
    """
    user_msgs = [m for m in event.messages_in if m.role == "user"]
    if not user_msgs:
        return None

    question = user_msgs[-1].content
    response = event.message_out.content if event.message_out else ""

    if not question.strip() or not response.strip():
        return None

    sem_score = sem_fn(question, response)
    if sem_score < threshold:
        return None

    return InjectionSignal(
        turn_index=event.turn_index,
        score=sem_score,
        matched_heuristics=["SEM01"],
        source=None,
        evidence_snippet=(
            f"Low semantic similarity between question and response "
            f"(score={sem_score:.3f}) — response may be unrelated to original request."
        ),
    )


# Stage 5 — Sliding window multi-turn detection

def classify_window(event: TraceEvent) -> list[InjectionSignal]:
    """
    Sliding window check: concatenate external-source messages from the last
    AF_WINDOW_SIZE turns (default 3) and score them as a single block.

    Only called when per-turn classify() found nothing - acts as a
    "second chance" detector for injections spread across multiple turns.

    Returns a list with at most one InjectionSignal tagged with WIN01.
    """
    size = _window_size()
    if size < 2:
        return []

    # Load all events for this session (current turn already stored).
    from .store import get_events
    from .semantic import instruction_boundary_score

    all_events = get_events(event.session_id)
    window = all_events[-size:]   # last N turns including current

    if len(window) < 2:
        return []   # need at least 2 turns for window to add value

    # Collect external-source content across all turns in the window.
    combined_parts: list[str] = []
    all_sources: list[str] = []

    for ev in window:
        all_sources.extend(ev.external_sources)
        for msg in ev.messages_in:
            if _is_external_message(msg, ev.external_sources) and msg.content:
                combined_parts.append(msg.content)

    if not combined_parts:
        return []

    combined_text = "\n\n---\n\n".join(combined_parts)

    # Run Stage 1 + 3 on combined text.
    h_score, matched_rules, evidence = _run_heuristics(combined_text)
    ib_score, ib_rules, ib_evidence = instruction_boundary_score(combined_text)
    matched_rules = matched_rules + ib_rules
    if ib_evidence and not evidence:
        evidence = ib_evidence

    # Stage 2 ML on combined text.
    ml_ready = _try_load_ml()
    if ml_ready:
        m_score     = _ml_score(combined_text)
        final_score = round(0.30 * h_score + 0.40 * m_score + 0.30 * ib_score, 4)
    else:
        final_score = round(0.50 * h_score + 0.50 * ib_score, 4)

    threshold = _threshold()
    if final_score < threshold:
        return []

    source = _first_source_in_text(combined_text, list(dict.fromkeys(all_sources)))
    rules  = (matched_rules if matched_rules else ["WIN_ML"]) + ["WIN01"]

    return [InjectionSignal(
        turn_index=event.turn_index,
        score=final_score,
        matched_heuristics=rules,
        source=source,
        evidence_snippet=f"[Window {len(window)} turns] {evidence[:460]}",
    )]
