"""
Behavior diff engine - detects whether an agent's actions changed significantly
after a suspected injection turn.

Public API: 

build_baseline(events) -> dict
diff(events, pivot_turn) -> BehaviorDiff
"""
from collections import Counter
from typing import Any

from .models import BehaviorDiff, TraceEvent

# Constants


_BASELINE_TURNS = 5  # number of early turns used to build the baseline

# Tool name substrings that indicate sensitive / write-side-effect operations.
# A new action whose name contains any of these gets the anomaly score
# inflated by the multiplier below.
_SENSITIVE_KEYWORDS = frozenset([
    "send", "email", "post", "write", "delete",
    "execute", "run", "upload", "forward",
])
_SENSITIVE_MULTIPLIER = 1.5


def _is_sensitive(tool_name: str) -> bool:
    lower = tool_name.lower()
    return any(kw in lower for kw in _SENSITIVE_KEYWORDS)


# Helpers

def _tool_names_for_events(events: list[TraceEvent]) -> list[str]:
    """Return a flat, ordered list of all tool names called across `events`."""
    names: list[str] = []
    for event in events:
        for tc in event.tool_calls:
            names.append(tc.name)
    return names


def _sequential_pairs(events: list[TraceEvent]) -> list[tuple[str, str]]:
    """
    Return all consecutive (tool_a, tool_b) pairs.

    Pairs are drawn from:
    - tools called sequentially within the same turn.
    - the last tool of turn N → first tool of turn N+1 (cross-turn pairs).
    """
    pairs: list[tuple[str, str]] = []
    prev_last: str | None = None

    for event in events:
        names = [tc.name for tc in event.tool_calls]
        if not names:
            prev_last = None
            continue
        # Cross-turn pair
        if prev_last is not None:
            pairs.append((prev_last, names[0]))
        # Within-turn pairs
        for i in range(len(names) - 1):
            pairs.append((names[i], names[i + 1]))
        prev_last = names[-1]

    return pairs


# Public API

def build_baseline(events: list[TraceEvent], n_turns: int = _BASELINE_TURNS) -> dict[str, Any]:
    """
    Summarise tool-call behaviour over the first ``n_turns`` turns.

    Returns
    dict with keys:
        action_distribution : dict[str, float]   - normalised frequency per tool
        avg_tools_per_turn  : float
        common_patterns     : list[tuple[str,str]] - most frequent consecutive pairs
    """
    baseline_events = events[:n_turns]
    tool_names = _tool_names_for_events(baseline_events)
    n_turns_used = len(baseline_events)

    # Normalised action distribution
    counts = Counter(tool_names)
    total = sum(counts.values())
    if total > 0:
        action_distribution: dict[str, float] = {
            name: round(count / total, 4) for name, count in counts.items()
        }
    else:
        action_distribution = {}

    avg_tools_per_turn = round(total / max(1, n_turns_used), 4)

    # Common sequential pairs - keep the top-5 by frequency
    pairs = _sequential_pairs(baseline_events)
    pair_counts = Counter(pairs)
    common_patterns = [pair for pair, _ in pair_counts.most_common(5)]

    return {
        "action_distribution": action_distribution,
        "avg_tools_per_turn": avg_tools_per_turn,
        "common_patterns": common_patterns,
    }


def diff(events: list[TraceEvent], pivot_turn: int) -> BehaviorDiff:
    """
    Compare tool-call behaviour before vs. after ``pivot_turn``.

    Parameters:
    events      : all TraceEvents for the session, ordered by turn_index.
    pivot_turn  : the turn index suspected of carrying an injection.

    Returns:
    BehaviorDiff with:
        before_actions  - tool names called in turns < pivot_turn
        after_actions   - tool names called in turns >= pivot_turn
        new_actions     - tools in after that never appeared before
        anomaly_score   - 0.0–1.0, inflated when new actions are sensitive
    """
    before_events = [e for e in events if e.turn_index < pivot_turn]
    after_events  = [e for e in events if e.turn_index >= pivot_turn]

    before_actions = _tool_names_for_events(before_events)
    after_actions  = _tool_names_for_events(after_events)

    before_set = set(before_actions)
    new_actions = [name for name in after_actions if name not in before_set]
    # Deduplicate while preserving first-occurrence order
    seen: dict[str, None] = {}
    for name in new_actions:
        seen[name] = None
    new_actions_unique = list(seen)

    # Base score: ratio of novel actions to total prior action count
    base_score = len(new_actions_unique) / max(1, len(before_actions))

    # Inflate if any new action is sensitive
    if any(_is_sensitive(name) for name in new_actions_unique):
        base_score *= _SENSITIVE_MULTIPLIER

    anomaly_score = round(min(1.0, base_score), 4)

    return BehaviorDiff(
        pivot_turn=pivot_turn,
        before_actions=before_actions,
        after_actions=after_actions,
        new_actions=new_actions_unique,
        anomaly_score=anomaly_score,
    )
