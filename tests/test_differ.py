"""Tests for differ.py — build_baseline() and diff()."""
import uuid
from datetime import datetime, timezone

import pytest

from agentforensics.differ import build_baseline, diff, _is_sensitive
from agentforensics.models import Message, ToolCall, TraceEvent


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _tc(name: str, idx: int = 0) -> ToolCall:
    return ToolCall(id=f"{name}-{idx}", name=name, arguments={})


def _event(turn_index: int, tool_names: list[str], session_id: str | None = None) -> TraceEvent:
    sid = session_id or str(uuid.uuid4())
    return TraceEvent(
        event_id=str(uuid.uuid4()),
        session_id=sid,
        turn_index=turn_index,
        timestamp=datetime.now(timezone.utc),
        messages_in=[Message(role="user", content=f"Turn {turn_index}")],
        message_out=Message(role="assistant", content="ok"),
        tool_calls=[_tc(n, turn_index) for n in tool_names],
        model="gpt-4o",
        external_sources=[],
        prev_hash="",
        event_hash="x",
    )


def _session(tool_sequences: list[list[str]]) -> list[TraceEvent]:
    """Build a full session from a list of per-turn tool-name lists."""
    sid = str(uuid.uuid4())
    return [_event(i, names, sid) for i, names in enumerate(tool_sequences)]


# ---------------------------------------------------------------------------
# _is_sensitive
# ---------------------------------------------------------------------------

class TestIsSensitive:
    @pytest.mark.parametrize("name", [
        "send_email", "email_user", "post_data", "write_file",
        "delete_record", "execute_code", "run_script", "upload_doc", "forward_msg",
        "SEND_REPORT", "Email", "doPost",
    ])
    def test_sensitive_names(self, name):
        assert _is_sensitive(name) is True

    @pytest.mark.parametrize("name", [
        "search", "read_file", "fetch_url", "list_items", "get_weather", "calculate",
    ])
    def test_non_sensitive_names(self, name):
        assert _is_sensitive(name) is False


# ---------------------------------------------------------------------------
# build_baseline
# ---------------------------------------------------------------------------

class TestBuildBaseline:
    def test_action_distribution_normalized(self):
        events = _session([
            ["search", "search", "fetch"],  # search×2, fetch×1
            ["search"],
            [],
        ])
        bl = build_baseline(events)
        dist = bl["action_distribution"]
        # 3 search, 1 fetch → search=0.75, fetch=0.25
        assert abs(dist["search"] - 0.75) < 0.01
        assert abs(dist["fetch"] - 0.25) < 0.01
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.01

    def test_avg_tools_per_turn(self):
        events = _session([
            ["a", "b"],   # 2 tools
            ["c"],        # 1 tool
            [],           # 0 tools
        ])
        bl = build_baseline(events)
        # 3 tools over 3 turns → 1.0
        assert abs(bl["avg_tools_per_turn"] - 1.0) < 0.01

    def test_common_patterns_within_turn(self):
        events = _session([
            ["search", "summarise"],
            ["search", "summarise"],
        ])
        bl = build_baseline(events)
        assert ("search", "summarise") in bl["common_patterns"]

    def test_common_patterns_cross_turn(self):
        # Last tool of turn 0 → first tool of turn 1
        events = _session([
            ["fetch"],
            ["parse"],
        ])
        bl = build_baseline(events)
        assert ("fetch", "parse") in bl["common_patterns"]

    def test_uses_only_first_n_turns(self):
        # 10-turn session; baseline should only see first 5
        events = _session([
            ["early_tool"] if i < 5 else ["late_tool"] for i in range(10)
        ])
        bl = build_baseline(events)
        assert "early_tool" in bl["action_distribution"]
        assert "late_tool" not in bl["action_distribution"]

    def test_custom_n_turns(self):
        events = _session([["t0"], ["t1"], ["t2"], ["t3"], ["t4"], ["t5"]])
        bl = build_baseline(events, n_turns=2)
        assert "t0" in bl["action_distribution"]
        assert "t1" in bl["action_distribution"]
        assert "t2" not in bl["action_distribution"]

    def test_empty_events(self):
        bl = build_baseline([])
        assert bl["action_distribution"] == {}
        assert bl["avg_tools_per_turn"] == 0.0
        assert bl["common_patterns"] == []

    def test_no_tools_used(self):
        events = _session([[], [], []])
        bl = build_baseline(events)
        assert bl["action_distribution"] == {}
        assert bl["avg_tools_per_turn"] == 0.0

    def test_common_patterns_max_five(self):
        # 6 distinct pairs — should return at most 5
        events = _session([["a", "b", "c", "d", "e", "f", "g"]])
        bl = build_baseline(events)
        assert len(bl["common_patterns"]) <= 5


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

class TestDiff:
    def test_no_new_actions_zero_score(self):
        events = _session([
            ["search", "fetch"],   # before pivot
            ["search"],            # after pivot (same tools)
        ])
        result = diff(events, pivot_turn=1)
        assert result.new_actions == []
        assert result.anomaly_score == 0.0

    def test_new_actions_detected(self):
        events = _session([
            ["search", "fetch"],           # turns 0
            ["search", "send_email"],      # turn 1 — send_email is new
        ])
        result = diff(events, pivot_turn=1)
        assert "send_email" in result.new_actions
        assert "search" not in result.new_actions  # appeared before

    def test_before_after_actions_populated(self):
        events = _session([
            ["a", "b"],    # turn 0
            ["c", "d"],    # turn 1
        ])
        result = diff(events, pivot_turn=1)
        assert set(result.before_actions) == {"a", "b"}
        assert set(result.after_actions) == {"c", "d"}

    def test_pivot_at_zero_empty_before(self):
        events = _session([
            ["search"],
            ["send_email"],
        ])
        result = diff(events, pivot_turn=0)
        assert result.before_actions == []
        # new_actions: everything after pivot (turn 0 itself included in after)
        assert "search" in result.after_actions

    def test_sensitive_tool_inflates_score(self):
        # Use 3 before-actions so base = 1/3 ≈ 0.33 — well below the clamp
        # after 1.5× inflation (0.50), so clamping doesn't erase the difference.
        events = _session([
            ["read_file", "parse", "summarise"],   # 3 before actions
            ["send_email"],                        # 1 new sensitive action
        ])
        result_sensitive = diff(events, pivot_turn=1)

        events2 = _session([
            ["read_file", "parse", "summarise"],   # same 3 before actions
            ["analyse"],                           # 1 new non-sensitive action
        ])
        result_clean = diff(events2, pivot_turn=1)

        assert result_sensitive.anomaly_score > result_clean.anomaly_score

    def test_sensitive_multiplier_15x(self):
        # 1 before action, 1 new sensitive action
        # base = 1/1 = 1.0 × 1.5 = 1.5 → clamped to 1.0
        events = _session([
            ["read_file"],
            ["send_email"],
        ])
        result = diff(events, pivot_turn=1)
        assert result.anomaly_score == 1.0

    def test_anomaly_score_clamped_to_1(self):
        # Many new tools vs few before — score must not exceed 1.0
        events = _session([
            ["a"],
            ["b", "c", "d", "e", "f", "send_x"],
        ])
        result = diff(events, pivot_turn=1)
        assert result.anomaly_score <= 1.0

    def test_new_actions_deduplicated(self):
        # send_email appears twice after pivot but only once in new_actions
        events = _session([
            ["read_file"],
            ["send_email", "send_email"],
        ])
        result = diff(events, pivot_turn=1)
        assert result.new_actions.count("send_email") == 1

    def test_pivot_field_preserved(self):
        events = _session([["a"], ["b"], ["c"]])
        result = diff(events, pivot_turn=2)
        assert result.pivot_turn == 2

    def test_multiple_sensitive_tools(self):
        events = _session([
            ["read_file"] * 5,                      # 5 before actions
            ["send_email", "upload_doc", "execute"], # 3 new sensitive
        ])
        result = diff(events, pivot_turn=1)
        # base = 3/5 = 0.6, × 1.5 = 0.9
        assert abs(result.anomaly_score - 0.9) < 0.01
        assert set(result.new_actions) == {"send_email", "upload_doc", "execute"}

    def test_all_turns_are_before_pivot(self):
        # pivot beyond last turn — after_actions is empty
        events = _session([["a"], ["b"]])
        result = diff(events, pivot_turn=99)
        assert result.after_actions == []
        assert result.new_actions == []
        assert result.anomaly_score == 0.0
