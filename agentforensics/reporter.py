"""
Report generator - produces a ForensicReport from a session's trace store,
then serialises it as JSON or as a self-contained HTML file.

Public API
generate(session_id, output="html") -> str
"""
import html as _html
import json
from datetime import datetime, timezone
from typing import Literal

from .classifier import classify
from .differ import diff as behavior_diff
from .models import BehaviorDiff, ForensicReport, InjectionSignal, TraceEvent
from .store import get_events, get_signals, verify_chain


# Verdict / confidence logic

def _determine_verdict(
    signals: list[InjectionSignal],
    diffs: list[BehaviorDiff],
) -> str:
    if not signals:
        return "clean"
    max_signal = max(s.score for s in signals)
    max_anomaly = max((d.anomaly_score for d in diffs), default=0.0)
    if max_signal >= 0.65 and max_anomaly >= 0.5:
        return "compromised"
    return "suspicious"


def _determine_confidence(
    signals: list[InjectionSignal],
    chain_valid: bool,
) -> float:
    if not chain_valid:
        return 0.0
    if not signals:
        return 0.0
    return round(max(s.score for s in signals), 4)


# Deterministic summary template

def _build_summary(report: ForensicReport) -> str:
    short_id = report.session_id[:12] + "…"
    chain_note = (
        "Hash chain verified."
        if report.chain_valid
        else "WARNING: hash chain integrity check failed - trace may have been tampered with."
    )

    if report.verdict == "clean":
        return (
            f"Session {short_id} completed {report.total_turns} turn(s) "
            f"with no injection signals above threshold. {chain_note}"
        )

    if report.verdict == "suspicious":
        top = max(report.injection_signals, key=lambda s: s.score)
        src = f" from source '{top.source}'" if top.source else ""
        diff_note = (
            "Behaviour analysis did not confirm a significant action pivot."
            if not report.behavior_diffs
            else f"Behaviour anomaly score: {max(d.anomaly_score for d in report.behavior_diffs):.2f}."
        )
        return (
            f"Session {short_id} shows {len(report.injection_signals)} potential injection "
            f"signal(s) across {report.total_turns} turn(s). "
            f"Highest signal score {top.score:.2f} at turn {top.turn_index}{src}. "
            f"{diff_note} {chain_note}"
        )

    # compromised
    top = max(report.injection_signals, key=lambda s: s.score)
    src = f" from '{top.source}'" if top.source else ""
    top_diff = max(report.behavior_diffs, key=lambda d: d.anomaly_score)
    new_str = ", ".join(top_diff.new_actions[:5]) or "none"
    if len(top_diff.new_actions) > 5:
        new_str += f" (+{len(top_diff.new_actions) - 5} more)"
    return (
        f"Session {short_id} appears compromised. "
        f"Injection signal score {top.score:.2f} detected at turn {top.turn_index}{src}. "
        f"After the pivot, {len(top_diff.new_actions)} new action(s) appeared ({new_str}) "
        f"with anomaly score {top_diff.anomaly_score:.2f}. {chain_note}"
    )

# Core generate() function

def generate(
    session_id: str,
    output: Literal["json", "html"] = "html",
) -> str:
    """
    Build a ForensicReport for ''session_id'' and return it as a JSON or HTML string.
    """
    events = get_events(session_id)
    chain_valid = verify_chain(session_id)

    # Use pre-stored signals if available (written at detection time - no ML needed).
    # Fall back to re-classifying only if no stored signals exist (old sessions).
    all_signals: list[InjectionSignal] = get_signals(session_id)
    if not all_signals:
        for event in events:
            all_signals.extend(classify(event))

    # Run behavior diff for each high-confidence signal (score >= 0.5)
    all_diffs: list[BehaviorDiff] = []
    seen_pivots: set[int] = set()
    for signal in all_signals:
        if signal.score >= 0.5 and signal.turn_index not in seen_pivots:
            seen_pivots.add(signal.turn_index)
            all_diffs.append(behavior_diff(events, signal.turn_index))

    verdict = _determine_verdict(all_signals, all_diffs)
    confidence = _determine_confidence(all_signals, chain_valid)

    report = ForensicReport(
        session_id=session_id,
        generated_at=datetime.now(timezone.utc),
        total_turns=len(events),
        chain_valid=chain_valid,
        injection_signals=all_signals,
        behavior_diffs=all_diffs,
        verdict=verdict,
        confidence=confidence,
        summary="",  # filled below to break circular dependency
    )
    report = report.model_copy(update={"summary": _build_summary(report)})

    # Fingerprint matches - look up which campaigns touched this session.
    fingerprint_matches: list[dict] = []
    try:
        from .fingerprinter import FingerprintStore
        fingerprint_matches = FingerprintStore().get_matches_for_session(session_id)
    except Exception:  # noqa: BLE001
        pass

    report = report.model_copy(update={"fingerprint_matches": fingerprint_matches})

    if output == "json":
        return report.model_dump_json(indent=2)

    return _render_html(report, events)


# HTML renderer

def _e(text: object) -> str:
    """HTML-escape a value."""
    return _html.escape(str(text))


def _verdict_color(verdict: str) -> tuple[str, str]:
    """Return (bg_hex, label) for a verdict badge."""
    return {
        "clean":       ("#22c55e", "CLEAN"),
        "suspicious":  ("#f59e0b", "SUSPICIOUS"),
        "compromised": ("#ef4444", "COMPROMISED"),
    }.get(verdict, ("#6b7280", verdict.upper()))


def _score_bar(score: float, width: int = 120) -> str:
    filled = int(score * width)
    color = "#22c55e" if score < 0.35 else ("#f59e0b" if score < 0.65 else "#ef4444")
    return (
        f'<span style="display:inline-block;width:{width}px;height:10px;'
        f'background:#e5e7eb;border-radius:4px;vertical-align:middle;">'
        f'<span style="display:block;width:{filled}px;height:10px;'
        f'background:{color};border-radius:4px;"></span></span>'
        f'&nbsp;<code style="font-size:0.8em">{score:.3f}</code>'
    )


def _render_timeline(events: list[TraceEvent], signals: list[InjectionSignal]) -> str:
    signal_by_turn: dict[int, InjectionSignal] = {}
    for s in signals:
        if s.turn_index not in signal_by_turn or s.score > signal_by_turn[s.turn_index].score:
            signal_by_turn[s.turn_index] = s

    rows = []
    for event in events:
        sig = signal_by_turn.get(event.turn_index)
        bg = ""
        badge = ""
        if sig:
            if sig.score >= 0.65:
                bg = "background:#fef2f2;border-left:4px solid #ef4444;"
            else:
                bg = "background:#fffbeb;border-left:4px solid #f59e0b;"
            badge = (
                f'<span style="float:right;padding:2px 8px;border-radius:12px;'
                f'background:{"#ef4444" if sig.score>=0.65 else "#f59e0b"};'
                f'color:#fff;font-size:0.75em;font-weight:700;">'
                f'INJECTION {sig.score:.2f}</span>'
            )

        # Build message previews
        msg_rows = []
        for msg in event.messages_in:
            preview = _e(msg.content[:200]) + ("…" if len(msg.content) > 200 else "")
            role_color = {
                "system": "#7c3aed", "user": "#2563eb",
                "assistant": "#059669", "tool": "#d97706",
            }.get(msg.role, "#374151")
            msg_rows.append(
                f'<div style="margin:2px 0">'
                f'<span style="color:{role_color};font-weight:600;font-size:0.8em">'
                f'[{_e(msg.role)}]</span> '
                f'<span style="color:#374151;font-size:0.85em">{preview}</span>'
                f'</div>'
            )

        tools_html = ""
        if event.tool_calls:
            tool_tags = "".join(
                f'<code style="background:#e0e7ff;color:#3730a3;padding:1px 6px;'
                f'border-radius:4px;margin-right:4px;font-size:0.8em">{_e(tc.name)}</code>'
                for tc in event.tool_calls
            )
            tools_html = f'<div style="margin-top:4px">Tools: {tool_tags}</div>'

        rows.append(
            f'<div style="padding:10px 14px;margin-bottom:6px;border-radius:6px;'
            f'border:1px solid #e5e7eb;{bg}">'
            f'{badge}'
            f'<span style="font-weight:700;color:#111">Turn {event.turn_index}</span>'
            f'<span style="color:#6b7280;font-size:0.8em;margin-left:8px">'
            f'{_e(event.model)} · {_e(event.timestamp.strftime("%H:%M:%S UTC"))}</span>'
            f'<div style="margin-top:6px">{"".join(msg_rows)}</div>'
            f'{tools_html}'
            f'</div>'
        )

    return "\n".join(rows)


_HEURISTIC_LABELS: dict[str, str] = {
    # Stage 1 - regex heuristics
    "H01":   "Classic instruction override - 'ignore previous instructions'",
    "H02":   "Role or persona hijack attempt",
    "H03":   "Concealment directive - telling the AI to hide information from the user",
    "H04":   "Encoded / token-based injection markers",
    "H05":   "Language-switch evasion - instructions embedded in a second script/language",
    "H06":   "Imperative command embedded in tool or external content output",
    "H07":   "Reference to system prompt or context window",
    "H08":   "Credential or secret exfiltration setup",
    # Stage 3 - instruction boundary
    "IB01":  "Temporal override marker - 'from this point forward …'",
    "IB02":  "Task redefinition - injected content attempts to rewrite the AI's purpose",
    "IB03":  "Response constraint - forcing the AI to reply in a specific restricted way",
    "IB04":  "Directive override - 'you must / will / should … ignore / forget'",
    "IB05":  "Context wipe - 'forget / disregard everything above'",
    "IB06":  "Concealment instruction - telling the AI not to reveal something",
    "IB07":  "Persona hijack - 'act as / pretend to be / you are now'",
    "IB08":  "Output constraint - 'output only / reply with exactly'",
    "IB09":  "New directive injection - explicit 'new instruction / command / rule'",
    "IB10":  "Greeting-behaviour hijack - manipulating how the AI introduces itself",
    # Stage 2 / 4 - ML / semantic
    "SEM01": "Semantic anomaly - AI response diverged significantly from the expected topic",
    "ML only": "ML classifier detected an injection pattern (no specific heuristic matched)",
}


def _render_evidence(signals: list[InjectionSignal]) -> str:
    if not signals:
        return '<p style="color:#6b7280;font-style:italic">No injection signals detected.</p>'

    cards = []
    for i, sig in enumerate(sorted(signals, key=lambda s: s.score, reverse=True)):
        def _heuristic_badge(h: str) -> str:
            label = _HEURISTIC_LABELS.get(h, "")
            badge = (
                f'<code style="background:#fef3c7;color:#92400e;padding:2px 6px;'
                f'border-radius:4px;margin-right:6px;font-size:0.8em">{_e(h)}</code>'
            )
            desc = (
                f'<span style="font-size:0.82em;color:#374151">{_e(label)}</span>'
                if label else ""
            )
            return (
                f'<span style="display:inline-flex;align-items:center;'
                f'margin-bottom:4px;margin-right:8px">{badge}{desc}</span>'
            )

        heuristics_html = (
            '<div style="display:flex;flex-wrap:wrap;gap:2px">'
            + "".join(_heuristic_badge(h) for h in sig.matched_heuristics)
            + '</div>'
        ) if sig.matched_heuristics else '<span style="color:#6b7280;font-style:italic">none (ML only)</span>'

        src_html = (
            f'<div><strong>Source:</strong> '
            f'<a href="{_e(sig.source)}" style="color:#2563eb;word-break:break-all">'
            f'{_e(sig.source)}</a></div>'
            if sig.source else
            '<div><strong>Source:</strong> <em>unknown</em></div>'
        )

        snippet_html = (
            f'<pre style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:4px;'
            f'padding:8px;font-size:0.82em;white-space:pre-wrap;word-break:break-word;'
            f'max-height:150px;overflow-y:auto">{_e(sig.evidence_snippet)}</pre>'
            if sig.evidence_snippet else ""
        )

        cards.append(
            f'<div style="border:1px solid #fca5a5;border-radius:8px;padding:14px;'
            f'margin-bottom:12px;background:#fff5f5">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-bottom:8px">'
            f'<strong>Signal #{i+1} - Turn {sig.turn_index}</strong>'
            f'<span>{_score_bar(sig.score)}</span>'
            f'</div>'
            f'{src_html}'
            f'<div style="margin:6px 0"><strong>Matched heuristics:</strong> {heuristics_html}</div>'
            f'{snippet_html}'
            f'</div>'
        )
    return "\n".join(cards)


def _render_diffs(diffs: list[BehaviorDiff]) -> str:
    if not diffs:
        return '<p style="color:#6b7280;font-style:italic">No behavior diffs computed.</p>'

    cards = []
    for d in sorted(diffs, key=lambda x: x.anomaly_score, reverse=True):
        before_html = "".join(
            f'<code style="background:#e0e7ff;color:#3730a3;padding:1px 6px;'
            f'border-radius:4px;margin-right:4px;margin-bottom:4px;'
            f'display:inline-block;font-size:0.82em">{_e(a)}</code>'
            for a in dict.fromkeys(d.before_actions)
        ) or '<em style="color:#6b7280">none</em>'

        after_html_parts = []
        before_set = set(d.before_actions)
        for a in dict.fromkeys(d.after_actions):
            is_new = a in d.new_actions
            style = (
                "background:#fee2e2;color:#991b1b;border:1px solid #fca5a5;"
                if is_new else
                "background:#dcfce7;color:#166534;"
            )
            after_html_parts.append(
                f'<code style="{style}padding:1px 6px;border-radius:4px;'
                f'margin-right:4px;margin-bottom:4px;display:inline-block;font-size:0.82em">'
                f'{_e(a)}{"  ★" if is_new else ""}</code>'
            )
        after_html = "".join(after_html_parts) or '<em style="color:#6b7280">none</em>'

        cards.append(
            f'<div style="border:1px solid #d1d5db;border-radius:8px;padding:14px;'
            f'margin-bottom:12px;background:#fafafa">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:10px">'
            f'<strong>Pivot at turn {d.pivot_turn}</strong>'
            f'<span>Anomaly: {_score_bar(d.anomaly_score)}</span>'
            f'</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">'
            f'<div><div style="font-size:0.8em;color:#6b7280;margin-bottom:4px">'
            f'BEFORE PIVOT</div>{before_html}</div>'
            f'<div><div style="font-size:0.8em;color:#6b7280;margin-bottom:4px">'
            f'AFTER PIVOT <span style="color:#991b1b">(★ = new)</span></div>'
            f'{after_html}</div>'
            f'</div>'
            f'</div>'
        )
    return "\n".join(cards)


def _render_html(report: ForensicReport, events: list[TraceEvent]) -> str:
    verdict_bg, verdict_label = _verdict_color(report.verdict)

    chain_html = (
        '<span style="color:#22c55e;font-weight:700">&#10003; Chain Verified</span>'
        if report.chain_valid else
        '<span style="color:#ef4444;font-weight:700">'
        '&#9888; Chain INVALID - trace may have been tampered</span>'
    )

    def section(title: str, body: str, anchor: str = "") -> str:
        id_attr = f'id="{_e(anchor)}"' if anchor else ""
        return (
            f'<section {id_attr} style="margin-bottom:32px">'
            f'<h2 style="font-size:1.1em;font-weight:700;color:#111;'
            f'border-bottom:2px solid #e5e7eb;padding-bottom:6px;margin-bottom:14px">'
            f'{_e(title)}</h2>'
            f'{body}'
            f'</section>'
        )

    timeline_html = _render_timeline(events, report.injection_signals)
    evidence_html = _render_evidence(report.injection_signals)
    diffs_html    = _render_diffs(report.behavior_diffs)

    nav_links = (
        '<a href="#timeline">Timeline</a> · '
        '<a href="#evidence">Evidence</a> · '
        '<a href="#diffs">Behavior Diff</a>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgentForensics Report - {_e(report.session_id)}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    font-size: 14px; line-height: 1.6; color: #111; background: #f3f4f6;
  }}
  .container {{ max-width: 960px; margin: 0 auto; padding: 24px 16px; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; }}
  .card {{
    background: #fff; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
    padding: 20px 24px; margin-bottom: 24px;
  }}
  .header-grid {{
    display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: start;
  }}
  .verdict-badge {{
    padding: 6px 18px; border-radius: 20px; color: #fff; font-weight: 700;
    font-size: 1em; letter-spacing: .04em; white-space: nowrap;
  }}
  .meta {{ color: #6b7280; font-size: 0.85em; margin-top: 6px; }}
  .summary-box {{
    background: #f0f9ff; border-left: 4px solid #38bdf8;
    padding: 10px 14px; border-radius: 4px; margin-top: 14px;
    font-size: 0.9em; color: #0c4a6e;
  }}
  .stat-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 12px; margin-top: 14px;
  }}
  .stat {{
    text-align: center; background: #f9fafb; border: 1px solid #e5e7eb;
    border-radius: 8px; padding: 10px;
  }}
  .stat .value {{ font-size: 1.5em; font-weight: 700; color: #111; }}
  .stat .label {{ font-size: 0.75em; color: #6b7280; text-transform: uppercase; }}
  .timeline-scroll {{ max-height: 500px; overflow-y: auto; padding-right: 4px; }}
  .nav {{ font-size: 0.85em; color: #6b7280; margin-bottom: 20px; }}
  @media print {{
    .timeline-scroll {{ max-height: none; overflow: visible; }}
    pre {{ max-height: none !important; overflow: visible !important; }}
    .section {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="container">

  <!-- ── Header ── -->
  <div class="card">
    <div class="header-grid">
      <div>
        <h1 style="font-size:1.3em;font-weight:800;color:#111">
          AgentForensics Incident Report
        </h1>
        <div class="meta">
          Session: <code>{_e(report.session_id)}</code><br>
          Generated: {_e(report.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC"))}<br>
          Chain integrity: {chain_html}
        </div>
      </div>
      <div>
        <span class="verdict-badge" style="background:{verdict_bg}">
          {_e(verdict_label)}
        </span>
      </div>
    </div>

    <div class="stat-grid">
      <div class="stat">
        <div class="value">{_e(report.total_turns)}</div>
        <div class="label">Turns</div>
      </div>
      <div class="stat">
        <div class="value">{_e(len(report.injection_signals))}</div>
        <div class="label">Signals</div>
      </div>
      <div class="stat">
        <div class="value">{_e(len(report.behavior_diffs))}</div>
        <div class="label">Pivots</div>
      </div>
      <div class="stat">
        <div class="value">{_e(f"{report.confidence:.0%}")}</div>
        <div class="label">Confidence</div>
      </div>
    </div>

    <div class="summary-box">{_e(report.summary)}</div>
  </div>

  <div class="nav">{nav_links}</div>

  <!Timeline>
  {section("Conversation Timeline",
      f'<div class="timeline-scroll">{timeline_html}</div>',
      anchor="timeline")}

  <!Evidence>
  {section("Injection Evidence", evidence_html, anchor="evidence")}

  <!Behavior Diff>
  {section("Behavior Diff", diffs_html, anchor="diffs")}

</div>
</body>
</html>"""
