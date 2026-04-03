"""
AgentForensics - ARPIbench Benchmark
Tests the detector against the ARPIbench dataset (alexcbecker/ARPIbench),
which contains real-world indirect prompt injection payloads across web,
email, and document scenarios - closely matching what AgentForensics detects.

Fields used:
  payload     - the injection text to score (main attack content)
  scenario    - web / email / local
  attack_name - naive / escape / ignore / ...
  objective   - what the attack tries to do

Usage:
    pip install datasets
    python benchtest.py                  # first 2000 rows
    python benchtest.py --all            # full 7560 rows
    python benchtest.py --limit 500
    python benchtest.py --verbose        # show per-row results
    python benchtest.py --ml             # enable ML stage
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))



# Dataset loader


def load_arpi(limit: int | None) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Install with: pip install datasets")
        sys.exit(1)

    print("Loading alexcbecker/ARPIbench from Hugging Face…")
    ds = load_dataset("alexcbecker/ARPIbench", split="train")
    rows = list(ds)
    print(f"  Total rows: {len(rows)}")

    if limit is not None:
        rows = rows[:limit]
        print(f"  Using first {limit} rows (pass --all for full set)")

    return rows



# Scoring


def _score(text: str) -> tuple[float, list[str]]:
    from agentforensics.classifier import _run_heuristics
    from agentforensics.semantic import instruction_boundary_score

    h_score, h_rules, _ = _run_heuristics(text)
    ib_score, ib_rules, _ = instruction_boundary_score(text)
    score = min(1.0, h_score + ib_score)
    return score, h_rules + ib_rules


def _score_ml(text: str) -> tuple[float, list[str]]:
    import agentforensics.classifier as clf
    score, rules = _score(text)
    if clf._ml_available:
        ml_s = clf._ml_score(text)
        score = min(1.0, score + ml_s * 0.4)
    return score, rules


def _bar(value: float, width: int = 25) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)



# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentForensics ARPIbench benchmark")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--all", action="store_true", help="Run full 7560-row dataset")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--ml", action="store_true", help="Enable ML stage")
    args = parser.parse_args()

    if args.ml:
        os.environ.pop("AF_DISABLE_ML", None)
        import agentforensics.classifier as clf
        clf._ml_load_attempted = False
        clf._ml_available = False
        clf._ml_pipeline = None
        clf._try_load_ml()
        score_fn = _score_ml
    else:
        os.environ["AF_DISABLE_ML"] = "true"
        score_fn = _score

    limit = None if args.all else args.limit
    rows  = load_arpi(limit)

    print(f"\nScoring {len(rows)} injection payloads…\n")

    detected = 0
    missed   = 0
    rule_counter: Counter  = Counter()
    scenario_stats: dict   = {}
    attack_stats:   dict   = {}
    missed_examples: list  = []

    sep = "─" * 65

    if args.verbose:
        print(f"  {'VERDICT':<12} {'SCENARIO':<10} {'ATTACK':<16} {'SCORE':>6}  RULES")
        print(f"  {'-------':<12} {'--------':<10} {'------':<16} {'-----':>6}  -----")

    for row in rows:
        payload     = (row.get("payload") or "").strip()
        scenario    = row.get("scenario", "unknown")
        attack_name = row.get("attack_name", "unknown")

        if not payload:
            continue

        score, rules = score_fn(payload)
        is_detected  = score >= 0.25
        rule_counter.update(rules)

        sc = scenario_stats.setdefault(scenario, {"detected": 0, "total": 0})
        sc["total"] += 1
        if is_detected:
            sc["detected"] += 1
            detected += 1
        else:
            missed += 1
            if len(missed_examples) < 5:
                missed_examples.append({
                    "scenario":    scenario,
                    "attack_name": attack_name,
                    "payload":     payload[:200],
                    "score":       round(score, 4),
                })

        at = attack_stats.setdefault(attack_name, {"detected": 0, "total": 0})
        at["total"] += 1
        if is_detected:
            at["detected"] += 1

        if args.verbose:
            verdict   = "DETECTED" if is_detected else "MISSED"
            rules_str = ",".join(rules) if rules else "-"
            print(f"  {verdict:<12} {scenario:<10} {attack_name:<16} {score:>6.3f}  {rules_str}")

    total    = detected + missed
    det_rate = detected / total if total else 0.0

    print()
    print(sep)
    print("  AgentForensics - ARPIbench Results")
    print(sep)
    print(f"  Dataset          : alexcbecker/ARPIbench")
    print(f"  ML enabled       : {args.ml}")
    print(f"  Payloads tested  : {total}")
    print(f"  Detected         : {detected}  ({det_rate*100:.1f}%)")
    print(f"  Missed           : {missed}  ({(1-det_rate)*100:.1f}%)")
    print()
    print(f"  Detection rate   : {_bar(det_rate)}  {det_rate*100:.1f}%")
    print(sep)

    print("\n  By scenario:")
    for sc, st in sorted(scenario_stats.items()):
        r = st["detected"] / st["total"] if st["total"] else 0
        print(f"    {sc:<12}  {_bar(r, 15)}  {st['detected']}/{st['total']}  ({r*100:.1f}%)")

    print("\n  By attack type:")
    for at, st in sorted(attack_stats.items(), key=lambda x: -x[1]["detected"]/max(x[1]["total"],1)):
        r = st["detected"] / st["total"] if st["total"] else 0
        print(f"    {at:<20}  {_bar(r, 15)}  {st['detected']}/{st['total']}  ({r*100:.1f}%)")

    print("\n  Top fired rules:")
    for rule, count in rule_counter.most_common(10):
        bar = _bar(count / max(rule_counter.values()), width=15)
        print(f"    {rule:<10}  {bar}  {count}")

    if missed_examples:
        print(f"\n  Sample missed payloads:")
        for ex in missed_examples:
            snippet = ex["payload"].replace("\n", " ")[:120]
            print(f"    [{ex['scenario']}/{ex['attack_name']}] score={ex['score']:.3f}  \"{snippet}...\"")

    print()

    output = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "dataset":         "alexcbecker/ARPIbench",
        "ml_enabled":      args.ml,
        "payloads_tested": total,
        "detected":        detected,
        "missed":          missed,
        "detection_rate":  round(det_rate, 4),
        "by_scenario":     {sc: round(st["detected"]/st["total"], 4)
                            for sc, st in scenario_stats.items()},
        "by_attack_type":  {at: round(st["detected"]/st["total"], 4)
                            for at, st in attack_stats.items()},
        "top_rules":       rule_counter.most_common(15),
        "missed_examples": missed_examples,
    }
    out_path = Path(__file__).parent / "benchtest_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"  Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
