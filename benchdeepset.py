"""
AgentForensics - deepset/prompt-injections Benchmark
Tests the detector against the deepset/prompt-injections dataset,
which contains both injection and benign prompts — allowing us to
measure detection rate AND false positive rate in one run.

Fields used:
  text  - the prompt text to score
  label - 1 = injection, 0 = benign

Usage:
    pip install datasets
    python benchdeepset.py               # full dataset
    python benchdeepset.py --verbose     # show per-row results
    python benchdeepset.py --ml          # enable ML stage
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


def load_deepset() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Install with: pip install datasets")
        sys.exit(1)

    print("Loading deepset/prompt-injections from Hugging Face...")
    ds = load_dataset("deepset/prompt-injections", split="train")
    rows = list(ds)
    injections = sum(1 for r in rows if r.get("label") == 1)
    benign     = sum(1 for r in rows if r.get("label") == 0)
    print(f"  Total rows : {len(rows)}")
    print(f"  Injections : {injections}")
    print(f"  Benign     : {benign}")
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
    parser = argparse.ArgumentParser(description="AgentForensics deepset benchmark")
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

    rows = load_deepset()

    print(f"\nScoring {len(rows)} samples...\n")

    # Counters
    tp = 0  # injection correctly detected
    fn = 0  # injection missed
    fp = 0  # benign flagged as injection
    tn = 0  # benign correctly passed

    rule_counter: Counter = Counter()
    missed_examples: list = []
    fp_examples:    list  = []

    sep        = "─" * 65
    total_rows = len(rows)
    bar_width  = 30

    if args.verbose:
        print(f"  {'LABEL':<12} {'VERDICT':<12} {'SCORE':>6}  RULES")
        print(f"  {'-----':<12} {'-------':<12} {'-----':>6}  -----")

    for i, row in enumerate(rows, 1):
        text  = (row.get("text") or "").strip()
        label = row.get("label", -1)  # 1=injection, 0=benign

        if not text:
            continue

        score, rules = score_fn(text)
        flagged = score >= 0.25
        rule_counter.update(rules)

        if label == 1:
            if flagged:
                tp += 1
            else:
                fn += 1
                if len(missed_examples) < 5:
                    missed_examples.append({
                        "text":  text[:200],
                        "score": round(score, 4),
                    })
        elif label == 0:
            if flagged:
                fp += 1
                if len(fp_examples) < 5:
                    fp_examples.append({
                        "text":  text[:200],
                        "score": round(score, 4),
                    })
            else:
                tn += 1

        if args.verbose:
            label_str  = "INJECTION" if label == 1 else "BENIGN"
            verdict    = "DETECTED" if flagged else "MISSED" if label == 1 else "CLEAN"
            rules_str  = ",".join(rules) if rules else "-"
            print(f"  {label_str:<12} {verdict:<12} {score:>6.3f}  {rules_str}")
        else:
            done = int(i / total_rows * bar_width)
            bar  = "█" * done + "░" * (bar_width - done)
            pct  = i / total_rows * 100
            print(f"\r  [{bar}] {pct:5.1f}%  {i}/{total_rows}", end="", flush=True)

    if not args.verbose:
        print()  # end progress bar line

    total_inj    = tp + fn
    total_benign = fp + tn
    det_rate     = tp / total_inj    if total_inj    else 0.0
    fp_rate      = fp / total_benign if total_benign else 0.0
    precision    = tp / (tp + fp)    if (tp + fp)    else 0.0
    recall       = det_rate
    f1           = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print()
    print(sep)
    print("  AgentForensics - deepset/prompt-injections Results")
    print(sep)
    print(f"  Dataset          : deepset/prompt-injections")
    print(f"  ML enabled       : {args.ml}")
    print(f"  Total samples    : {total_inj + total_benign}")
    print(f"  Injections       : {total_inj}    Benign: {total_benign}")
    print()
    print(f"  True Positives   : {tp}   (injections caught)")
    print(f"  False Negatives  : {fn}   (injections missed)")
    print(f"  False Positives  : {fp}   (benign flagged)")
    print(f"  True Negatives   : {tn}   (benign passed)")
    print()
    print(f"  Detection rate   : {_bar(det_rate)}  {det_rate*100:.1f}%")
    print(f"  False pos. rate  : {_bar(fp_rate)}  {fp_rate*100:.1f}%")
    print(f"  Precision        : {_bar(precision)}  {precision*100:.1f}%")
    print(f"  F1 Score         : {_bar(f1)}  {f1:.3f}")
    print(sep)

    if missed_examples:
        print(f"\n  Sample missed injections:")
        for ex in missed_examples:
            snippet = ex["text"].replace("\n", " ")[:120]
            print(f"    score={ex['score']:.3f}  \"{snippet}\"")

    if fp_examples:
        print(f"\n  Sample false positives (benign flagged):")
        for ex in fp_examples:
            snippet = ex["text"].replace("\n", " ")[:120]
            print(f"    score={ex['score']:.3f}  \"{snippet}\"")

    print("\n  Top fired rules:")
    for rule, count in rule_counter.most_common(10):
        bar = _bar(count / max(rule_counter.values()), width=15)
        print(f"    {rule:<10}  {bar}  {count}")

    print()

    output = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "dataset":         "deepset/prompt-injections",
        "ml_enabled":      args.ml,
        "total_samples":   total_inj + total_benign,
        "injections":      total_inj,
        "benign":          total_benign,
        "true_positives":  tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives":  tn,
        "detection_rate":  round(det_rate, 4),
        "fp_rate":         round(fp_rate, 4),
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1_score":        round(f1, 4),
        "top_rules":       rule_counter.most_common(15),
        "missed_examples": missed_examples,
        "fp_examples":     fp_examples,
    }
    out_path = Path(__file__).parent / "deepsetresults.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"  Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
