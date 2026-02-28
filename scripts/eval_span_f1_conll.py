#!/usr/bin/env python3
"""Span-level exact-match evaluation: micro P/R/F1 + per-label breakdown.

Usage (run from project root):
    python scripts/eval_span_f1_conll.py \
        --gold data/test.conll \
        --pred outputs/xlmr_baseline/test_predictions.conll

FIX BUG 8: sys.path insert so the script works when run from any directory,
not just the project root.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# FIX BUG 8: make sure project root is on sys.path so `scripts.conll_utils` resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.conll_utils import bio_to_spans, micro_prf, per_label_prf, read_conll


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Span-level exact-match micro P/R/F1 on CoNLL BIO files."
    )
    ap.add_argument("--gold", required=True, help="Gold CoNLL file (ground truth).")
    # FIX BUG 9: README had --pred pointing to same file as --gold (F1 always 1.0).
    # The help text now makes clear this must be the MODEL PREDICTION file.
    ap.add_argument(
        "--pred",
        required=True,
        help="Prediction CoNLL file (model output, NOT the gold file).",
    )
    ap.add_argument(
        "--per_label",
        action="store_true",
        default=True,
        help="Print per-label F1 breakdown (default: True).",
    )
    args = ap.parse_args()

    gold_sents = read_conll(args.gold)
    pred_sents = read_conll(args.pred)

    if len(gold_sents) != len(pred_sents):
        raise ValueError(
            f"Sentence count mismatch: gold={len(gold_sents)} pred={len(pred_sents)}"
        )

    gold_spans = [bio_to_spans(s.tags) for s in gold_sents]
    pred_spans = [bio_to_spans(s.tags) for s in pred_sents]

    # ── Overall micro metrics ──────────────────────────────────────────────
    m = micro_prf(gold_spans, pred_spans)
    print("=" * 48)
    print(f"  Micro Precision : {m['precision']:.4f}")
    print(f"  Micro Recall    : {m['recall']:.4f}")
    print(f"  Micro F1        : {m['f1']:.4f}")
    print(f"  TP / FP / FN    : {m['tp']} / {m['fp']} / {m['fn']}")
    print("=" * 48)

    # ── Per-label breakdown ────────────────────────────────────────────────
    if args.per_label:
        per_lab = per_label_prf(gold_spans, pred_spans)
        if per_lab:
            print(f"\n{'Label':<12} {'P':>7} {'R':>7} {'F1':>7}  {'TP':>5} {'FP':>5} {'FN':>5}")
            print("-" * 56)
            for label, lm in per_lab.items():
                print(
                    f"{label:<12} {lm['precision']:>7.4f} {lm['recall']:>7.4f} "
                    f"{lm['f1']:>7.4f}  {lm['tp']:>5} {lm['fp']:>5} {lm['fn']:>5}"
                )
        else:
            print("\n(No entities found in gold spans.)")


if __name__ == "__main__":
    main()
