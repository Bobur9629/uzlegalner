\
#!/usr/bin/env python3
from __future__ import annotations
import argparse
from scripts.conll_utils import read_conll, bio_to_spans, micro_prf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    gold = read_conll(args.gold)
    pred = read_conll(args.pred)
    if len(gold) != len(pred):
        raise ValueError(f"Sentence count mismatch: gold={len(gold)} pred={len(pred)}")

    gold_spans = [bio_to_spans(s.tags) for s in gold]
    pred_spans = [bio_to_spans(s.tags) for s in pred]
    m = micro_prf(gold_spans, pred_spans)

    print(f"Micro Precision: {m['precision']:.4f}")
    print(f"Micro Recall:    {m['recall']:.4f}")
    print(f"Micro F1:        {m['f1']:.4f}")
    print(f"TP/FP/FN:        {m['tp']}/{m['fp']}/{m['fn']}")

if __name__ == "__main__":
    main()
