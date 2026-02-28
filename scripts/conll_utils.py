from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# FIX BUG 1: Use Tuple[str, ...] instead of List[str] so that frozen=True
# actually works (List is mutable and not hashable, Tuple is).
@dataclass(frozen=True)
class Sentence:
    tokens: Tuple[str, ...]
    tags: Tuple[str, ...]


def read_conll(path: str) -> List[Sentence]:
    """Read CoNLL BIO file: TOKEN<TAB>TAG, blank line separates sentences."""
    sents: List[Sentence] = []
    tokens: List[str] = []
    tags: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    sents.append(Sentence(tokens=tuple(tokens), tags=tuple(tags)))
                    tokens, tags = [], []
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{lineno}: invalid CoNLL line "
                    f"(expected TOKEN<TAB>TAG): {line!r}"
                )
            tok, tag = parts
            tokens.append(tok)
            tags.append(tag)

    if tokens:  # last sentence without trailing blank line
        sents.append(Sentence(tokens=tuple(tokens), tags=tuple(tags)))
    return sents


def bio_to_spans(tags: Tuple[str, ...]) -> List[Tuple[int, int, str]]:
    """Convert BIO tags → list of (start, end, label) token-index spans.

    ``end`` is exclusive (Python slice convention).

    FIX BUG 2: Silent BIO repair is kept intentionally for robustness, but
    now documented clearly.  An I-tag whose entity type does not match the
    current open span is treated as a new B-tag (standard CoNLL repair).
    This is intentional — if you want strict validation instead, raise a
    ValueError here.
    """
    spans: List[Tuple[int, int, str]] = []
    start: Optional[int] = None
    cur_label: Optional[str] = None

    def _close(i: int) -> None:
        nonlocal start, cur_label
        if start is not None and cur_label is not None:
            spans.append((start, i, cur_label))
        start, cur_label = None, None

    for i, tag in enumerate(tags):
        if tag == "O":
            _close(i)
            continue
        if "-" not in tag:
            raise ValueError(f"Invalid BIO tag at position {i}: {tag!r}")
        prefix, label = tag.split("-", 1)
        if prefix == "B":
            _close(i)
            start, cur_label = i, label
        elif prefix == "I":
            # BIO repair: I-X after O or I-Y → treat as B-X
            if start is None or cur_label != label:
                _close(i)
                start, cur_label = i, label
            # else: continuation of current span — do nothing
        else:
            raise ValueError(f"Unknown BIO prefix at position {i}: {tag!r}")

    _close(len(tags))
    return spans


def micro_prf(
    gold_spans: List[List[Tuple[int, int, str]]],
    pred_spans: List[List[Tuple[int, int, str]]],
) -> Dict[str, float]:
    """Entity-level exact-match micro Precision / Recall / F1.

    Each span is a (start, end, label) triple; matching is exact.
    """
    tp = fp = fn = 0
    for g, p in zip(gold_spans, pred_spans):
        g_set, p_set = set(g), set(p)
        tp += len(g_set & p_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def per_label_prf(
    gold_spans: List[List[Tuple[int, int, str]]],
    pred_spans: List[List[Tuple[int, int, str]]],
) -> Dict[str, Dict[str, float]]:
    """Per-label exact-match micro P/R/F1 (same logic as micro_prf but split by label)."""
    from collections import defaultdict
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for g_list, p_list in zip(gold_spans, pred_spans):
        g_set, p_set = set(g_list), set(p_list)
        for span in g_set & p_set:
            counts[span[2]]["tp"] += 1
        for span in p_set - g_set:
            counts[span[2]]["fp"] += 1
        for span in g_set - p_set:
            counts[span[2]]["fn"] += 1

    result: Dict[str, Dict[str, float]] = {}
    for label, c in sorted(counts.items()):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        result[label] = {"precision": prec, "recall": rec, "f1": f1,
                         "tp": tp, "fp": fp, "fn": fn}
    return result
