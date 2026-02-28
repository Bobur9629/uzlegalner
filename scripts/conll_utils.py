\
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

@dataclass(frozen=True)
class Sentence:
    tokens: List[str]
    tags: List[str]

def read_conll(path: str) -> List[Sentence]:
    \"\"\"Read CoNLL BIO: token<TAB>tag, blank line separates sentences.\"\"\"
    sents: List[Sentence] = []
    tokens: List[str] = []
    tags: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    sents.append(Sentence(tokens=tokens, tags=tags))
                tokens, tags = [], []
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Invalid CoNLL line (expect token<TAB>tag): {line}")
            tok, tag = parts
            tokens.append(tok)
            tags.append(tag)
    if tokens:
        sents.append(Sentence(tokens=tokens, tags=tags))
    return sents

def bio_to_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
    \"\"\"Convert BIO tags to spans (start,end,label) by token indices.\"\"\"
    spans: List[Tuple[int, int, str]] = []
    start: Optional[int] = None
    cur_label: Optional[str] = None

    def close(i: int):
        nonlocal start, cur_label
        if start is not None and cur_label is not None:
            spans.append((start, i, cur_label))
        start, cur_label = None, None

    for i, tag in enumerate(tags):
        if tag == "O":
            close(i)
            continue
        if "-" not in tag:
            raise ValueError(f"Invalid BIO tag: {tag}")
        prefix, label = tag.split("-", 1)
        if prefix == "B":
            close(i)
            start, cur_label = i, label
        elif prefix == "I":
            # BIO repair: inconsistent I becomes B
            if start is None or cur_label != label:
                close(i)
                start, cur_label = i, label
        else:
            raise ValueError(f"Invalid BIO prefix: {tag}")
    close(len(tags))
    return spans

def micro_prf(
    gold_spans: List[List[Tuple[int, int, str]]],
    pred_spans: List[List[Tuple[int, int, str]]],
) -> Dict[str, float]:
    \"\"\"Span-level exact match micro P/R/F1.\"\"\"
    tp = fp = fn = 0
    for g, p in zip(gold_spans, pred_spans):
        gset, pset = set(g), set(p)
        tp += len(gset & pset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
