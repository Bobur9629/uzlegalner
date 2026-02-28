#!/usr/bin/env python3
"""Train XLM-R baseline NER model on Uzbek legal CoNLL data.

Usage (run from project root):
    python scripts/train_xlmr_baseline.py \
        --train data/train.conll \
        --dev   data/dev.conll \
        --test  data/test.conll \
        --output_dir outputs/xlmr_baseline

FIX BUG 8: sys.path insert so script works from any directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# FIX BUG 8: resolve imports regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from scripts.conll_utils import read_conll

# ── Label schema (fixed; must match CoNLL files) ───────────────────────────
LABELS = [
    "O",
    "B-PER",  "I-PER",
    "B-ORG",  "I-ORG",
    "B-LOC",  "I-LOC",
    "B-POSITION", "I-POSITION",
    "B-DATE", "I-DATE",
    "B-MONEY","I-MONEY",
    "B-DOCNO","I-DOCNO",
]
label2id: dict[str, int] = {l: i for i, l in enumerate(LABELS)}
id2label:  dict[int, str] = {i: l for l, i in label2id.items()}


# ── Dataset helpers ────────────────────────────────────────────────────────

def conll_to_ds(path: str) -> Dataset:
    sents = read_conll(path)
    rows = []
    for s in sents:
        # FIX BUG 4: use .get() with fallback to avoid KeyError on unknown tags
        tag_ids = []
        for t in s.tags:
            if t not in label2id:
                raise ValueError(
                    f"Unknown tag {t!r} in {path}. "
                    f"Valid tags: {list(label2id.keys())}"
                )
            tag_ids.append(label2id[t])
        rows.append({"tokens": list(s.tokens), "ner_tags": tag_ids})
    return Dataset.from_list(rows)


def tokenize_and_align(examples: dict, tokenizer) -> dict:
    tok = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=512,
    )
    all_labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tok.word_ids(batch_index=i)
        prev_word_id = None
        lab_ids = []
        for word_id in word_ids:
            if word_id is None:
                lab_ids.append(-100)
            elif word_id != prev_word_id:
                # first subtoken of a word: use the original label
                lab_ids.append(word_labels[word_id])
            else:
                # continuation subtoken: B-X → I-X, I-X stays I-X
                orig_label = id2label[word_labels[word_id]]
                if orig_label.startswith("B-"):
                    orig_label = "I-" + orig_label[2:]
                lab_ids.append(label2id[orig_label])
            prev_word_id = word_id
        all_labels.append(lab_ids)
    tok["labels"] = all_labels
    return tok


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred) -> dict:
    # FIX BUG 3: renamed parameter from `p` to `eval_pred` to avoid collision
    # with the loop variable `p` that existed in the original code.
    raw_preds = np.argmax(eval_pred.predictions, axis=-1)
    raw_labels = eval_pred.label_ids

    true_preds: list[list[str]] = []
    true_labels: list[list[str]] = []
    for pred_seq, label_seq in zip(raw_preds, raw_labels):
        seq_pred, seq_label = [], []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            seq_pred.append(id2label[int(pred_id)])
            seq_label.append(id2label[int(label_id)])
        true_preds.append(seq_pred)
        true_labels.append(seq_label)

    return {"micro_f1": f1_score(true_labels, true_preds)}


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",      required=True)
    ap.add_argument("--dev",        required=True)
    ap.add_argument("--test",       required=True)
    ap.add_argument("--model_name", default="xlm-roberta-base")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs",     type=int,   default=5)
    ap.add_argument("--lr",         type=float, default=2e-5)
    ap.add_argument("--batch",      type=int,   default=8)
    # FIX BUG 7: expose seed as argument (default 42 for reproducibility)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds  = conll_to_ds(args.train)
    dev_ds    = conll_to_ds(args.dev)
    test_ds   = conll_to_ds(args.test)

    fn = lambda x: tokenize_and_align(x, tokenizer)
    train_tok = train_ds.map(fn, batched=True, remove_columns=train_ds.column_names)
    dev_tok   = dev_ds.map(fn,   batched=True, remove_columns=dev_ds.column_names)
    test_tok  = test_ds.map(fn,  batched=True, remove_columns=test_ds.column_names)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_steps=50,
        # FIX BUG 7: set seed for reproducibility
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # FIX BUG 6: explicitly save the best model to output_dir root
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Best model saved to: {args.output_dir}")

    # FIX BUG 5: use trainer.predict() for test set so metric keys are correct
    test_output = trainer.predict(test_tok)
    test_metrics = compute_metrics(test_output)
    print("\n=== TEST SET RESULTS ===")
    print(f"  micro_f1 : {test_metrics['micro_f1']:.4f}")

    # Bonus: seqeval classification report on test
    raw_preds  = np.argmax(test_output.predictions, axis=-1)
    raw_labels = test_output.label_ids
    true_preds, true_labels = [], []
    for pred_seq, label_seq in zip(raw_preds, raw_labels):
        sp, sl = [], []
        for pid, lid in zip(pred_seq, label_seq):
            if lid == -100:
                continue
            sp.append(id2label[int(pid)])
            sl.append(id2label[int(lid)])
        true_preds.append(sp)
        true_labels.append(sl)
    print("\n=== PER-LABEL REPORT ===")
    print(classification_report(true_labels, true_preds))


if __name__ == "__main__":
    main()
