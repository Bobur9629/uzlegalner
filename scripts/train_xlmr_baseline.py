\
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from seqeval.metrics import f1_score
from scripts.conll_utils import read_conll

LABELS = [
  "O",
  "B-PER","I-PER",
  "B-ORG","I-ORG",
  "B-LOC","I-LOC",
  "B-POSITION","I-POSITION",
  "B-DATE","I-DATE",
  "B-MONEY","I-MONEY",
  "B-DOCNO","I-DOCNO",
]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for l,i in label2id.items()}

def conll_to_ds(path: str) -> Dataset:
    sents = read_conll(path)
    rows = [{"tokens": s.tokens, "ner_tags": [label2id[t] for t in s.tags]} for s in sents]
    return Dataset.from_list(rows)

def tokenize_and_align(examples, tokenizer):
    tok = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True)
    labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tok.word_ids(batch_index=i)
        prev = None
        lab_ids = []
        for w in word_ids:
            if w is None:
                lab_ids.append(-100)
            elif w != prev:
                lab_ids.append(word_labels[w])
            else:
                # subword continuation: turn B-X into I-X
                orig = id2label[word_labels[w]]
                if orig.startswith("B-"):
                    orig = "I-" + orig[2:]
                lab_ids.append(label2id.get(orig, word_labels[w]))
            prev = w
        labels.append(lab_ids)
    tok["labels"] = labels
    return tok

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    true_preds, true_labels = [], []
    for pred, lab in zip(preds, labels):
        sp, sl = [], []
        for p_i, l_i in zip(pred, lab):
            if l_i == -100:
                continue
            sp.append(id2label[int(p_i)])
            sl.append(id2label[int(l_i)])
        true_preds.append(sp)
        true_labels.append(sl)
    return {"micro_f1": f1_score(true_labels, true_preds)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--model_name", default="xlm-roberta-base")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    train_ds = conll_to_ds(args.train)
    dev_ds = conll_to_ds(args.dev)
    test_ds = conll_to_ds(args.test)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_tok = train_ds.map(lambda x: tokenize_and_align(x, tokenizer), batched=True)
    dev_tok = dev_ds.map(lambda x: tokenize_and_align(x, tokenizer), batched=True)
    test_tok = test_ds.map(lambda x: tokenize_and_align(x, tokenizer), batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    targs = TrainingArguments(
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
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("TEST:", trainer.evaluate(test_tok))

if __name__ == "__main__":
    main()
