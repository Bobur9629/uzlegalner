# UzLegalNER: Uzbek Legal Contracts NER (Gazetteer paper)

Dataset DOI (Zenodo): **10.5281/zenodo.18816402**

This repository contains minimal code to:
- train a baseline XLM-R NER model (BIO tagging),
- evaluate span-level exact-match micro P/R/F1 on CoNLL files.

## Dataset
Download from Zenodo DOI: **10.5281/zenodo.18816402**.

Copy the CoNLL files into:
- `data/train.conll`
- `data/dev.conll`
- `data/test.conll`

## Install
```bash
pip install -r requirements.txt
```

## Train baseline (XLM-R)
```bash
python scripts/train_xlmr_baseline.py   --train data/train.conll --dev data/dev.conll --test data/test.conll   --output_dir outputs/xlmr_baseline
```

## Evaluate (span-level exact match)
```bash
python scripts/eval_span_f1_conll.py --gold data/test.conll --pred data/test.conll
```

## Notes
- Labels: PER, ORG, LOC, POSITION, DATE, MONEY, DOCNO
- CoNLL format: TOKEN<TAB>BIO_TAG, blank line separates sentences.

## License
Code: MIT. Dataset: CC BY 4.0 (Zenodo record).
