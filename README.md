# Q2 — Fine-tuning RoBERTa vs a BiGRU + attention baseline for 5-class rating prediction

Dataset: `SetFit/amazon_reviews_multi_en` (200,000 train / 5,000 validation / 5,000 test,
1–5 stars, perfectly balanced). Two models:

| | Baseline | Transformer |
|---|---|---|
| Input | cleaned, lowercased, spaCy tokens, max 200 | byte-level BPE (`RobertaTokenizerFast`), max 256, dynamic padding |
| Encoder | fastText 300-d embeddings (frozen in epoch 1, then trainable) → 2-layer BiGRU (256/direction) | `roberta-base`, fine-tuned end-to-end |
| Pooling | additive attention pooling | masked **mean** pooling over non-padded tokens (ablation: `<s>` token) |
| Head | Dropout 0.5 → Linear(512→5) | Dropout 0.1 → Linear(768→5) |
| Optimiser | Adam 1e-3, grad clip 1.0, early stop on val macro-F1 | AdamW 2e-5, wd 0.01, 6 % linear warm-up + linear decay, fp16/bf16, grad clip 1.0, early stop on val macro-F1 |

Both are trained with cross-entropy for **2 seeds (42, 13)**; results are reported as mean ± std.
The test set is evaluated exactly once per run, using the checkpoint with the best validation macro-F1.

## Folder layout

```
q2_rating_prediction/
├── notebooks/
│   ├── 01_preprocessing.ipynb              data loading, both preprocessing pipelines (deliverable 1)
│   ├── 02_baseline_bigru_attention.ipynb   baseline training + evaluation (deliverable 2)
│   ├── 03_roberta_finetune_ablation.ipynb  RoBERTa fine-tuning, ablation, token importance (deliverable 3)
│   └── 04_comparison_and_analysis.ipynb    comparison tables, confusion matrices, qualitative analysis
├── src/
│   ├── data.py            loading + title recovery, classic pipeline, RoBERTa tokenization, datasets
│   ├── models.py          BiGRUAttentionClassifier, RobertaRatingClassifier (mean / first pooling)
│   ├── engine.py          shared training loop (AMP, grad accumulation, clipping, early stopping)
│   ├── train_baseline.py  baseline run (also a CLI)
│   ├── train_roberta.py   RoBERTa run (also a CLI)
│   ├── metrics.py         accuracy, macro / per-class F1, MAE, confusion matrices, adjacent vs distant errors
│   ├── analysis.py        aggregates all runs → results/ (tables, plots, qualitative examples, summary.md)
│   ├── explain.py         BiGRU attention, RoBERTa attention and word-occlusion heatmaps
│   ├── prepare_data.py    builds all caches up front (optional, the trainers do it lazily)
│   └── utils.py           seeds, mixed precision, length-bucketed batch sampler
├── run_all.sh             the whole pipeline from the command line
└── requirements.txt
```

Created while running: `data/` (caches + embeddings), `outputs/<run>/` (one folder per model × seed:
`config.json`, `train.log`, `history.json`, `best.pt`, `metrics.json`, `validation_predictions.csv`,
`test_predictions.csv`) and `results/` (everything for the report).

## Important: review titles

The assignment asks for **title + body**, but the SetFit copy of the dataset only contains
`id, text, label, label_text`, where `text` is the review **body**. `load_reviews()` therefore
recovers each title from the MTEB copy of the same corpus (`mteb/AmazonReviewsClassification`,
which stores `title\nbody`) by matching the body text, and builds `text = title + ". " + body`.
The share of reviews whose title was found is printed in notebook 01 and saved in
`data/merged/merge_stats.json` — **report it**. If that copy cannot be downloaded the code
continues with the body only and says so (you can also force this with `--no_titles`).

## What to run

You need a GPU (a free Colab or Kaggle T4 is enough) and internet access (the dataset,
`roberta-base` and the fastText vectors are downloaded on first use).

### Option A — notebooks (recommended for submission)

1. Upload the project folder (or the zip) and open it from the project root.
   * **Colab:** Runtime → Change runtime type → T4 GPU. Put the folder on Google Drive so
     `outputs/` survives a disconnect:
     ```python
     from google.colab import drive; drive.mount('/content/drive')
     %cd /content/drive/MyDrive/q2_rating_prediction
     !pip install -q -r requirements.txt
     ```
   * **Kaggle:** add the zip as a dataset, copy it to `/kaggle/working`, turn on *Internet* and a
     GPU in the notebook settings. "Save Version → Save & Run All" runs it in the background.
2. Run, in order: `01_preprocessing` → `02_baseline_bigru_attention` → `03_roberta_finetune_ablation`
   → `04_comparison_and_analysis`.

Finished runs are skipped automatically, so after a disconnect just re-run the same cell.

### Option B — command line

```bash
pip install -r requirements.txt
python -m src.prepare_data                               # dataset + titles, spaCy, fastText, RoBERTa tokens
python -m src.train_baseline --seed 42
python -m src.train_baseline --seed 13
python -m src.train_roberta  --seed 42 --pooling mean    # main model
python -m src.train_roberta  --seed 13 --pooling mean
python -m src.train_roberta  --seed 42 --pooling first   # ablation
python -m src.train_roberta  --seed 13 --pooling first
python -m src.analysis                                   # → results/summary.md and all figures
```
or simply `bash run_all.sh`. Every hyper-parameter is a flag (`python -m src.train_roberta -h`).
If you go this way, still run the notebooks afterwards to have them with outputs for submission
(training is skipped because the runs already exist).

### Time and memory (rough guide)

On a T4, expect a few minutes per BiGRU epoch and roughly 20–40 minutes per RoBERTa run
(2 epochs over 200k reviews with dynamic padding); an A100/L4 is several times faster.
Four RoBERTa runs (2 seeds × 2 pooling variants) are the bulk of the compute.

* **Out of GPU memory:** `--batch_size 16 --grad_accum 2` (same effective batch of 32).
* **Short on time:** `--max_train 50000` trains on a fixed random 50k subset — **state this in
  the report**, since the assignment specifies the full 200k.
* `precision="auto"` picks bf16 on Ampere+ GPUs and fp16 with loss scaling on T4/P100.

## Deliverables → where they come from

| Deliverable | Source |
|---|---|
| Preprocessing & data-loading notebook (classic + RoBERTa) | `notebooks/01_preprocessing.ipynb` |
| Baseline notebook (training + evaluation) | `notebooks/02_baseline_bigru_attention.ipynb` |
| RoBERTa notebook (fine-tuning, ablation, evaluation) | `notebooks/03_roberta_finetune_ablation.ipynb` (+ `04_…`) |
| Loss and macro-F1 curves | `results/curves_all.png`, `results/curves_<model>.png` |
| Accuracy, macro-F1, per-class F1, MAE (mean ± std) | `results/summary.md`, `results/table_*.csv` |
| Normalised confusion matrices + adjacent vs distant errors | `results/confusion_matrices.png`, `results/error_distance.png`, "Ordinal error profile" table |
| ≥5 "baseline fails / RoBERTa succeeds" + ≥2 opposite examples | `results/qualitative_*.md` (full lists in `results/all_*.csv`) |
| Attention / token-importance visualisation | `results/explain/*.png`, notebook 03 |
| Ablation result | "Ablation" table in `results/summary.md` |
| Report (≤ 6 pages) | write from `results/summary.md` + figures |

## Notes for the report

* **Ablation choice:** mean vs first-token pooling. Class-weighted loss was not chosen because
  every class has exactly 20 % of the data, so the weights would all be 1. Choose the winner on
  the validation columns; the test columns are only reported.
* **Ordinal analysis:** the metrics treat labels 0–4 as stars, so MAE is in stars; "errors that are
  adjacent" is the share of mistakes that are off by exactly one star.
* **Qualitative tags** (negation, contrast, mixed sentiment, sarcasm cue, polarity mismatch,
  long, OOV-heavy, very short) are keyword heuristics meant to help you *find* examples; read each
  example before using it, and write the hypotheses in your own words. `tag_enrichment.csv` shows
  whether a phenomenon is over-represented where only RoBERTa is right.
* **Token importance:** attention weights are not a faithful explanation by themselves; that is why
  the occlusion map (remove a word, measure the probability drop) is shown next to them. RoBERTa's
  last layer typically puts a large share of attention on `<s>`/`</s>`; the plots report that share.
* `train_*` values in the curves are averages over the batches since the previous evaluation
  with dropout active, so they can sit below the validation values early on.

## Troubleshooting

* `RobertaModel LOAD REPORT … UNEXPECTED` when loading `roberta-base`: harmless — the pre-training
  head / pooler weights are not used by this classifier.
* fastText download fails: download `wiki-news-300d-1M.vec.zip` manually, unzip it, and pass
  `embeddings="/path/to/wiki-news-300d-1M.vec"` (or use `embeddings="glove"`).
* Changing preprocessing settings rebuilds `data/classic/` automatically; delete `data/` to rebuild
  everything from scratch.
* Re-train a finished run: delete its `outputs/<run>/` folder or pass `--overwrite`.
