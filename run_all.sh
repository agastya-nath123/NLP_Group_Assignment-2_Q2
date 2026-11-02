#!/usr/bin/env bash
# Full pipeline from the command line (same code the notebooks call).
# Usage:  bash run_all.sh            (all seeds, main model + ablation)
set -euo pipefail
SEEDS="${SEEDS:-42 13}"

python -m src.prepare_data --embeddings fasttext --roberta_max_len 256

for s in $SEEDS; do python -m src.train_baseline --seed "$s"; done
for s in $SEEDS; do python -m src.train_roberta  --seed "$s" --pooling mean;  done   # main model
for s in $SEEDS; do python -m src.train_roberta  --seed "$s" --pooling first; done   # ablation

python -m src.analysis       # -> results/summary.md + all figures
