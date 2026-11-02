# Q2 results summary (auto-generated)

All numbers: mean ± sample std (ddof=1) over seeds; test set evaluated once per run with the checkpoint that had the best validation macro-F1.

## Main comparison

| Model | Seeds | Test accuracy (%) | Test macro-F1 (%) | Test MAE (stars) | Val macro-F1 (%) | Trainable params (M) | Train time / seed (min) | Best val epoch |
|---|---|---|---|---|---|---|---|---|
| BiGRU + attention | 13, 42 | 63.72 ± 0.31 | 63.65 ± 0.36 | 0.426 ± 0.000 | 63.42 ± 0.22 | 10.0 | 2.3 | 3.00, 2.00 |
| RoBERTa (mean pooling, len 256) | 13, 42 | 66.82 ± 0.23 | 66.70 ± 0.15 | 0.373 ± 0.001 | 66.32 ± 0.05 | 124.1 | 26.9 | 1.50, 2.00 |
| RoBERTa (first pooling, len 256) | 13, 42 | 66.70 ± 0.06 | 66.50 ± 0.19 | 0.375 ± 0.002 | 66.51 ± 0.04 | 124.1 | 27.2 | 1.50, 1.50 |

## Per-class F1 (test)

| Model | F1 1 star (%) | F1 2 stars (%) | F1 3 stars (%) | F1 4 stars (%) | F1 5 stars (%) |
|---|---|---|---|---|---|
| BiGRU + attention | 71.01 ± 0.25 | 54.67 ± 0.01 | 54.74 ± 1.17 | 60.01 ± 0.43 | 77.83 ± 0.06 |
| RoBERTa (mean pooling, len 256) | 74.41 ± 0.01 | 58.40 ± 0.49 | 57.67 ± 0.35 | 63.23 ± 0.52 | 79.81 ± 0.40 |
| RoBERTa (first pooling, len 256) | 74.38 ± 0.10 | 58.22 ± 0.09 | 56.85 ± 0.24 | 62.91 ± 0.81 | 80.15 ± 0.30 |

## Ordinal error profile (test)

| Model | Exact (%) | Off by 1 star (%) | Off by >=2 stars (%) | Errors that are adjacent (%) | 4<->5 confusion rate (%) | 1<->5 confusions (count / 5000) | MAE (stars) |
|---|---|---|---|---|---|---|---|
| BiGRU + attention | 63.72 ± 0.31 | 30.95 ± 0.61 | 5.33 ± 0.30 | 85.30 ± 0.94 | 19.38 ± 0.18 | 10.0 ± 0.0 | 0.426 ± 0.000 |
| RoBERTa (mean pooling, len 256) | 66.82 ± 0.23 | 29.62 ± 0.31 | 3.56 ± 0.08 | 89.27 ± 0.33 | 18.88 ± 0.25 | 6.5 ± 0.7 | 0.373 ± 0.001 |
| RoBERTa (first pooling, len 256) | 66.70 ± 0.06 | 29.70 ± 0.03 | 3.60 ± 0.08 | 89.19 ± 0.24 | 18.75 ± 0.07 | 5.5 ± 0.7 | 0.375 ± 0.002 |

## Ablation: mean pooling vs first-token (<s>) pooling

Decide on the **validation** columns; test columns are reported for completeness.

| Variant | Seeds | Val macro-F1 (%) | Val acc (%) | Val MAE | Test macro-F1 (%) | Test acc (%) | Test MAE |
|---|---|---|---|---|---|---|---|
| RoBERTa (mean pooling, len 256) | 2 | 66.32 ± 0.05 | 66.44 ± 0.14 | 0.378 ± 0.002 | 66.70 ± 0.15 | 66.82 ± 0.23 | 0.373 ± 0.001 |
| RoBERTa (first pooling, len 256) | 2 | 66.51 ± 0.04 | 66.69 ± 0.16 | 0.375 ± 0.003 | 66.50 ± 0.19 | 66.70 ± 0.06 | 0.375 ± 0.002 |

## Where the models disagree (test, seed 42)

|  | RoBERTa correct | RoBERTa wrong |
|---|---|---|
| BiGRU correct | 2871 | 326 |
| BiGRU wrong | 478 | 1325 |

### How often each phenomenon occurs (auto-tagged, heuristic)

| Phenomenon | % of all test | % of BiGRU wrong, RoBERTa right | % of RoBERTa wrong, BiGRU right | % of both wrong |
|---|---|---|---|---|
| negation | 52.8 | 62.3 | 54.9 | 55.4 |
| contrast | 38.8 | 47.3 | 45.7 | 45.4 |
| mixed sentiment | 13.6 | 17.6 | 14.7 | 15.7 |
| sarcasm cue | 5.8 | 5.0 | 3.7 | 4.2 |
| polarity mismatch | 7.6 | 13.8 | 4.6 | 11.8 |
| long (>150 words) | 1.3 | 1.5 | 1.2 | 1.1 |
| OOV-heavy | 4.6 | 5.6 | 5.2 | 5.7 |
| very short (<=8 words) | 7.2 | 3.3 | 4.9 | 4.4 |

Selected examples: `qualitative_baseline_fails_roberta_ok.md`, `qualitative_roberta_fails_baseline_ok.md` (full lists in the `all_*.csv` files).

## Token-importance example (test id `en_0600154`, true 5★)

> I don’t have cravings after taking this at night! I’m horrible with eating late at night, I do it almost every night after dinner and I’m sure it’s part of the reason I’m overweight. But when I take this, the cravings are gone!!

RoBERTa's last layer sends 14% of its attention to <s>/</s>; see `explain/` for the BiGRU attention, RoBERTa attention and occlusion maps.

## Figures

- `curves_all.png`
- `curves_bigru_attention.png`
- `curves_roberta_mean_pooling_len_256.png`
- `curves_roberta_first_pooling_len_256.png`
- `confusion_matrices.png`
- `error_distance.png`
- `per_class_f1.png`
