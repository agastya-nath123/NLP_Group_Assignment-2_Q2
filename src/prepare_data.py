"""Build every preprocessed cache once (both training scripts also call this lazily).

    python -m src.prepare_data                       # fastText embeddings, RoBERTa len 256
    python -m src.prepare_data --embeddings glove
"""
from __future__ import annotations

import argparse

from .data import load_reviews, prepare_classic, prepare_roberta
from .utils import log


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--embeddings", default="fasttext", help="fasttext | glove | path to vectors file")
    p.add_argument("--min_freq", type=int, default=3)
    p.add_argument("--classic_max_len", type=int, default=200)
    p.add_argument("--tokenizer", default="spacy", choices=["spacy", "regex"])
    p.add_argument("--model_name", default="roberta-base")
    p.add_argument("--roberta_max_len", type=int, nargs="+", default=[256])
    p.add_argument("--no_titles", action="store_true", help="do not try to recover review titles")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    _, stats = load_reviews(a.data_dir, attach_titles=not a.no_titles, force=a.force)
    log(f"merged dataset: {stats}")
    cs = prepare_classic(a.data_dir, embeddings=a.embeddings, min_freq=a.min_freq,
                         max_len=a.classic_max_len, tokenizer=a.tokenizer, force=a.force)
    log(f"classic pipeline: vocab {cs['vocab_size']:,}, coverage {cs['embeddings']['type_coverage']:.1%}")
    for n in a.roberta_max_len:
        prepare_roberta(a.data_dir, a.model_name, n, force=a.force)


if __name__ == "__main__":
    main()
