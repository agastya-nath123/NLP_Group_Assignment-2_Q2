"""Train + evaluate the BiGRU + additive-attention baseline.

    python -m src.train_baseline --seed 42
    python -m src.train_baseline --seed 13

Adam (lr 1e-3), gradient clipping 1.0, embeddings frozen for the first epoch then
unfrozen, early stopping on validation macro F1. Outputs go to outputs/bigru_attn_seed<S>/.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ClassicSplit, load_reviews, prepare_classic
from .engine import finalize_run, fit
from .models import BiGRUAttentionClassifier
from .utils import (LengthBucketBatchSampler, get_device, load_json, log, resolve_amp_dtype,
                    save_json, set_seed)


@dataclass
class BaselineConfig:
    seed: int = 42
    data_dir: str = "data"
    out_dir: str = "outputs"
    run_name: str | None = None
    # preprocessing
    embeddings: str = "fasttext"      # fasttext | glove | path to a .vec/.txt file
    tokenizer: str = "spacy"
    min_freq: int = 3
    max_len: int = 200
    # model
    hidden_size: int = 256
    num_layers: int = 2
    attn_dim: int = 256
    gru_dropout: float = 0.3
    emb_dropout: float = 0.25
    dropout: float = 0.5
    # optimisation
    batch_size: int = 128
    eval_batch_size: int = 512
    lr: float = 1e-3
    emb_lr: float | None = None       # lr for the embeddings once unfrozen (None = lr)
    clip_norm: float = 1.0
    max_epochs: int = 10
    freeze_epochs: int = 1
    patience: int = 2                 # evaluations without val macro-F1 improvement
    evals_per_epoch: int = 1
    precision: str = "auto"           # auto | bf16 | fp16 | fp32
    num_workers: int = 0
    max_train: int | None = None      # use a fixed random subset of the training set
    overwrite: bool = False


def run_baseline(cfg: BaselineConfig) -> dict:
    run_dir = Path(cfg.out_dir) / (cfg.run_name or f"bigru_attn_seed{cfg.seed}")
    if (run_dir / "metrics.json").exists() and not cfg.overwrite:
        log(f"{run_dir} already finished - skipping (use overwrite=True to retrain)")
        return load_json(run_dir / "metrics.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train.log"
    save_json(asdict(cfg), run_dir / "config.json")

    set_seed(cfg.seed)
    device, amp_dtype = get_device(), resolve_amp_dtype(cfg.precision)
    log(f"device={device} amp={amp_dtype} run={run_dir.name}", log_file)

    prep = prepare_classic(cfg.data_dir, embeddings=cfg.embeddings, min_freq=cfg.min_freq,
                           max_len=cfg.max_len, tokenizer=cfg.tokenizer)
    merged, _ = load_reviews(cfg.data_dir)
    ids_by_split = {s: merged[s]["id"] for s in ("validation", "test")}

    train = ClassicSplit(cfg.data_dir, "train", max_items=cfg.max_train)
    val = ClassicSplit(cfg.data_dir, "validation")
    test = ClassicSplit(cfg.data_dir, "test")
    pin = device.type == "cuda"

    def loader(ds, bs, shuffle):
        sampler = LengthBucketBatchSampler(ds.batch_lengths(), bs, shuffle=shuffle, seed=cfg.seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=ds.collate,
                          num_workers=cfg.num_workers, pin_memory=pin)

    train_loader = loader(train, cfg.batch_size, True)
    val_loader = loader(val, cfg.eval_batch_size, False)
    test_loader = loader(test, cfg.eval_batch_size, False)

    emb = np.load(Path(cfg.data_dir) / "classic" / "embedding_matrix.npy")
    model = BiGRUAttentionClassifier(
        emb, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, attn_dim=cfg.attn_dim,
        gru_dropout=cfg.gru_dropout, emb_dropout=cfg.emb_dropout, dropout=cfg.dropout).to(device)

    emb_params = [model.embedding.weight]
    other = [p for n, p in model.named_parameters() if not n.startswith("embedding.")]
    optimizer = torch.optim.Adam([{"params": other, "lr": cfg.lr},
                                  {"params": emb_params, "lr": cfg.emb_lr or cfg.lr}])

    def on_epoch_start(epoch: int) -> None:
        trainable = epoch >= cfg.freeze_epochs
        if model.embedding.weight.requires_grad != trainable or epoch == 0:
            model.set_embedding_trainable(trainable)
            log(f"epoch {epoch + 1}: embeddings {'TRAINABLE' if trainable else 'FROZEN'}", log_file)

    def forward_fn(m, b):
        return m(b["input_ids"], b["lengths"])

    log(f"train={len(train):,} val={len(val):,} test={len(test):,} | vocab={prep['vocab_size']:,}",
        log_file)
    history, best, train_time = fit(
        model, train_loader=train_loader, val_loader=val_loader, forward_fn=forward_fn,
        optimizer=optimizer, device=device, amp_dtype=amp_dtype, ckpt_path=run_dir / "best.pt",
        max_epochs=cfg.max_epochs, evals_per_epoch=cfg.evals_per_epoch, patience=cfg.patience,
        clip_norm=cfg.clip_norm, on_epoch_start=on_epoch_start, log_file=log_file)
    model.set_embedding_trainable(True)  # so the parameter count reflects the fine-tuned model
    return finalize_run(run_dir, model, forward_fn, val_loader=val_loader, test_loader=test_loader,
                        device=device, amp_dtype=amp_dtype, history=history, best_entry=best,
                        train_time_s=train_time, ids_by_split=ids_by_split,
                        extra={"model": "BiGRU+AdditiveAttention", "train_size": len(train)},
                        log_file=log_file)


def _parse() -> BaselineConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(BaselineConfig):
        if f.type in ("bool", bool):
            p.add_argument(f"--{f.name}", action="store_true")
        else:
            caster = {"int": int, "float": float, "int | None": int, "float | None": float}.get(str(f.type), str)
            p.add_argument(f"--{f.name}", type=caster, default=f.default)
    return BaselineConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    run_baseline(_parse())
