"""Fine-tune roberta-base end-to-end for 5-class rating prediction.

    python -m src.train_roberta --seed 42 --pooling mean      # main model
    python -m src.train_roberta --seed 42 --pooling first     # ablation (<s> pooling)

AdamW (lr 2e-5, weight decay 0.01, no decay on biases/LayerNorm), linear warm-up over
the first 6% of steps then linear decay, fp16/bf16 mixed precision, optional gradient
accumulation, early stopping on validation macro F1.
Outputs go to outputs/roberta_<pooling>_len<max_len>_seed<S>/.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .data import RobertaSplit, load_reviews, load_roberta_tokenizer, prepare_roberta, roberta_cache_dir
from .engine import finalize_run, fit
from .models import RobertaRatingClassifier
from .utils import (LengthBucketBatchSampler, get_device, load_json, log, resolve_amp_dtype,
                    save_json, set_seed)


@dataclass
class RobertaConfig:
    seed: int = 42
    data_dir: str = "data"
    out_dir: str = "outputs"
    run_name: str | None = None
    model_name: str = "roberta-base"
    pooling: str = "mean"             # mean | first
    max_len: int = 256
    dropout: float = 0.1
    batch_size: int = 32              # per optimizer micro-batch
    grad_accum: int = 1               # effective batch = batch_size * grad_accum
    eval_batch_size: int = 128
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    epochs: int = 2
    clip_norm: float = 1.0
    evals_per_epoch: int = 4
    patience: int = 4                 # evaluations without val macro-F1 improvement
    precision: str = "auto"           # auto | bf16 | fp16 | fp32
    num_workers: int = 2
    max_train: int | None = None
    overwrite: bool = False


def linear_warmup_decay(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """lr factor rises linearly 0 -> 1 over warmup_steps, then falls linearly to 0."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return LambdaLR(optimizer, factor)


def param_groups(model, weight_decay: float):
    """Standard practice: no weight decay on biases and LayerNorm weights."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if name.endswith("bias") or "LayerNorm" in name or "layer_norm" in name
         else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def default_run_name(cfg: RobertaConfig) -> str:
    return f"roberta_{cfg.pooling}_len{cfg.max_len}_seed{cfg.seed}"


def run_roberta(cfg: RobertaConfig) -> dict:
    run_dir = Path(cfg.out_dir) / (cfg.run_name or default_run_name(cfg))
    if (run_dir / "metrics.json").exists() and not cfg.overwrite:
        log(f"{run_dir} already finished - skipping (use overwrite=True to retrain)")
        return load_json(run_dir / "metrics.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train.log"
    save_json(asdict(cfg), run_dir / "config.json")

    set_seed(cfg.seed)
    device, amp_dtype = get_device(), resolve_amp_dtype(cfg.precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log(f"device={device} amp={amp_dtype} run={run_dir.name}", log_file)

    tokd = prepare_roberta(cfg.data_dir, cfg.model_name, cfg.max_len)
    tokenizer = load_roberta_tokenizer(cfg.model_name)
    merged, _ = load_reviews(cfg.data_dir)
    ids_by_split = {s: merged[s]["id"] for s in ("validation", "test")}

    train = RobertaSplit(tokd["train"], tokenizer, max_items=cfg.max_train)
    val = RobertaSplit(tokd["validation"], tokenizer)
    test = RobertaSplit(tokd["test"], tokenizer)
    pin = device.type == "cuda"

    def loader(ds, bs, shuffle):
        sampler = LengthBucketBatchSampler(ds.batch_lengths(), bs, shuffle=shuffle, seed=cfg.seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=ds.collate,
                          num_workers=cfg.num_workers, pin_memory=pin,
                          persistent_workers=cfg.num_workers > 0)

    train_loader = loader(train, cfg.batch_size, True)
    val_loader = loader(val, cfg.eval_batch_size, False)
    test_loader = loader(test, cfg.eval_batch_size, False)

    model = RobertaRatingClassifier(cfg.model_name, pooling=cfg.pooling, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr,
                                  betas=(0.9, 0.98), eps=1e-6)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = round(cfg.warmup_ratio * total_steps)
    scheduler = linear_warmup_decay(optimizer, warmup_steps, total_steps)
    log(f"train={len(train):,} | {steps_per_epoch:,} optimizer steps/epoch, total {total_steps:,}, "
        f"warm-up {warmup_steps:,} | effective batch {cfg.batch_size * cfg.grad_accum}", log_file)

    def forward_fn(m, b):
        return m(input_ids=b["input_ids"], attention_mask=b["attention_mask"])

    history, best, train_time = fit(
        model, train_loader=train_loader, val_loader=val_loader, forward_fn=forward_fn,
        optimizer=optimizer, scheduler=scheduler, device=device, amp_dtype=amp_dtype,
        ckpt_path=run_dir / "best.pt", max_epochs=cfg.epochs, evals_per_epoch=cfg.evals_per_epoch,
        patience=cfg.patience, grad_accum=cfg.grad_accum, clip_norm=cfg.clip_norm, log_file=log_file)
    tok_stats = load_json(roberta_cache_dir(cfg.data_dir, cfg.model_name, cfg.max_len) / "tok_stats.json")
    return finalize_run(run_dir, model, forward_fn, val_loader=val_loader, test_loader=test_loader,
                        device=device, amp_dtype=amp_dtype, history=history, best_entry=best,
                        train_time_s=train_time, ids_by_split=ids_by_split,
                        extra={"model": f"RoBERTa ({cfg.pooling} pooling)", "train_size": len(train),
                               "truncated_frac_test": tok_stats["test"]["truncated_frac"]},
                        log_file=log_file)


def _parse() -> RobertaConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(RobertaConfig):
        if f.type in ("bool", bool):
            p.add_argument(f"--{f.name}", action="store_true")
        else:
            caster = {"int": int, "float": float, "int | None": int, "float | None": float}.get(str(f.type), str)
            p.add_argument(f"--{f.name}", type=caster, default=f.default)
    return RobertaConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    run_roberta(_parse())
