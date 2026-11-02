"""Training / evaluation loop shared by both models.

* mixed precision (bf16/fp16 autocast, GradScaler for fp16)
* gradient accumulation and gradient clipping (max norm 1.0)
* validation `evals_per_epoch` times per epoch; the history (train loss / macro F1 over
  the last window, val loss / macro F1 / acc / MAE) is what the curves are drawn from
* early stopping on validation macro F1; the best weights are written to best.pt
* finalize_run(): reload best.pt, predict validation + test ONCE, save predictions/metrics
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .metrics import compute_metrics, macro_f1
from .utils import autocast, count_parameters, log, make_grad_scaler, save_json

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def move_to(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, forward_fn, device, amp_dtype) -> dict:
    model.eval()
    idx, logits, labels = [], [], []
    for batch in loader:
        batch = move_to(batch, device)
        with autocast(amp_dtype):
            out = forward_fn(model, batch)
        idx.append(batch["idx"].cpu())
        logits.append(out.float().cpu())
        labels.append(batch["labels"].cpu())
    idx = torch.cat(idx).numpy()
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    loss = F.cross_entropy(logits, labels).item()
    order = np.argsort(idx)
    probs = torch.softmax(logits, dim=-1).numpy()
    return {"idx": idx[order], "labels": labels.numpy()[order],
            "preds": logits.argmax(-1).numpy()[order], "probs": probs[order], "loss": loss}


def fit(model, *, train_loader, val_loader, forward_fn, optimizer, device, amp_dtype,
        ckpt_path, max_epochs: int, scheduler=None, evals_per_epoch: int = 1, patience: int = 2,
        grad_accum: int = 1, clip_norm: float = 1.0, on_epoch_start=None, log_file=None,
        progress: bool = True):
    scaler = make_grad_scaler(amp_dtype)
    n_iters = len(train_loader)
    eval_points = sorted({max(1, round(k * n_iters / evals_per_epoch))
                          for k in range(1, evals_per_epoch + 1)})
    history, best_entry = [], None
    best, bad, opt_steps, stop = -math.inf, 0, 0, False
    t0 = time.time()

    for epoch in range(max_epochs):
        if on_epoch_start is not None:
            on_epoch_start(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        win_loss = torch.zeros((), device=device)
        win_n, win_preds, win_labels = 0, [], []
        it_loader = train_loader
        if progress and tqdm is not None:
            it_loader = tqdm(train_loader, desc=f"epoch {epoch + 1}", mininterval=5, leave=False)

        for it, batch in enumerate(it_loader, start=1):
            batch = move_to(batch, device)
            with autocast(amp_dtype):
                logits = forward_fn(model, batch)
            loss = F.cross_entropy(logits.float(), batch["labels"])
            scaler.scale(loss / grad_accum).backward()

            bs = batch["labels"].size(0)
            win_loss += loss.detach() * bs
            win_n += bs
            win_preds.append(logits.detach().argmax(-1).cpu())
            win_labels.append(batch["labels"].cpu())

            if it % grad_accum == 0 or it == n_iters:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                opt_steps += 1

            if it in eval_points:
                val = predict(model, val_loader, forward_fn, device, amp_dtype)
                vm = compute_metrics(val["labels"], val["preds"])
                tp, tl = torch.cat(win_preds).numpy(), torch.cat(win_labels).numpy()
                entry = {
                    "epoch": round(epoch + it / n_iters, 4), "step": opt_steps,
                    "train_loss": float(win_loss.item() / win_n),
                    "train_acc": float((tp == tl).mean()), "train_macro_f1": macro_f1(tl, tp),
                    "val_loss": val["loss"], "val_acc": vm["accuracy"],
                    "val_macro_f1": vm["macro_f1"], "val_mae": vm["mae"],
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_min": round((time.time() - t0) / 60, 2),
                }
                history.append(entry)
                win_loss.zero_()
                win_n, win_preds, win_labels = 0, [], []

                improved = entry["val_macro_f1"] > best + 1e-5
                if improved:
                    best, best_entry, bad = entry["val_macro_f1"], entry, 0
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    bad += 1
                log(f"epoch {entry['epoch']:.2f} | train loss {entry['train_loss']:.4f} "
                    f"F1 {entry['train_macro_f1']:.4f} | val loss {entry['val_loss']:.4f} "
                    f"F1 {entry['val_macro_f1']:.4f} acc {entry['val_acc']:.4f} "
                    f"MAE {entry['val_mae']:.4f} | {'*best*' if improved else f'no gain {bad}/{patience}'}",
                    log_file)
                model.train()
                if bad >= patience:
                    stop = True
                    break
        if stop:
            log(f"early stopping: no val macro-F1 gain for {patience} evaluations", log_file)
            break
    return history, best_entry, time.time() - t0


def save_predictions(pred: dict, ids, path) -> None:
    df = pd.DataFrame({"idx": pred["idx"], "id": [ids[i] for i in pred["idx"]],
                       "label": pred["labels"], "pred": pred["preds"],
                       "stars_true": pred["labels"] + 1, "stars_pred": pred["preds"] + 1})
    for k in range(pred["probs"].shape[1]):
        df[f"p{k + 1}"] = pred["probs"][:, k].round(5)
    df.to_csv(path, index=False)


def finalize_run(run_dir, model, forward_fn, *, val_loader, test_loader, device, amp_dtype,
                 history, best_entry, train_time_s, ids_by_split, extra=None, log_file=None) -> dict:
    """Reload the best checkpoint and evaluate it on validation and (once) on test."""
    run_dir = Path(run_dir)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    results = {}
    for split, loader in (("validation", val_loader), ("test", test_loader)):
        pred = predict(model, loader, forward_fn, device, amp_dtype)
        save_predictions(pred, ids_by_split[split], run_dir / f"{split}_predictions.csv")
        results[split] = compute_metrics(pred["labels"], pred["preds"])
        results[split]["loss"] = pred["loss"]
    metrics = {
        "val": results["validation"], "test": results["test"], "best_eval": best_entry,
        "train_time_min": round(train_time_s / 60, 2),
        "params_total": count_parameters(model),
        "params_trainable": count_parameters(model, trainable_only=True),
        **(extra or {}),
    }
    save_json(history, run_dir / "history.json")
    save_json(metrics, run_dir / "metrics.json")
    t = results["test"]
    log(f"TEST  acc {t['accuracy']:.4f} | macro-F1 {t['macro_f1']:.4f} | MAE {t['mae']:.4f}  "
        f"-> {run_dir}", log_file)
    return metrics
