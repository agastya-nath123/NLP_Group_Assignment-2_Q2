"""Small shared helpers: seeding, device / mixed precision, JSON I/O, logging and
a length-bucketed batch sampler (used by both models to cut padding)."""
from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- logging
def log(msg: str, file: str | Path | None = None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if file is not None:
        with open(file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# --------------------------------------------------------------------------- reproducibility
def set_seed(seed: int) -> None:
    """Fix Python, NumPy and PyTorch (CPU + CUDA) seeds.

    cuDNN kernels for GRUs / attention are not bit-exact across runs even with a fixed
    seed, which is one reason the assignment asks for results over several seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- mixed precision
def resolve_amp_dtype(precision: str = "auto"):
    """'auto' -> bf16 on Ampere or newer GPUs (A100, L4, H100), fp16 on older ones
    (T4, P100, V100); 'fp32' or no GPU -> None (no autocast)."""
    if precision == "fp32" or not torch.cuda.is_available():
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "auto":
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    raise ValueError(f"unknown precision '{precision}' (use auto, bf16, fp16 or fp32)")


def autocast(amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def make_grad_scaler(amp_dtype):
    """Loss scaling is only needed for fp16 (bf16 has the same exponent range as fp32)."""
    enabled = amp_dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


# --------------------------------------------------------------------------- I/O
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# --------------------------------------------------------------------------- batching
class LengthBucketBatchSampler(Sampler):
    """Yields lists of dataset positions so that each batch holds sequences of similar
    length, which keeps padding (and wasted compute) small.

    Training: shuffle everything, cut into chunks of `batch_size * bucket_multiplier`,
    sort each chunk by length, split into batches, then shuffle the batch order. This
    keeps randomness between epochs while still grouping similar lengths.
    Evaluation: sort the whole split by length (order does not matter, predictions are
    re-ordered by their original index afterwards)."""

    def __init__(self, lengths, batch_size: int, shuffle: bool, seed: int = 0,
                 bucket_multiplier: int = 50):
        self.lengths = np.asarray(lengths)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_multiplier = bucket_multiplier
        self.epoch = 0
        n = len(self.lengths)
        if shuffle:
            chunk = self.batch_size * bucket_multiplier
            sizes = [min(chunk, n - s) for s in range(0, n, chunk)]
            self._len = sum(math.ceil(s / self.batch_size) for s in sizes)
        else:
            self._len = math.ceil(n / self.batch_size)

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        n = len(self.lengths)
        bs = self.batch_size
        if not self.shuffle:
            order = np.argsort(self.lengths, kind="stable")
            for s in range(0, n, bs):
                yield order[s:s + bs].tolist()
            return
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        perm = rng.permutation(n)
        chunk = bs * self.bucket_multiplier
        batches = []
        for s in range(0, n, chunk):
            c = perm[s:s + chunk]
            c = c[np.argsort(self.lengths[c], kind="stable")]
            batches.extend(c[i:i + bs] for i in range(0, len(c), bs))
        for j in rng.permutation(len(batches)):
            yield batches[j].tolist()
