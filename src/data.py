"""Data loading and the two preprocessing pipelines.

1. load_reviews()      : loads SetFit/amazon_reviews_multi_en, recovers review titles and
                         builds  text = title + ". " + body   (cached in data/merged/)
2. prepare_classic()   : clean -> tokenize (spaCy) -> vocabulary (min freq) -> ids
                         (max 200 tokens) -> fastText/GloVe embedding matrix
                         (cached in data/classic/)
3. prepare_roberta()   : RobertaTokenizerFast byte-level BPE with <s> ... </s>,
                         truncation to max_len; padding is done later per batch
                         (dynamic padding)       (cached in data/roberta_<model>_len<N>/)

Row order is identical in all three caches, so a row index ("idx") identifies the same
review everywhere; prediction files use it to line the two models up.
"""
from __future__ import annotations

import html
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .utils import load_json, log, save_json

DATASET_NAME = "SetFit/amazon_reviews_multi_en"
# The SetFit copy keeps only the review body in its "text" column. These copies of the
# same corpus store "title\nbody", so titles are recovered by matching the body text.
TITLE_SOURCES = [("mteb/AmazonReviewsClassification", "en"), ("mteb/amazon_reviews_multi", "en")]
SPLITS = ("train", "validation", "test")
NUM_CLASSES = 5
PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1


# =========================================================================== loading
def merge_title_body(title: str, body: str) -> str:
    """'Great product' + 'Works well.' -> 'Great product. Works well.'"""
    title, body = (title or "").strip(), (body or "").strip()
    if not title:
        return body
    if not body:
        return title
    sep = " " if title[-1] in ".!?" else ". "
    return title + sep + body


def _norm(s: str) -> str:
    return " ".join(str(s).split())


def _title_lookup():
    """body (whitespace-normalised) -> title, built from a copy that stores both."""
    from datasets import load_dataset

    for name, config in TITLE_SOURCES:
        try:
            ds = load_dataset(name, config)
        except Exception as e:  # noqa: BLE001 - any failure just means "try the next one"
            log(f"[data] could not load {name}/{config} for titles ({type(e).__name__}: {e})")
            continue
        lookup: dict[str, str | None] = {}
        for split in ds:
            if "text" not in ds[split].column_names:
                continue
            for text in ds[split]["text"]:
                if "\n" not in text:
                    continue
                title, body = text.split("\n", 1)
                key, title = _norm(body), title.strip()
                if key in lookup and lookup[key] != title:
                    lookup[key] = None  # same body with different titles -> ambiguous, skip
                else:
                    lookup.setdefault(key, title)
        if lookup:
            return f"{name}/{config}", lookup
    return None, {}


def load_reviews(data_dir: str = "data", attach_titles: bool = True, force: bool = False):
    """Returns (DatasetDict with columns id, title, body, text, label[0-4]), stats)."""
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    out = Path(data_dir) / "merged"
    if (out / "merge_stats.json").exists() and not force:
        return load_from_disk(str(out)), load_json(out / "merge_stats.json")

    log(f"[data] downloading {DATASET_NAME}")
    raw = load_dataset(DATASET_NAME)
    cols = raw["train"].column_names
    stats = {"dataset": DATASET_NAME, "original_columns": cols}

    title_col = next((c for c in ("review_title", "title") if c in cols), None)
    body_col = next((c for c in ("review_body", "body") if c in cols), None)
    lookup, source = {}, None
    if title_col and body_col:
        source = f"columns {title_col} + {body_col}"
    elif attach_titles:
        log("[data] 'text' holds only the review body; recovering titles from a copy of the corpus")
        source, lookup = _title_lookup()
        if not lookup:
            log("[data] WARNING: titles could not be recovered; using the review body only")
    stats["title_source"] = source

    merged = {}
    for split in SPLITS:
        d = raw[split]
        ids = d["id"] if "id" in cols else [f"{split}_{i}" for i in range(len(d))]
        if "stars" in cols:
            labels = [int(s) - 1 for s in d["stars"]]
        else:
            labels = [int(x) for x in d["label"]]
        assert set(labels) <= set(range(NUM_CLASSES)), "labels must be 0..4 (1..5 stars)"

        if title_col and body_col:
            titles, bodies = list(d[title_col]), list(d[body_col])
        else:
            titles, bodies = [], []
            for t in d["text"]:
                title, body = (lookup.get(_norm(t)) if lookup else None), t
                if title is None and lookup and "\n" in t:  # text may already be "title\nbody"
                    head, tail = t.split("\n", 1)
                    if lookup.get(_norm(tail)) == head.strip():
                        title, body = head.strip(), tail
                titles.append(title or "")
                bodies.append(body)
        texts = [merge_title_body(a, b) for a, b in zip(titles, bodies)]
        merged[split] = Dataset.from_dict(
            {"id": list(ids), "title": titles, "body": bodies, "text": texts, "label": labels})
        stats[f"{split}_size"] = len(texts)
        stats[f"{split}_title_coverage"] = float(np.mean([bool(t) for t in titles]))
        stats[f"{split}_label_counts"] = np.bincount(labels, minlength=NUM_CLASSES).tolist()

    ds = DatasetDict(merged)
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    save_json(stats, out / "merge_stats.json")
    log(f"[data] saved merged dataset to {out}  (title coverage train="
        f"{stats['train_title_coverage']:.1%})")
    return ds, stats


# =========================================================================== classic pipeline
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Unescape HTML entities, drop HTML tags and URLs, lowercase, squeeze whitespace."""
    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.lower()
    return _WS.sub(" ", text).strip()


class WordTokenizer:
    """spaCy's rule-based English tokenizer (no model download needed). Splits
    "don't" -> "do", "n't", which matches how GloVe/fastText vocabularies store it.
    backend="regex" is a dependency-free fallback."""

    _REGEX = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")

    def __init__(self, backend: str = "spacy"):
        self.backend = backend
        if backend == "spacy":
            import spacy
            self._nlp = spacy.blank("en")
        elif backend != "regex":
            raise ValueError("backend must be 'spacy' or 'regex'")

    def tokenize_many(self, texts, batch_size: int = 2000):
        if self.backend == "spacy":
            return [[t.text for t in doc if not t.is_space]
                    for doc in self._nlp.pipe(texts, batch_size=batch_size)]
        return [self._REGEX.findall(t) for t in texts]


def build_vocab(token_lists, min_freq: int = 3):
    counter = Counter(tok for toks in token_lists for tok in toks)
    words = [w for w, c in counter.most_common() if c >= min_freq]
    return [PAD, UNK] + words, counter


def encode(token_lists, stoi, max_len: int):
    n = len(token_lists)
    ids = np.zeros((n, max_len), dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int32)
    raw_lengths = np.zeros(n, dtype=np.int32)
    n_unk = np.zeros(n, dtype=np.int32)
    for i, toks in enumerate(token_lists):
        seq = [stoi.get(t, UNK_ID) for t in toks[:max_len]] or [UNK_ID]  # never empty
        ids[i, :len(seq)] = seq
        lengths[i] = len(seq)
        raw_lengths[i] = len(toks)
        n_unk[i] = sum(1 for s in seq if s == UNK_ID)
    return ids, lengths, raw_lengths, n_unk


EMBEDDING_SOURCES = {
    "fasttext": {"urls": ["https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"],
                 "member": "wiki-news-300d-1M.vec"},
    "glove": {"urls": ["https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
                       "https://nlp.stanford.edu/data/glove.6B.zip"],
              "member": "glove.6B.300d.txt"},
}


def _download(url: str, dest: Path) -> None:
    state = {"next": 0.0}

    def hook(blocks, block_size, total):
        if total > 0:
            frac = blocks * block_size / total
            if frac >= state["next"]:
                print(f"  {min(frac, 1):6.1%} of {total / 1e6:,.0f} MB", flush=True)
                state["next"] += 0.1

    urllib.request.urlretrieve(url, dest, reporthook=hook)


def download_embeddings(name_or_path: str, emb_dir) -> Path:
    """'fasttext' | 'glove' (downloaded once) or a path to any .vec/.txt vectors file."""
    p = Path(name_or_path)
    if p.exists():
        return p
    if name_or_path not in EMBEDDING_SOURCES:
        raise ValueError(f"unknown embeddings '{name_or_path}': use fasttext, glove or a file path")
    spec = EMBEDDING_SOURCES[name_or_path]
    emb_dir = Path(emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)
    target = emb_dir / spec["member"]
    if target.exists():
        return target
    for url in spec["urls"]:
        zip_path = emb_dir / url.rsplit("/", 1)[-1]
        try:
            log(f"[emb] downloading {url}")
            _download(url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extract(spec["member"], emb_dir)
            zip_path.unlink(missing_ok=True)
            return target
        except Exception as e:  # noqa: BLE001
            log(f"[emb] failed ({type(e).__name__}: {e}); trying next mirror")
            zip_path.unlink(missing_ok=True)
    raise RuntimeError("Could not download embeddings. Download them manually and pass the "
                       "file path, e.g. --embeddings /path/to/wiki-news-300d-1M.vec")


def load_embedding_matrix(path, itos, counter=None, seed: int = 0):
    """Reads a GloVe (.txt) or fastText (.vec, with header) file and keeps only the words in
    our vocabulary. Cased vectors (fastText) are folded to lowercase; the first (most
    frequent) casing wins. Words without a vector get N(0, sigma^2) with sigma = std of
    the found vectors; <pad> is all zeros."""
    stoi = {w: i for i, w in enumerate(itos)}
    found = np.zeros(len(itos), dtype=bool)
    emb, dim = None, None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f):
            parts = line.rstrip().split(" ", 1)
            if len(parts) < 2:
                continue
            word, rest = parts
            if ln == 0 and word.isdigit() and rest.strip().isdigit():  # fastText header
                dim = int(rest)
                continue
            idx = stoi.get(word.lower())
            if idx is None or found[idx] or idx == PAD_ID:
                continue
            vec = np.asarray(rest.split(" "), dtype=np.float32)
            if dim is None:
                dim = len(vec)
            if len(vec) != dim:
                continue
            if emb is None:
                emb = np.zeros((len(itos), dim), dtype=np.float32)
            emb[idx] = vec
            found[idx] = True
    if emb is None:
        raise RuntimeError(f"no vocabulary word was found in {path}")
    rng = np.random.default_rng(seed)
    missing = ~found
    missing[PAD_ID] = False
    sigma = float(emb[found].std())
    emb[missing] = rng.normal(0.0, sigma, size=(int(missing.sum()), emb.shape[1]))
    emb[PAD_ID] = 0.0

    stats = {"embedding_file": str(path), "dim": int(emb.shape[1]),
             "vocab_words_with_vector": int(found[2:].sum()),
             "type_coverage": float(found[2:].mean())}
    if counter is not None:
        vocab_tokens = sum(counter[w] for w in itos[2:])
        covered = sum(counter[w] for w, ok in zip(itos, found) if ok)
        stats["token_coverage_in_vocab"] = covered / max(vocab_tokens, 1)
        stats["token_coverage_all_train_tokens"] = covered / max(sum(counter.values()), 1)
    return emb, stats


def prepare_classic(data_dir: str = "data", embeddings: str = "fasttext", min_freq: int = 3,
                    max_len: int = 200, tokenizer: str = "spacy", force: bool = False) -> dict:
    out = Path(data_dir) / "classic"
    params = {"embeddings": str(embeddings), "min_freq": min_freq, "max_len": max_len,
              "tokenizer": tokenizer}
    if (out / "stats.json").exists() and not force:
        stats = load_json(out / "stats.json")
        if stats.get("params") == params:
            return stats
        log("[classic] parameters changed -> rebuilding")

    ds, _ = load_reviews(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = WordTokenizer(tokenizer)
    stats = {"params": params}
    tokens = {}
    for split in SPLITS:
        log(f"[classic] cleaning + tokenizing {split} ({len(ds[split]):,} reviews)")
        tokens[split] = tok.tokenize_many([clean_text(t) for t in ds[split]["text"]])

    itos, counter = build_vocab(tokens["train"], min_freq=min_freq)
    stoi = {w: i for i, w in enumerate(itos)}
    stats["vocab_size"] = len(itos)
    stats["train_types_before_min_freq"] = len(counter)
    save_json(itos, out / "vocab.json")

    for split in SPLITS:
        ids, lengths, raw_lengths, n_unk = encode(tokens[split], stoi, max_len)
        labels = np.asarray(ds[split]["label"], dtype=np.int64)
        np.savez_compressed(out / f"{split}.npz", ids=ids, lengths=lengths, labels=labels,
                            raw_lengths=raw_lengths, n_unk=n_unk)
        stats[split] = {"n": int(len(labels)),
                        "mean_tokens": float(raw_lengths.mean()),
                        "p95_tokens": float(np.percentile(raw_lengths, 95)),
                        "truncated_frac": float((raw_lengths > max_len).mean()),
                        "unk_rate": float(n_unk.sum() / lengths.sum())}

    emb_path = download_embeddings(embeddings, Path(data_dir) / "embeddings")
    log(f"[classic] building embedding matrix from {emb_path}")
    matrix, emb_stats = load_embedding_matrix(emb_path, itos, counter)
    np.save(out / "embedding_matrix.npy", matrix)
    stats["embeddings"] = emb_stats
    save_json(stats, out / "stats.json")
    log(f"[classic] vocab={len(itos):,}  embedding coverage={emb_stats['type_coverage']:.1%} of types")
    return stats


class ClassicSplit(torch.utils.data.Dataset):
    """Padded id matrix kept in memory; each batch is cut to its longest review."""

    def __init__(self, data_dir: str, split: str, max_items: int | None = None, subset_seed: int = 0):
        d = np.load(Path(data_dir) / "classic" / f"{split}.npz")
        self.ids, self.lengths, self.labels = d["ids"], d["lengths"], d["labels"]
        self.n_unk = d["n_unk"]
        n = len(self.labels)
        self.rows = np.arange(n)
        if max_items is not None and max_items < n:  # fixed random subset (same for all runs)
            self.rows = np.sort(np.random.default_rng(subset_seed).permutation(n)[:max_items])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, pos):
        return pos

    def batch_lengths(self):
        return self.lengths[self.rows]

    def collate(self, positions):
        rows = self.rows[np.asarray(positions)]
        lens = self.lengths[rows]
        t = int(lens.max())
        return {"idx": torch.from_numpy(rows.astype(np.int64)),
                "input_ids": torch.from_numpy(self.ids[rows, :t].astype(np.int64)),
                "lengths": torch.from_numpy(lens.astype(np.int64)),
                "labels": torch.from_numpy(self.labels[rows].astype(np.int64))}


# =========================================================================== RoBERTa pipeline
def load_roberta_tokenizer(model_name: str = "roberta-base"):
    try:
        from transformers import RobertaTokenizerFast
        return RobertaTokenizerFast.from_pretrained(model_name)
    except ImportError:  # newer transformers versions may only expose the unified class
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)


def roberta_cache_dir(data_dir, model_name, max_len) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(model_name))
    return Path(data_dir) / f"roberta_{safe}_len{max_len}"


def prepare_roberta(data_dir: str = "data", model_name: str = "roberta-base", max_len: int = 256,
                    num_proc: int | None = None, force: bool = False):
    """Byte-level BPE tokenization with special tokens, truncated to max_len. No padding
    here: DataCollatorWithPadding pads each batch to its own longest sequence."""
    from datasets import load_from_disk

    out = roberta_cache_dir(data_dir, model_name, max_len)
    if (out / "tok_stats.json").exists() and not force:
        return load_from_disk(str(out))
    ds, _ = load_reviews(data_dir)
    tok = load_roberta_tokenizer(model_name)
    eos = tok.eos_token_id

    def fn(batch):
        # Tokenize without truncation once, then truncate by hand (keeping </s>), so the
        # untruncated length is available for the statistics.
        enc = tok(batch["text"], truncation=False, verbose=False)
        ids, full = [], []
        for x in enc["input_ids"]:
            full.append(len(x))
            ids.append(x if len(x) <= max_len else x[:max_len - 1] + [eos])
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids],
                "length": [len(x) for x in ids], "full_length": full}

    tokd = ds.map(fn, batched=True, batch_size=1000, num_proc=num_proc,
                  remove_columns=["title", "body", "text"], desc="RoBERTa tokenization")
    tokd.save_to_disk(str(out))
    stats = {}
    for split in SPLITS:
        full = np.asarray(tokd[split]["full_length"])
        stats[split] = {"mean_tokens": float(full.mean()), "median_tokens": float(np.median(full)),
                        "p95_tokens": float(np.percentile(full, 95)),
                        "truncated_frac": float((full > max_len).mean())}
    save_json({"model_name": model_name, "max_len": max_len, **stats}, out / "tok_stats.json")
    log(f"[roberta] tokenized -> {out}  (train truncated: {stats['train']['truncated_frac']:.1%})")
    return tokd


class RobertaSplit(torch.utils.data.Dataset):
    """Fetches a whole batch from the Arrow table at once and pads it dynamically."""

    def __init__(self, hf_split, tokenizer, max_items: int | None = None, subset_seed: int = 0,
                 pad_to_multiple_of: int | None = 8):
        from transformers import DataCollatorWithPadding

        self.ds = hf_split
        self.lengths = np.asarray(hf_split["length"])
        n = len(self.lengths)
        self.rows = np.arange(n)
        if max_items is not None and max_items < n:
            self.rows = np.sort(np.random.default_rng(subset_seed).permutation(n)[:max_items])
        self.collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=pad_to_multiple_of,
                                                return_tensors="pt")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, pos):
        return pos

    def batch_lengths(self):
        return self.lengths[self.rows]

    def collate(self, positions):
        rows = self.rows[np.asarray(positions)]
        batch = self.ds[rows.tolist()]
        features = [{"input_ids": a, "attention_mask": m}
                    for a, m in zip(batch["input_ids"], batch["attention_mask"])]
        out = dict(self.collator(features))
        out["labels"] = torch.tensor(batch["label"], dtype=torch.long)
        out["idx"] = torch.from_numpy(rows.astype(np.int64))
        return out
