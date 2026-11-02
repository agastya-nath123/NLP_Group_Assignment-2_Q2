"""Token-level explanations for single reviews.

* BiGRU          : the additive-attention pooling weights alpha_t (they directly weight
                   the token states that form the review vector).
* RoBERTa (a)    : last-layer self-attention, averaged over the 12 heads; for each token
                   we report the attention it receives, averaged over all query positions
                   (the natural choice when the classifier mean-pools all positions).
                   Much of it lands on <s> and </s>; that share is reported separately and
                   the plot is renormalised over real words.
* RoBERTa (b)    : occlusion - remove one word at a time and measure the drop in the
                   probability of the predicted class. Positive = the word supports the
                   prediction, negative = it argues against it.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from .data import UNK_ID, WordTokenizer, clean_text, load_roberta_tokenizer
from .models import RobertaRatingClassifier, build_bigru
from .utils import get_device, load_json


# =========================================================================== BiGRU
def load_bigru(run_dir, data_dir="data", device=None):
    device = device or get_device()
    cfg = load_json(Path(run_dir) / "config.json")
    emb = np.load(Path(data_dir) / "classic" / "embedding_matrix.npy")
    model = build_bigru(cfg, emb)
    model.load_state_dict(torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=True))
    vocab = load_json(Path(data_dir) / "classic" / "vocab.json")
    return model.to(device).eval(), {w: i for i, w in enumerate(vocab)}, cfg


@torch.no_grad()
def bigru_attention(model, stoi, cfg, text, device=None):
    device = device or next(model.parameters()).device
    tokens = WordTokenizer(cfg.get("tokenizer", "spacy")).tokenize_many([clean_text(text)])[0]
    tokens = tokens[:cfg["max_len"]] or ["<unk>"]
    ids = [stoi.get(t, UNK_ID) for t in tokens]
    x = torch.tensor([ids], device=device)
    logits, alpha = model(x, torch.tensor([len(ids)]), return_attention=True)
    probs = torch.softmax(logits.float(), -1)[0].cpu().numpy()
    shown = [t if i != UNK_ID else f"{t}[UNK]" for t, i in zip(tokens, ids)]
    return {"words": shown, "attention": alpha[0].cpu().numpy(), "probs": probs,
            "pred": int(probs.argmax())}


# =========================================================================== RoBERTa
def load_roberta(run_dir, device=None):
    """Loads the fine-tuned weights into an 'eager' attention model so that attention
    probabilities can be returned (fast SDPA kernels do not expose them)."""
    device = device or get_device()
    cfg = load_json(Path(run_dir) / "config.json")
    model = RobertaRatingClassifier(cfg["model_name"], pooling=cfg["pooling"], dropout=cfg["dropout"],
                                    attn_implementation="eager")
    model.load_state_dict(torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval(), load_roberta_tokenizer(cfg["model_name"]), cfg


@torch.no_grad()
def roberta_explain(model, tokenizer, text, max_len=256, batch_size=32, device=None):
    device = device or next(model.parameters()).device
    enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    logits, attentions = model(input_ids=input_ids, attention_mask=mask, output_attentions=True)
    probs = torch.softmax(logits.float(), -1)[0].cpu().numpy()
    pred = int(probs.argmax())

    att = attentions[-1][0].float().mean(0)          # (T, T): last layer, mean over heads
    received = att.mean(0).cpu().numpy()            # attention each token receives

    word_ids = enc.word_ids(0)
    groups: dict[int, list[int]] = {}
    for pos, w in enumerate(word_ids):
        if w is not None:
            groups.setdefault(w, []).append(pos)
    order = sorted(groups)
    ids_list = input_ids[0].tolist()
    words = [tokenizer.decode([ids_list[p] for p in groups[w]]).strip() or "_" for w in order]
    word_att = np.array([received[groups[w]].sum() for w in order])
    special_share = float(received[[i for i, w in enumerate(word_ids) if w is None]].sum())

    # occlusion: drop each word's sub-tokens, re-run, measure drop in P(pred)
    variants = []
    for w in order:
        drop = set(groups[w])
        variants.append([t for p, t in enumerate(ids_list) if p not in drop])
    drops = []
    for s in range(0, len(variants), batch_size):
        chunk = variants[s:s + batch_size]
        t = max(len(v) for v in chunk)
        ids = torch.full((len(chunk), t), tokenizer.pad_token_id, dtype=torch.long)
        m = torch.zeros((len(chunk), t), dtype=torch.long)
        for i, v in enumerate(chunk):
            ids[i, :len(v)] = torch.tensor(v)
            m[i, :len(v)] = 1
        out = model(input_ids=ids.to(device), attention_mask=m.to(device))
        drops.append(probs[pred] - torch.softmax(out.float(), -1)[:, pred].cpu().numpy())
    occlusion = np.concatenate(drops) if drops else np.zeros(0)

    return {"words": words, "attention": word_att / max(word_att.sum(), 1e-9),
            "attention_raw": word_att, "special_token_attention_share": special_share,
            "occlusion": occlusion, "probs": probs, "pred": pred}


# =========================================================================== plotting
def plot_word_heatmap(words, scores, title, path=None, diverging=False, per_row=10, max_words=120):
    """Words laid out in rows; each cell coloured by its score."""
    words, scores = list(words)[:max_words], np.asarray(scores, dtype=float)[:max_words]
    n = len(words)
    rows = max(1, math.ceil(n / per_row))
    grid = np.full((rows, per_row), np.nan)
    grid.flat[:n] = scores
    fig, ax = plt.subplots(figsize=(1.25 * per_row, 0.55 * rows + 1.0))
    if diverging:
        vmax = float(np.nanmax(np.abs(scores))) or 1e-6
        im = ax.imshow(grid, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -vmax, vmax), aspect="auto")
    else:
        im = ax.imshow(grid, cmap="Reds", vmin=0, vmax=float(np.nanmax(scores)) or 1e-6, aspect="auto")
    for k, w in enumerate(words):
        r, c = divmod(k, per_row)
        ax.text(c, r, w if len(w) <= 13 else w[:12] + "…", ha="center", va="center", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def explain_example(text, true_label, baseline_run, roberta_run, data_dir="data", out_dir="results/explain",
                    tag="example"):
    """Creates three figures for one review and returns the raw numbers."""
    out_dir = Path(out_dir)
    device = get_device()
    gm, stoi, gcfg = load_bigru(baseline_run, data_dir, device)
    g = bigru_attention(gm, stoi, gcfg, text, device)
    rm, tok, rcfg = load_roberta(roberta_run, device)
    r = roberta_explain(rm, tok, text, max_len=rcfg["max_len"], device=device)
    stars = lambda k: f"{k + 1}★"  # noqa: E731
    head = f"true {stars(true_label)}"
    plot_word_heatmap(g["words"], g["attention"],
                      f"BiGRU additive attention | {head}, pred {stars(g['pred'])} (p={g['probs'].max():.2f})",
                      out_dir / f"{tag}_bigru_attention.png")
    plot_word_heatmap(r["words"], r["attention"],
                      f"RoBERTa last-layer attention received (avg over heads; {r['special_token_attention_share']:.0%} "
                      f"went to <s>/</s>) | {head}, pred {stars(r['pred'])} (p={r['probs'].max():.2f})",
                      out_dir / f"{tag}_roberta_attention.png")
    plot_word_heatmap(r["words"], r["occlusion"],
                      f"RoBERTa occlusion: drop in P(pred={stars(r['pred'])}) when the word is removed",
                      out_dir / f"{tag}_roberta_occlusion.png", diverging=True)
    plt.close("all")
    return {"bigru": g, "roberta": r}
