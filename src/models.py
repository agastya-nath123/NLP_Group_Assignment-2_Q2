"""The two classifiers.

Baseline : Embedding (fastText/GloVe, frozen for epoch 1) -> 2-layer BiGRU
           -> additive attention pooling -> Dropout -> Linear(5)
RoBERTa  : RobertaModel -> masked mean of last hidden states (or <s> for the ablation)
           -> Dropout -> Linear(5)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# =========================================================================== baseline
class AdditiveAttentionPooling(nn.Module):
    """Additive (Bahdanau-style) attention with a learned query vector v:

        e_t     = v^T tanh(W h_t + b)
        alpha_t = softmax_t(e_t)      over non-padding positions only
        c       = sum_t alpha_t h_t
    """

    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h, mask):
        scores = self.v(torch.tanh(self.proj(h))).squeeze(-1).float()        # (B, T)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        alpha = torch.softmax(scores, dim=-1)                                # (B, T)
        context = torch.bmm(alpha.unsqueeze(1), h.float()).squeeze(1)        # (B, H)
        return context, alpha


class BiGRUAttentionClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_size: int = 256, num_layers: int = 2,
                 attn_dim: int = 256, gru_dropout: float = 0.3, emb_dropout: float = 0.25,
                 dropout: float = 0.5, num_classes: int = 5, pad_idx: int = 0):
        super().__init__()
        weights = torch.as_tensor(np.asarray(embedding_matrix), dtype=torch.float32)
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.gru = nn.GRU(weights.size(1), hidden_size, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=gru_dropout if num_layers > 1 else 0.0)
        self.attention = AdditiveAttentionPooling(2 * hidden_size, attn_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def set_embedding_trainable(self, trainable: bool) -> None:
        self.embedding.weight.requires_grad_(trainable)

    def forward(self, input_ids, lengths, return_attention: bool = False):
        mask = input_ids != self.pad_idx                                     # (B, T)
        x = self.emb_dropout(self.embedding(input_ids))
        # Packing makes the backward GRU start at each review's real last token,
        # not at the padding.
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        h, _ = pad_packed_sequence(out, batch_first=True, total_length=input_ids.size(1))
        context, alpha = self.attention(h, mask)
        logits = self.fc(self.dropout(context))
        return (logits, alpha) if return_attention else logits


def build_bigru(cfg: dict, embedding_matrix) -> BiGRUAttentionClassifier:
    return BiGRUAttentionClassifier(
        embedding_matrix, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
        attn_dim=cfg["attn_dim"], gru_dropout=cfg["gru_dropout"], emb_dropout=cfg["emb_dropout"],
        dropout=cfg["dropout"])


# =========================================================================== RoBERTa
def masked_mean(hidden, attention_mask):
    """Average of the token vectors where attention_mask == 1 (padding excluded)."""
    m = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class RobertaRatingClassifier(nn.Module):
    """pooling='mean'  : masked mean over all non-padded tokens of the last layer (main model)
       pooling='first' : the first-token <s> vector (ablation)"""

    def __init__(self, model_name: str = "roberta-base", pooling: str = "mean",
                 dropout: float = 0.1, num_classes: int = 5, attn_implementation: str | None = None):
        super().__init__()
        from transformers import RobertaModel

        if pooling not in ("mean", "first"):
            raise ValueError("pooling must be 'mean' or 'first'")
        kwargs = {"add_pooling_layer": False}   # we pool ourselves; drop the unused pooler
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        try:
            self.encoder = RobertaModel.from_pretrained(model_name, **kwargs)
        except TypeError:  # very old transformers without attn_implementation
            kwargs.pop("attn_implementation", None)
            self.encoder = RobertaModel.from_pretrained(model_name, **kwargs)
        hidden = self.encoder.config.hidden_size
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)
        nn.init.normal_(self.classifier.weight, std=self.encoder.config.initializer_range)
        nn.init.zeros_(self.classifier.bias)

    def pool(self, hidden, attention_mask):
        hidden = hidden.float()  # pool in fp32 even under autocast
        if self.pooling == "mean":
            return masked_mean(hidden, attention_mask)
        return hidden[:, 0]

    def forward(self, input_ids=None, attention_mask=None, output_attentions: bool = False):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           output_attentions=output_attentions, return_dict=True)
        pooled = self.pool(out.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        return (logits, out.attentions) if output_attentions else logits
