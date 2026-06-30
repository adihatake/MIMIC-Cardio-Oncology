"""
ehr_encoder.py

BERT-style encoder for cycle-level cardiotoxicity prediction.

Architecture:
    EHR_Event_Embedding  →  stack of TransformerEncoderLayers  →  CLS pooling  →  Linear classifier

Usage (standalone check):
    python model_src/ehr_encoder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_layers import EHR_Event_Embedding


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_head    = d_model // num_heads
        self.num_heads = num_heads
        self.d_model   = d_model

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # padding_mask: (batch, seq_len)  — 1 for real tokens, 0 for padding
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask[:, None, None, :] == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(context)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn    = MultiHeadedAttention(d_model, num_heads)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, padding_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class EHR_Encoder(nn.Module):
    """
    Args:
        num_concepts:    vocabulary size (read from vocab.json at runtime)
        max_num_visits:  upper bound on per-patient visit count; must cover the
                         max visit_id in the tokenized data
        d_model:         embedding and hidden dimension
        num_heads:       attention heads (must divide d_model)
        num_layers:      number of TransformerEncoderLayers
        ff_dim:          feed-forward inner dimension
        dropout:         dropout probability
        max_seq_len:     must match the value used during tokenization
        num_classes:     2 for binary cardiotoxicity prediction
    """

    def __init__(
        self,
        num_concepts:         int,
        max_num_visits:       int   = 512,
        d_model:              int   = 128,
        num_heads:            int   = 4,
        num_layers:           int   = 4,
        ff_dim:               int   = 512,
        dropout:              float = 0.1,
        max_seq_len:          int   = 600,
        num_classes:          int   = 2,
        embedding_mode:      str   = "additive",
        time_scaling_factor: float = 365.25,
    ) -> None:
        super().__init__()

        self.embedding = EHR_Event_Embedding(
            num_concepts=num_concepts,
            max_num_visits=max_num_visits,
            d_token_embedding=d_model,
            max_seq_len=max_seq_len,
            embedding_mode=embedding_mode,
            time_scaling_factor=time_scaling_factor,
        )

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        concept_ids:  torch.Tensor,
        type_ids:     torch.Tensor,
        visit_ids:    torch.Tensor,
        position_ids: torch.Tensor,
        age_ids:      torch.Tensor,
        dates:        torch.Tensor | None = None,
        age_years:    torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(
            concept_ids, type_ids, visit_ids, position_ids,
            age_ids, dates, age_years,
        )

        padding_mask = (concept_ids != 0).long()

        for layer in self.layers:
            x = layer(x, padding_mask)

        cls = x[:, 0, :]
        return self.classifier(cls)

# Smoke test using random tensors
if __name__ == "__main__":
    B, S, V = 2, 64, 500
    concept_ids  = torch.randint(0, V,  (B, S))
    type_ids     = torch.randint(0, 5,  (B, S))
    visit_ids    = torch.randint(0, 10, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    age_ids      = torch.randint(0, 10, (B,))
    dates        = torch.randint(0, 9000, (B, S))
    age_years    = torch.FloatTensor([62.0, 47.5])

    # 1. additive (BEHRT-style, no time)
    m1 = EHR_Encoder(num_concepts=V, embedding_mode="additive")
    print("additive      :", m1(concept_ids, type_ids, visit_ids, position_ids, age_ids).shape)

    # 2. additive+time (BEHRT + sinusoidal time)
    m2 = EHR_Encoder(num_concepts=V, embedding_mode="additive+time")
    print("additive+time :", m2(concept_ids, type_ids, visit_ids, position_ids, age_ids, dates).shape)

    # 3. concat (CEHR-BERT / EHRMamba — time + continuous age inside the projection)
    m3 = EHR_Encoder(num_concepts=V, embedding_mode="concat")
    print("concat        :", m3(concept_ids, type_ids, visit_ids, position_ids, age_ids, dates, age_years).shape)
    print("ehr_encoder.py OK")
