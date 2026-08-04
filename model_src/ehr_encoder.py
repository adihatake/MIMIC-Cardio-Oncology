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
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_head    = d_model // num_heads
        self.num_heads = num_heads
        self.d_model   = d_model

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, d_model)
        # padding_mask: (batch, seq_len)  — 1 for real tokens, 0 for padding
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask[:, None, None, :] == 0, float("-inf"))

        # Pre-dropout softmax weights — used for visualization when return_attention=True.
        # Dropout is applied only to the value-weighted context, preserving the true
        # attention distribution for inspection.
        attn_weights = torch.softmax(scores, dim=-1)          # (B, heads, seq, seq)
        context = torch.matmul(self.attn_dropout(attn_weights), V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(context)
        if return_attention:
            return out, attn_weights
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn    = MultiHeadedAttention(d_model, num_heads, dropout)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm: normalize input before each sub-layer (more stable than post-norm)
        if return_attention:
            attn_out, attn_w = self.attn(self.norm1(x), padding_mask, return_attention=True)
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.ffn(self.norm2(x)))
            return x, attn_w
        x = x + self.dropout(self.attn(self.norm1(x), padding_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
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
        num_concepts:        int,
        max_num_visits:      int   = 512,
        d_model:             int   = 128,
        num_heads:           int   = 4,
        num_layers:          int   = 4,
        ff_dim:              int   = 512,
        dropout:             float = 0.1,
        max_seq_len:         int   = 600,
        num_classes:         int   = 2,
        fusion:              str   = "add",
        use_time:            bool  = False,
        use_age:             bool  = False,
        time_scaling_factor: float = 365.25,
        num_tasks:           int   = 0,
    ) -> None:
        super().__init__()

        self.embedding = EHR_Event_Embedding(
            num_concepts=num_concepts,
            max_num_visits=max_num_visits,
            d_token_embedding=d_model,
            max_seq_len=max_seq_len,
            fusion=fusion,
            use_time=use_time,
            use_age=use_age,
            time_scaling_factor=time_scaling_factor,
            dropout=dropout,
        )

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm        = nn.LayerNorm(d_model)   # final norm (required after pre-norm stack)
        self.cls_dropout = nn.Dropout(dropout)
        self.classifier  = nn.Linear(d_model, num_classes)

        # Task token for multi-task prompt finetuning (num_tasks=0 disables this).
        # When enabled, a learned task embedding is prepended before [CLS] and the
        # model pools from that position, conditioning the representation on the
        # prediction window (90d / 180d / 365d).
        self.num_tasks  = num_tasks
        self.task_embed = nn.Embedding(num_tasks, d_model) if num_tasks > 0 else None

    def forward(
        self,
        concept_ids:  torch.Tensor,
        type_ids:     torch.Tensor,
        visit_ids:    torch.Tensor,
        position_ids: torch.Tensor,
        age_ids:      torch.Tensor,
        dates:        torch.Tensor | None = None,
        age_years:    torch.Tensor | None = None,
        task_ids:     torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.embedding(
            concept_ids, type_ids, visit_ids, position_ids,
            age_ids, dates, age_years,
        )

        padding_mask = (concept_ids != 0).long()

        if task_ids is not None and self.task_embed is not None:
            task_tok  = self.task_embed(task_ids).unsqueeze(1)   # (B, 1, d_model)
            x         = torch.cat([task_tok, x], dim=1)          # (B, S+1, d_model)
            task_mask = torch.ones(x.size(0), 1, dtype=padding_mask.dtype, device=padding_mask.device)
            padding_mask = torch.cat([task_mask, padding_mask], dim=1)

        all_attn: list[torch.Tensor] = []
        for layer in self.layers:
            if return_attention:
                x, attn_w = layer(x, padding_mask, return_attention=True)
                all_attn.append(attn_w)   # (B, num_heads, seq, seq)
            else:
                x = layer(x, padding_mask)

        x   = self.norm(x)
        cls = self.cls_dropout(x[:, 0, :])
        logits = self.classifier(cls)
        if return_attention:
            return logits, all_attn   # list has one entry per layer
        return logits

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

    task_ids = torch.randint(0, 3, (B,))
    cases = [
        ("A0 add",            dict(fusion="add",    use_time=False, use_age=False, num_tasks=0), None,  None,      None),
        ("A1 add+time",       dict(fusion="add",    use_time=True,  use_age=False, num_tasks=0), dates, None,      None),
        ("A2 add+age",        dict(fusion="add",    use_time=False, use_age=True,  num_tasks=0), None,  age_years, None),
        ("A3 add+time+age",   dict(fusion="add",    use_time=True,  use_age=True,  num_tasks=0), dates, age_years, None),
        ("B0 concat",         dict(fusion="concat", use_time=False, use_age=False, num_tasks=0), None,  None,      None),
        ("B1 concat+time",    dict(fusion="concat", use_time=True,  use_age=False, num_tasks=0), dates, None,      None),
        ("B2 concat+time+age",dict(fusion="concat", use_time=True,  use_age=True,  num_tasks=0), dates, age_years, None),
        ("MT add+tasks",      dict(fusion="add",    use_time=False, use_age=False, num_tasks=3), None,  None,      task_ids),
        ("MT add+time+tasks", dict(fusion="add",    use_time=True,  use_age=False, num_tasks=3), dates, None,      task_ids),
    ]
    for name, kwargs, d, a, t in cases:
        m = EHR_Encoder(num_concepts=V, **kwargs)
        out = m(concept_ids, type_ids, visit_ids, position_ids, age_ids, d, a, t)
        print(f"{name:<25}: {out.shape}")
    print("ehr_encoder.py OK")
