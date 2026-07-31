"""
ehr_lstm.py

Bidirectional LSTM / GRU encoder for cycle-level cardiotoxicity prediction.

Architecture:
    EHR_Event_Embedding  →  stacked BiLSTM (or BiGRU)  →  mean pool  →  merge  →  Linear classifier

Drop-in replacement for EHR_Encoder: same forward() signature, select with
model_type="lstm" or model_type="gru" in TrainConfig / run_train.py.

Key design choices vs. the Transformer:
  - hidden_size = d_model per direction; bidirectional output (2·d_model) is
    projected back to d_model via a learned merge layer (same pattern as BiMamba).
  - Mean pooling over non-padding tokens (rather than CLS) is used because the
    forward LSTM at position 0 has seen only one token — mean pooling aggregates
    all real tokens symmetrically and typically outperforms CLS on small datasets.
  - Padding is handled by zeroing pad positions before the mean, not by packing,
    so batch sizes with variable-length sequences are handled correctly on MPS/CPU.

Usage (standalone smoke test):
    python model_src/ehr_lstm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_layers import EHR_Event_Embedding


class EHR_LSTM(nn.Module):
    """
    Bidirectional LSTM / GRU encoder for cardiotoxicity prediction.

    Args:
        num_concepts:        vocabulary size (from vocab.json)
        max_num_visits:      visit embedding table size; must cover max visit_id
        d_model:             embedding dimension and per-direction hidden size
        num_layers:          stacked RNN layers
        dropout:             dropout probability (inter-layer dropout requires num_layers > 1)
        max_seq_len:         must match tokenization max_seq_len
        num_classes:         2 for binary cardiotoxicity prediction
        rnn_type:            "lstm" or "gru"
        bidirectional:       True → BiLSTM/BiGRU (full context); False → causal
        fusion:              "add" | "concat" — same embedding ablation as EHR_Encoder
        use_time:            sinusoidal time-gap embedding (requires dates.pt)
        use_age:             continuous-age sinusoidal embedding (requires age_years.pt)
        time_scaling_factor: divisor applied to dates before sin(); default 365.25
    """

    def __init__(
        self,
        num_concepts:        int,
        max_num_visits:      int   = 512,
        d_model:             int   = 128,
        num_layers:          int   = 2,
        dropout:             float = 0.1,
        max_seq_len:         int   = 600,
        num_classes:         int   = 2,
        rnn_type:            str   = "lstm",
        bidirectional:       bool  = True,
        fusion:              str   = "add",
        use_time:            bool  = False,
        use_age:             bool  = False,
        time_scaling_factor: float = 365.25,
    ) -> None:
        super().__init__()

        if rnn_type not in ("lstm", "gru"):
            raise ValueError(f"rnn_type must be 'lstm' or 'gru', got '{rnn_type}'")

        self.bidirectional = bidirectional

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

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            # inter-layer dropout only applies when num_layers > 1
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Project bidirectional output (2·d_model) back to d_model before classifier.
        # bias=False matches BiMamba's merge convention.
        if bidirectional:
            self.merge = nn.Linear(2 * d_model, d_model, bias=False)

        self.norm        = nn.LayerNorm(d_model)
        self.cls_dropout = nn.Dropout(dropout)
        self.classifier  = nn.Linear(d_model, num_classes)

    def forward(
        self,
        concept_ids:      torch.Tensor,
        type_ids:         torch.Tensor,
        visit_ids:        torch.Tensor,
        position_ids:     torch.Tensor,
        age_ids:          torch.Tensor,
        dates:            torch.Tensor | None = None,
        age_years:        torch.Tensor | None = None,
        return_attention: bool = False,   # unused; kept for API compatibility with EHR_Encoder
    ) -> torch.Tensor:
        x = self.embedding(
            concept_ids, type_ids, visit_ids, position_ids,
            age_ids, dates, age_years,
        )

        output, _ = self.rnn(x)     # (B, S, d_model * num_directions)

        if self.bidirectional:
            output = self.merge(output)     # (B, S, d_model)

        output = self.norm(output)

        # Mean pool over real (non-padding) positions only.
        mask = (concept_ids != 0).float().unsqueeze(-1)   # (B, S, 1)
        pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1)  # (B, d_model)

        pooled = self.cls_dropout(pooled)
        return self.classifier(pooled)


# Smoke test using random tensors
if __name__ == "__main__":
    B, S, V = 2, 64, 500
    concept_ids  = torch.randint(1, V,  (B, S))   # 0 = padding, start from 1
    concept_ids[0, 50:] = 0                        # introduce padding in sample 0
    type_ids     = torch.randint(0, 5,  (B, S))
    visit_ids    = torch.randint(0, 10, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    age_ids      = torch.randint(0, 10, (B,))
    dates        = torch.randint(0, 9000, (B, S))
    age_years    = torch.FloatTensor([62.0, 47.5])

    cases = [
        ("BiLSTM  add",          dict(rnn_type="lstm", bidirectional=True,  fusion="add",    use_time=False, use_age=False), None,  None     ),
        ("BiLSTM  add+time",     dict(rnn_type="lstm", bidirectional=True,  fusion="add",    use_time=True,  use_age=False), dates, None     ),
        ("BiLSTM  concat+t+a",   dict(rnn_type="lstm", bidirectional=True,  fusion="concat", use_time=True,  use_age=True),  dates, age_years),
        ("UniLSTM add",          dict(rnn_type="lstm", bidirectional=False, fusion="add",    use_time=False, use_age=False), None,  None     ),
        ("BiGRU   add",          dict(rnn_type="gru",  bidirectional=True,  fusion="add",    use_time=False, use_age=False), None,  None     ),
        ("UniGRU  add",          dict(rnn_type="gru",  bidirectional=False, fusion="add",    use_time=False, use_age=False), None,  None     ),
    ]
    for name, kwargs, d, a in cases:
        m = EHR_LSTM(num_concepts=V, **kwargs)
        out = m(concept_ids, type_ids, visit_ids, position_ids, age_ids, d, a)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{name:<25}: out={out.shape}  params={n:,}")

    print("ehr_lstm.py OK")
