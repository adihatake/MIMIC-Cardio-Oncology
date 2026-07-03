"""
ehr_mamba.py

Bidirectional Mamba encoder for cycle-level cardiotoxicity prediction,
using the official mamba_ssm.Mamba block from github.com/state-spaces/mamba.

Requires CUDA + mamba-ssm:
    pip install causal-conv1d mamba-ssm

EHR-specific code on top of mamba_ssm (the only custom parts):
  - BiMambaBlock : bidirectional wrapper — two Mamba instances (fwd + bwd)
                   with a learned linear merge.
  - EHR_Mamba   : EHR_Event_Embedding → BiMambaBlocks → CLS pooling → classifier.
                   Drop-in replacement for EHR_Encoder (same forward signature).

Usage (smoke test, requires CUDA):
    python model_src/ehr_mamba.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from mamba_ssm import Mamba  # pip install causal-conv1d mamba-ssm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mamba_embedding import MambaEmbedding


class MambaBlock(nn.Module):
    """
    Causal Mamba block: LayerNorm → mamba_ssm.Mamba → residual + Dropout.

    mamba_ssm.Mamba internally handles in_proj, depthwise Conv1d, the
    selective state-space scan, SiLU gating, and out_proj.
    """

    def __init__(
        self,
        d_model:  int,
        d_state:  int   = 16,
        d_conv:   int   = 4,
        d_expand: int   = 2,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.norm  = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=d_expand)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, _padding_mask=None) -> torch.Tensor:
        return self.drop(self.mamba(self.norm(x))) + x


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block: two independent mamba_ssm.Mamba instances scan
    the sequence in opposite directions, giving every token access to full
    left- and right-context (analogous to a BiLSTM, but with SSM dynamics).
    Their outputs are concatenated and projected back to d_model via a learned
    linear merge.

    mamba_ssm.Mamba handles all SSM internals — in_proj, depthwise Conv1d,
    selective state-space scan, SiLU gating, out_proj. This class only adds
    the pre-norm, the reverse pass, and the merge.

    Layout:
        xn    = LayerNorm(x)
        y_fwd = Mamba_fwd( xn )               # (B, L, d_model)
        y_bwd = Mamba_bwd( flip(xn) ).flip()  # (B, L, d_model)
        y     = merge( cat([y_fwd, y_bwd]) )  # Linear(2·d_model → d_model)
        return Dropout(y) + x
    """

    def __init__(
        self,
        d_model:  int,
        d_state:  int   = 16,
        d_conv:   int   = 4,
        d_expand: int   = 2,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.norm      = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=d_expand)
        self.mamba_bwd = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=d_expand)
        self.merge     = nn.Linear(2 * d_model, d_model, bias=False)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, _padding_mask=None) -> torch.Tensor:
        xn    = self.norm(x)
        y_fwd = self.mamba_fwd(xn)
        y_bwd = self.mamba_bwd(xn.flip(1)).flip(1)
        return self.drop(self.merge(torch.cat([y_fwd, y_bwd], dim=-1))) + x


class EHR_Mamba(nn.Module):
    """
    Bidirectional Mamba encoder for cardiotoxicity prediction.

    Architecture:
        EHR_Event_Embedding  →  N × BiMambaBlock  →  CLS pooling  →  Linear classifier

    Drop-in replacement for EHR_Encoder: same forward() signature, select with
    model_type="mamba" in TrainConfig / run_mamba.py.

    Args:
        num_concepts:        vocabulary size (from vocab.json)
        max_num_visits:      visit embedding table size; must cover max visit_id
        d_model:             embedding and hidden dimension
        num_layers:          number of Mamba blocks
        d_state:             SSM latent state size N; larger → more capacity (default 16)
        d_conv:              depthwise Conv1d kernel width (default 4)
        d_expand:            d_inner = d_expand × d_model (default 2)
        dropout:             dropout probability
        max_seq_len:         must match tokenization max_seq_len
        num_classes:         2 for binary cardiotoxicity prediction
        fusion:              "add" | "concat" — same options as EHR_Encoder
        use_time:            sinusoidal time-gap embedding (requires dates.pt)
        use_age:             continuous-age sinusoidal embedding (requires age_years.pt)
        bidirectional:       True → BiMambaBlock (recommended); False → causal MambaBlock
        time_scaling_factor: divisor applied to dates before sin(); default 365.25
    """

    def __init__(
        self,
        num_concepts:        int,
        max_num_visits:      int   = 512,
        d_model:             int   = 128,
        num_layers:          int   = 4,
        d_state:             int   = 16,
        d_conv:              int   = 4,
        d_expand:            int   = 2,
        dropout:             float = 0.1,
        max_seq_len:         int   = 600,
        num_classes:         int   = 2,
        fusion:              str   = "add",
        use_time:            bool  = False,
        use_age:             bool  = False,
        bidirectional:       bool  = True,
        time_scaling_factor: float = 365.25,
    ) -> None:
        super().__init__()

        self.embedding = MambaEmbedding(
            num_concepts=num_concepts,
            max_num_visits=max_num_visits,
            d_token_embedding=d_model,
            max_seq_len=max_seq_len,
            fusion=fusion,
            use_time=use_time,
            use_age=use_age,
            time_scaling_factor=time_scaling_factor,
        )

        BlockClass = BiMambaBlock if bidirectional else MambaBlock
        self.layers = nn.ModuleList([
            BlockClass(d_model=d_model, d_state=d_state, d_conv=d_conv,
                       d_expand=d_expand, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm        = nn.LayerNorm(d_model)
        self.cls_dropout = nn.Dropout(dropout)
        self.classifier  = nn.Linear(d_model, num_classes)

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

        for layer in self.layers:
            x = layer(x)

        x   = self.norm(x)
        cls = self.cls_dropout(x[:, 0, :])
        return self.classifier(cls)


# Smoke test — run on your CUDA cluster:  python model_src/ehr_mamba.py
if __name__ == "__main__":
    assert torch.cuda.is_available(), (
        "Smoke test requires CUDA. Install mamba-ssm on your cluster:\n"
        "  pip install causal-conv1d mamba-ssm"
    )

    B, S, V = 2, 64, 500
    kw_int   = dict(device="cuda", dtype=torch.long)
    kw_float = dict(device="cuda", dtype=torch.float32)

    concept_ids  = torch.randint(0, V,    (B, S), **kw_int)
    type_ids     = torch.randint(0, 5,    (B, S), **kw_int)
    visit_ids    = torch.randint(0, 10,   (B, S), **kw_int)
    position_ids = torch.arange(S, **kw_int).unsqueeze(0).expand(B, -1)
    age_ids      = torch.randint(0, 10,   (B,),   **kw_int)
    dates        = torch.randint(0, 9000, (B, S), **kw_int)
    age_years    = torch.tensor([62.0, 47.5], **kw_float)

    cases = [
        ("BiMamba  A0 add",          dict(fusion="add",    use_time=False, use_age=False, bidirectional=True),  None,      None     ),
        ("BiMamba  A1 add+time",     dict(fusion="add",    use_time=True,  use_age=False, bidirectional=True),  dates,     None     ),
        ("BiMamba  B2 concat+t+a",   dict(fusion="concat", use_time=True,  use_age=True,  bidirectional=True),  dates,     age_years),
        ("UniMamba A0 add (causal)", dict(fusion="add",    use_time=False, use_age=False, bidirectional=False), None,      None     ),
    ]
    for name, kwargs, d, a in cases:
        m   = EHR_Mamba(num_concepts=V, **kwargs).cuda()
        out = m(concept_ids, type_ids, visit_ids, position_ids, age_ids, d, a)
        n   = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{name:<35}: out={out.shape}  params={n:,}")

    print("ehr_mamba.py OK")
