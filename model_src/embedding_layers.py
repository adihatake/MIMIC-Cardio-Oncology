"""
Following code defines the EHR and ECG embedding layers used for the transformers
"""


import torch
from torch import nn


class TimeEmbeddingLayer(nn.Module):
    """
    Learned sinusoidal time embedding from CEHR-BERT (Pang et al. 2021).

    Formula:  sin( (dates / scaling_factor) * w + φ )

    w and φ are learned parameters of shape (1, embedding_size), shared across
    all positions.  The result is a dense (batch, seq_len, embedding_size) tensor
    whose frequency and phase are tuned by back-prop.

    Args:
        embedding_size:  Output dimension — must equal d_model so the result can
                         be added directly to the token embedding sum.
        scaling_factor:  Divides raw day counts before multiplication.  Default
                         365.25 converts days-since-2000 to years, giving the
                         learned frequencies a clinically sensible initialisation
                         range.
    """

    def __init__(self, embedding_size: int, scaling_factor: float = 365.25) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.scaling_factor = scaling_factor
        self.w   = nn.Parameter(torch.randn(1, embedding_size))
        self.phi = nn.Parameter(torch.randn(1, embedding_size))

    def forward(self, dates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dates: (batch, seq_len) integer day counts since TIME_REFERENCE_DATE
        Returns:
            (batch, seq_len, embedding_size)
        """
        # (batch, seq_len, 1) * (1, embedding_size) → (batch, seq_len, embedding_size)
        t = (dates.float() / self.scaling_factor).unsqueeze(-1)
        return torch.sin(t * self.w + self.phi)


class EHR_Event_Embedding(nn.Module):
    """
    Multi-level embedding for EHR event sequences.

    Three embedding modes, selected by `embedding_mode`:

    "additive"  (default, BEHRT-style):
        sum(concept, type, visit, segment, position, age_bucket)
        No time signal.

    "additive+time"  (BEHRT + CEHR-BERT time):
        sum(concept, type, visit, segment, position, age_bucket)
        + sinusoidal time embedding  sin((days/365.25) * w + φ)
        Requires dates tensor.

    "concat"  (CEHR-BERT / EHRMamba style):
        Linear( cat([concept(d), time(d), age_sinusoidal(d), position(d)]) ) → GELU
        + type + visit + segment  (additive residuals)
        Time and continuous age are always active in this mode.
        Requires dates and age_years tensors.

    Args:
        num_concepts:         Vocabulary size.
        max_num_visits:       Visit embedding table size.
        num_event_types:      Event type categories (default 5).
        num_age_buckets:      Decade buckets used in additive modes (default 10).
        d_token_embedding:    Embedding dimension for all sub-embeddings.
        max_seq_len:          Position table size; must match tokenisation.
        padding_idx:          Concept index with zero embedding/gradient.
        embedding_mode:       One of "additive", "additive+time", "concat".
        time_scaling_factor:  Divisor for dates before sin(); default 365.25 (days→years).
    """

    MODES = {"additive", "additive+time", "concat"}

    def __init__(
            self,
            num_concepts:        int   = 20000,
            max_num_visits:      int   = 50,
            num_event_types:     int   = 5,
            num_age_buckets:     int   = 10,
            d_token_embedding:   int   = 128,
            max_seq_len:         int   = 600,
            padding_idx:         int   = 0,
            embedding_mode:      str   = "additive",
            time_scaling_factor: float = 365.25,
            ) -> None:

        super().__init__()
        if embedding_mode not in self.MODES:
            raise ValueError(f"embedding_mode must be one of {self.MODES}, got '{embedding_mode}'")
        self.embedding_mode = embedding_mode
        d = d_token_embedding

        # ── shared tables (used in both modes) ───────────────────────────────
        self.concept_embedding = nn.Embedding(num_concepts, d, padding_idx=padding_idx)
        self.type_embedding    = nn.Embedding(num_event_types, d)
        self.visit_embedding   = nn.Embedding(max_num_visits,  d)
        self.position_embedding = nn.Embedding(max_seq_len,    d)
        self.segment_embedding  = nn.Embedding(2,              d)

        # ── age: decade bucket (additive modes) ─────────────────────────────
        self.age_embedding = nn.Embedding(num_age_buckets, d)

        # ── sinusoidal time/age layers (always created for consistent state_dict) ──
        self.time_embedding = TimeEmbeddingLayer(d, scaling_factor=time_scaling_factor)
        self.age_sinusoidal = TimeEmbeddingLayer(d, scaling_factor=1.0)

        # ── concat-mode projection: cat([concept,time,age,pos]) → Linear(4d→d) ──
        self.proj = nn.Linear(4 * d, d)

        self.layer_norm = nn.LayerNorm(d)
        self.dropout    = nn.Dropout(0.1)

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
        """
        Args:
            concept_ids:  (B, S) long
            type_ids:     (B, S) long
            visit_ids:    (B, S) long
            position_ids: (B, S) long
            age_ids:      (B,)   long  — decade bucket; used in "additive" and "additive+time"
            dates:        (B, S) long  — days since TIME_REFERENCE_DATE;
                          required for "additive+time" and "concat"
            age_years:    (B,)   float — continuous age in years;
                          required for "concat"
        Returns:
            (B, S, d_token_embedding)
        """
        segment_ids = visit_ids % 2

        if self.embedding_mode == "concat":
            if dates is None:
                raise ValueError(
                    "embedding_mode='concat' requires dates. "
                    "Re-run tokenization to generate dates.pt."
                )
            if age_years is None:
                raise ValueError(
                    "embedding_mode='concat' requires age_years. "
                    "Re-run tokenization to generate age_years.pt."
                )
            # Continuous age broadcast across all positions
            age_emb = self.age_sinusoidal(age_years.unsqueeze(1))  # (B, 1, d)
            age_emb = age_emb.expand(-1, concept_ids.size(1), -1)  # (B, S, d)

            # CEHR-BERT / EHRMamba: concat four temporal components → project → GELU
            fused = torch.cat([
                self.concept_embedding(concept_ids),    # (B, S, d)
                self.time_embedding(dates),             # (B, S, d)
                age_emb,                                # (B, S, d)
                self.position_embedding(position_ids),  # (B, S, d)
            ], dim=-1)                                  # (B, S, 4d)
            embedding = torch.nn.functional.gelu(self.proj(fused))  # (B, S, d)

            # Type, visit, segment added as residuals after projection (CEHR-BERT style)
            embedding = (
                embedding
                + self.type_embedding(type_ids)
                + self.visit_embedding(visit_ids)
                + self.segment_embedding(segment_ids)
            )

        else:
            # "additive" and "additive+time": BEHRT-style sum
            age_emb = self.age_embedding(age_ids).unsqueeze(1)  # (B, 1, d) → broadcasts
            embedding = (
                self.concept_embedding(concept_ids)
                + self.type_embedding(type_ids)
                + self.visit_embedding(visit_ids)
                + self.segment_embedding(segment_ids)
                + self.position_embedding(position_ids)
                + age_emb
            )
            if self.embedding_mode == "additive+time":
                if dates is None:
                    raise ValueError(
                        "embedding_mode='additive+time' requires dates. "
                        "Re-run tokenization to generate dates.pt."
                    )
                embedding = embedding + self.time_embedding(dates)

        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)
        return embedding