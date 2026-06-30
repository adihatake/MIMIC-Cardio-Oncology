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

    Two combination modes are supported (selected by use_concat_embedding):

    Additive mode (default, BEHRT-style):
        sum(concept, type, visit, segment, position, age_bucket)
        + optional CEHR-BERT sinusoidal time embedding

    Concat mode (CEHR-BERT / EHRMamba style):
        Linear( cat([concept, time, age_sinusoidal, position], dim=-1) ) → GELU
        + type + visit + segment  (additive residuals, same as CEHR-BERT)

    Args:
        num_concepts:          Vocabulary size.
        max_num_visits:        Visit embedding table size.
        num_event_types:       Event type categories (default 5).
        num_age_buckets:       Decade buckets for additive mode (default 10).
        d_token_embedding:     Embedding dimension for all sub-embeddings.
        max_seq_len:           Position table size; must match tokenisation.
        padding_idx:           Concept index with zero embedding/gradient.
        use_time_embedding:    Additive sinusoidal time embedding (additive mode only).
        time_scaling_factor:   Divisor for dates before sin(); default 365.25 (days→years).
        use_concat_embedding:  Switch to concat→FC→GELU combination (CEHR-BERT style).
                               Requires dates and age_years tensors at forward time.
    """

    def __init__(
            self,
            num_concepts:         int   = 20000,
            max_num_visits:       int   = 50,
            num_event_types:      int   = 5,
            num_age_buckets:      int   = 10,
            d_token_embedding:    int   = 128,
            max_seq_len:          int   = 600,
            padding_idx:          int   = 0,
            use_time_embedding:   bool  = False,
            time_scaling_factor:  float = 365.25,
            use_concat_embedding: bool  = False,
            ) -> None:

        super().__init__()
        self.use_time_embedding   = use_time_embedding
        self.use_concat_embedding = use_concat_embedding
        d = d_token_embedding

        # ── shared tables (used in both modes) ───────────────────────────────
        self.concept_embedding = nn.Embedding(num_concepts, d, padding_idx=padding_idx)
        self.type_embedding    = nn.Embedding(num_event_types, d)
        self.visit_embedding   = nn.Embedding(max_num_visits,  d)
        self.position_embedding = nn.Embedding(max_seq_len,    d)
        self.segment_embedding  = nn.Embedding(2,              d)

        # ── additive-mode age (decade buckets, BEHRT-style) ──────────────────
        self.age_embedding = nn.Embedding(num_age_buckets, d)

        # ── sinusoidal layers (always created; used selectively) ─────────────
        # time:  per-token absolute date  → (batch, seq, d)
        # age:   per-patient age in years → broadcast (batch, 1, d)
        self.time_embedding  = TimeEmbeddingLayer(d, scaling_factor=time_scaling_factor)
        self.age_sinusoidal  = TimeEmbeddingLayer(d, scaling_factor=1.0)

        # ── concat-mode projection ────────────────────────────────────────────
        # cat([concept(d), time(d), age(d), pos(d)]) → Linear(4d→d) → GELU
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
            age_ids:      (B,)   long  — decade bucket; used in additive mode
            dates:        (B, S) long  — days since TIME_REFERENCE_DATE;
                          required for use_time_embedding=True or use_concat_embedding=True
            age_years:    (B,)   float — continuous age in years;
                          required for use_concat_embedding=True
        Returns:
            (B, S, d_token_embedding)
        """
        segment_ids = visit_ids % 2

        if self.use_concat_embedding:
            if dates is None:
                raise ValueError(
                    "use_concat_embedding=True requires dates. "
                    "Re-run tokenization to generate dates.pt."
                )
            if age_years is None:
                raise ValueError(
                    "use_concat_embedding=True requires age_years. "
                    "Re-run tokenization to generate age_years.pt."
                )
            # age_years: (B,) → (B, 1) so TimeEmbeddingLayer gives (B, 1, d)
            age_emb = self.age_sinusoidal(age_years.unsqueeze(1))  # (B, 1, d)
            age_emb = age_emb.expand(-1, concept_ids.size(1), -1)  # (B, S, d)

            # CEHR-BERT / EHRMamba: concat the four temporal components, project
            fused = torch.cat([
                self.concept_embedding(concept_ids),   # (B, S, d)
                self.time_embedding(dates),            # (B, S, d)
                age_emb,                               # (B, S, d)
                self.position_embedding(position_ids), # (B, S, d)
            ], dim=-1)                                 # (B, S, 4d)
            embedding = torch.nn.functional.gelu(self.proj(fused))  # (B, S, d)

            # Residual discrete embeddings added after projection (CEHR-BERT style)
            embedding = (
                embedding
                + self.type_embedding(type_ids)
                + self.visit_embedding(visit_ids)
                + self.segment_embedding(segment_ids)
            )

        else:
            # Additive mode: sum all embeddings (BEHRT-style)
            age_emb = self.age_embedding(age_ids).unsqueeze(1)  # (B, 1, d) → broadcast
            embedding = (
                self.concept_embedding(concept_ids)
                + self.type_embedding(type_ids)
                + self.visit_embedding(visit_ids)
                + self.segment_embedding(segment_ids)
                + self.position_embedding(position_ids)
                + age_emb
            )
            if self.use_time_embedding:
                if dates is None:
                    raise ValueError(
                        "use_time_embedding=True requires dates. "
                        "Re-run tokenization to generate dates.pt."
                    )
                embedding = embedding + self.time_embedding(dates)

        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)
        return embedding