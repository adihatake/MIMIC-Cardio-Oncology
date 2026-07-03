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

    Controlled by three orthogonal flags:

    fusion      "add"    (default) BEHRT-style element-wise sum of all tables.
                "concat" CEHR-BERT style: cat([concept, time*, age*, position])
                         → Linear(4d→d) → GELU, then type/visit/segment added as residuals.
                         Components disabled by use_time=False / use_age=False are zeroed
                         before projection so the Linear weight shape stays constant.
    use_time    bool  add learned sinusoidal time-gap embedding (requires dates tensor)
    use_age     bool  add continuous-age sinusoidal embedding (requires age_years tensor)

    In "add" mode the discrete decade-bucket age embedding is always included;
    use_age adds a continuous sinusoidal age signal on top of it.

    Ablation grid:
        A0  add,    use_time=F, use_age=F  — baseline
        A1  add,    use_time=T, use_age=F  — + time gap
        A2  add,    use_time=F, use_age=T  — + age
        A3  add,    use_time=T, use_age=T  — + time + age
        B0  concat, use_time=F, use_age=F  — concat fusion only
        B1  concat, use_time=T, use_age=F  — concat + time
        B2  concat, use_time=T, use_age=T  — CEHR-BERT
        C1/C2 — same flags, data_dir built with insert_att=True

    Args:
        num_concepts:         Vocabulary size.
        max_num_visits:       Visit embedding table size.
        num_event_types:      Event type categories (default 5).
        num_age_buckets:      Decade buckets for additive age (default 10).
        d_token_embedding:    Embedding dimension for all sub-embeddings.
        max_seq_len:          Position table size; must match tokenisation.
        padding_idx:          Concept index with zero embedding/gradient.
        fusion:               "add" or "concat".
        use_time:             Enable sinusoidal time-gap embedding.
        use_age:              Enable continuous-age sinusoidal embedding.
        time_scaling_factor:  Divisor for dates before sin(); default 365.25 (days→years).
    """

    def __init__(
            self,
            num_concepts:        int   = 20000,
            max_num_visits:      int   = 50,
            num_event_types:     int   = 5,
            num_age_buckets:     int   = 10,
            d_token_embedding:   int   = 128,
            max_seq_len:         int   = 600,
            padding_idx:         int   = 0,
            fusion:              str   = "add",
            use_time:            bool  = False,
            use_age:             bool  = False,
            time_scaling_factor: float = 365.25,
            ) -> None:

        super().__init__()
        if fusion not in ("add", "concat"):
            raise ValueError(f"fusion must be 'add' or 'concat', got '{fusion}'")
        self.fusion   = fusion
        self.use_time = use_time
        self.use_age  = use_age
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
            age_ids:      (B,)   long  — decade bucket; always used in "add" fusion
            dates:        (B, S) long  — days since TIME_REFERENCE_DATE; required if use_time=True
            age_years:    (B,)   float — continuous age in years; required if use_age=True
        Returns:
            (B, S, d_token_embedding)
        """
        segment_ids = visit_ids % 2
        B, S = concept_ids.shape

        if self.fusion == "concat":
            concept_emb = self.concept_embedding(concept_ids)   # (B, S, d)
            pos_emb     = self.position_embedding(position_ids) # (B, S, d)

            if self.use_time:
                if dates is None:
                    raise ValueError("use_time=True requires dates.pt.")
                t_emb = self.time_embedding(dates)              # (B, S, d)
            else:
                t_emb = torch.zeros_like(concept_emb)

            if self.use_age:
                if age_years is None:
                    raise ValueError("use_age=True requires age_years.pt.")
                a_emb = self.age_sinusoidal(age_years.unsqueeze(1)).expand(-1, S, -1)
            else:
                a_emb = torch.zeros_like(concept_emb)

            # cat → Linear(4d→d) → GELU; disabled components are zeroed, not dropped,
            # so the projection weight shape is identical across all concat ablations.
            fused = torch.cat([concept_emb, t_emb, a_emb, pos_emb], dim=-1)  # (B, S, 4d)
            embedding = torch.nn.functional.gelu(self.proj(fused))            # (B, S, d)

            embedding = (
                embedding
                + self.type_embedding(type_ids)
                + self.visit_embedding(visit_ids)
                + self.segment_embedding(segment_ids)
            )

        else:  # fusion == "add"
            age_bucket = self.age_embedding(age_ids).unsqueeze(1)  # (B, 1, d) → broadcasts
            embedding = (
                self.concept_embedding(concept_ids)
                + self.type_embedding(type_ids)
                + self.visit_embedding(visit_ids)
                + self.segment_embedding(segment_ids)
                + self.position_embedding(position_ids)
                + age_bucket
            )
            if self.use_time:
                if dates is None:
                    raise ValueError("use_time=True requires dates.pt.")
                embedding = embedding + self.time_embedding(dates)
            if self.use_age:
                if age_years is None:
                    raise ValueError("use_age=True requires age_years.pt.")
                age_sin = self.age_sinusoidal(age_years.unsqueeze(1)).expand(-1, S, -1)
                embedding = embedding + age_sin

        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)
        return embedding