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

    Sums six learned embeddings per token — concept, event type, visit,
    segment, position, and age — following the BERT-style approach used in
    clinical NLP models such as BEHRT and Med-BERT.

    Segment embedding alternates between 0 and 1 across consecutive hospital
    admissions (visit_ids % 2), analogous to BERT's sentence A/B tokens.
    This lets the model distinguish between even and odd admissions without
    consuming extra capacity in the visit embedding.

    Age is a patient-level attribute (one decade bucket per patient) that is
    broadcast across every token in the sequence so every attention head can
    reference it alongside per-token signals.

    Args:
        num_concepts:       Vocabulary size (number of unique clinical concepts).
        num_visits:         Visit embedding table size. Defaults to 50, which
                            safely covers the vast majority of MIMIC patients.
                            Visit IDs are re-indexed to 1..K per sequence window
                            in the tokenizer, so this is an upper bound on K.
        num_event_types:    Number of event type categories (default 5:
                            special, diagnosis, procedure, lab, medication).
        num_age_buckets:    Number of age decade buckets (default 10: 0 to 9,
                            10 to 19, …, 90+).  Computed by the tokenizer.
        d_token_embedding:  Embedding dimension for all sub-embeddings.
        max_seq_len:        Position embedding table size; must match the
                            max_seq_len used in tokenize_ehr_dataset.
        padding_idx:        Concept index that receives a zero embedding and
                            zero gradient. Should be concept_vocab["[PAD]"] = 0.
    """

    def __init__(
            self,
            num_concepts: int = 20000,
            max_num_visits: int = 50,
            num_event_types: int = 5,
            num_age_buckets: int = 10,
            d_token_embedding: int = 128,
            max_seq_len: int = 600,
            padding_idx: int = 0,
            use_time_embedding: bool = False,
            time_scaling_factor: float = 365.25,
            ) -> None:

        super().__init__()
        self.use_time_embedding = use_time_embedding

        self.concept_embedding = nn.Embedding(
            num_embeddings=num_concepts,
            embedding_dim=d_token_embedding,
            padding_idx=padding_idx,
        )

        self.type_embedding = nn.Embedding(
            num_embeddings=num_event_types,
            embedding_dim=d_token_embedding,
        )

        self.visit_embedding = nn.Embedding(
            num_embeddings=max_num_visits,
            embedding_dim=d_token_embedding,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=d_token_embedding,
        )

        self.age_embedding = nn.Embedding(
            num_embeddings=num_age_buckets,
            embedding_dim=d_token_embedding,
        )

        # 2-way segment embedding: visit_ids % 2 alternates 0/1 across admissions.
        # CLS and padding both have visit_id=0 and receive segment 0.
        self.segment_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=d_token_embedding,
        )

        # CEHR-BERT TimeEmbeddingLayer: sin((dates / scaling_factor) * w + φ)
        # Created unconditionally so state_dicts stay consistent across ablations;
        # only applied in forward when use_time_embedding=True.
        self.time_embedding = TimeEmbeddingLayer(
            embedding_size=d_token_embedding,
            scaling_factor=time_scaling_factor,
        )

        self.layer_norm = nn.LayerNorm(d_token_embedding)
        self.dropout    = nn.Dropout(0.1)

    def forward(
        self,
        concept_ids:  torch.Tensor,
        type_ids:     torch.Tensor,
        visit_ids:    torch.Tensor,
        position_ids: torch.Tensor,
        age_ids:      torch.Tensor,
        dates:        torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            concept_ids:  (batch_size, seq_len) — clinical concept indices
            type_ids:     (batch_size, seq_len) — event type indices
            visit_ids:    (batch_size, seq_len) — visit indices
            position_ids: (batch_size, seq_len) — position indices 0..seq_len-1
            age_ids:      (batch_size,)          — decade bucket per patient
            dates:        (batch_size, seq_len) — days since TIME_REFERENCE_DATE;
                          required when use_time_embedding=True

        Returns:
            (batch_size, seq_len, d_token_embedding)
        """
        # age_ids is (batch,); unsqueeze to (batch, 1, d_model) for broadcasting
        age_emb = self.age_embedding(age_ids).unsqueeze(1)

        # Segment alternates 0 and 1 per admission:
        # CLS/padding (visit_id=0) → 0,
        # admission 1 → 1,
        # admission 2 → 0, ...
        segment_ids = visit_ids % 2

        embedding: torch.Tensor = (
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
                    "use_time_embedding=True but dates tensor was not provided. "
                    "Re-run tokenization to generate dates.pt, then pass dates in the batch."
                )
            embedding = embedding + self.time_embedding(dates)

        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)

        return embedding