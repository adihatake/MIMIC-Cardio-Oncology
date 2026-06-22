""" 
Following code defines the EHR and ECG embedding layers used for the transformers
"""


import torch
from torch import nn


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
            self, # Define some default values if not specified, but these need to be specific during data exploration. 
            num_concepts: int = 20000,
            max_num_visits: int = 50,
            num_event_types: int = 5,
            num_age_buckets: int = 10,
            d_token_embedding: int = 128,
            max_seq_len: int = 600,
            padding_idx: int = 0,
            ) -> None:
        
        super().__init__()

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

        self.layer_norm = nn.LayerNorm(d_token_embedding)
        self.dropout    = nn.Dropout(0.1)

    def forward(
        self,
        concept_ids:  torch.Tensor,
        type_ids:     torch.Tensor,
        visit_ids:    torch.Tensor,
        position_ids: torch.Tensor,
        age_ids:      torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            concept_ids:  (batch_size, seq_len) — clinical concept indices
            type_ids:     (batch_size, seq_len) — event type indices
            visit_ids:    (batch_size, seq_len) — visit indices
            position_ids: (batch_size, seq_len) — position indices 0..seq_len-1
            age_ids:      (batch_size,)          — decade bucket per patient

        Returns:
            (batch_size, seq_len, d_token_embedding)
        """
        # age_ids is (batch,); unsqueeze to (batch, 1, d_model) for broadcasting
        age_emb = self.age_embedding(age_ids).unsqueeze(1)

        # Segment alternates 0 and 1 per admission: 
        # CLS/padding (visit_id=0) → 0,
        # admission 1 → 1, 
        # admission 2 → 0
        # admission 3 → 1, ...
        segment_ids = visit_ids % 2 # dividing by modulo is the best way to do this


        # Perform element-wise addition to combine all embeddings
        embedding: torch.Tensor = (
            self.concept_embedding(concept_ids)
            + self.type_embedding(type_ids)
            + self.visit_embedding(visit_ids)
            + self.segment_embedding(segment_ids)
            + self.position_embedding(position_ids)
            + age_emb
        )

        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)

        return embedding