"""
mamba_embedding.py

Embedding layer for the Mamba encoder. Re-exports EHR_Event_Embedding from
embedding_layers.py under the name MambaEmbedding to keep the Mamba module
chain self-contained (ehr_mamba.py → mamba_embedding.py → embedding_layers.py).

Nothing is changed from EHR_Event_Embedding — the same fusion, use_time, and
use_age ablation flags apply identically.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_layers import EHR_Event_Embedding as MambaEmbedding

__all__ = ["MambaEmbedding"]
