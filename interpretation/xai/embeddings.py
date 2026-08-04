"""
embeddings.py

CLS token extraction, prediction probabilities, and dimensionality reduction.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model_src.ehr_encoder import EHR_Encoder


# ── CLS extraction ────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_cls_embeddings(
    model: EHR_Encoder,
    tensors: dict[str, torch.Tensor],
    indices: list[int],
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Extract post-norm CLS token embeddings for the given sample indices.

    Handles optional task tokens (multitask models) automatically.
    Returns a (N, d_model) float tensor on CPU.
    """
    all_cls: list[torch.Tensor] = []
    n_batches = (len(indices) + batch_size - 1) // batch_size

    for b, start in enumerate(range(0, len(indices), batch_size)):
        rows = indices[start : start + batch_size]

        concept_ids  = tensors["concept_ids"][rows].to(device)
        type_ids     = tensors["type_ids"][rows].to(device)
        visit_ids    = tensors["visit_ids"][rows].to(device)
        position_ids = tensors["position_ids"][rows].to(device)
        age_ids      = tensors["age_ids"][rows].to(device)
        dates     = tensors["dates"][rows].to(device)     if "dates"     in tensors else None
        age_years = tensors["age_years"][rows].to(device) if "age_years" in tensors else None
        task_ids  = tensors["task_ids"][rows].to(device)  if "task_ids"  in tensors else None

        x            = model.embedding(concept_ids, type_ids, visit_ids,
                                        position_ids, age_ids, dates, age_years)
        padding_mask = (concept_ids != 0).long()

        if task_ids is not None and model.task_embed is not None:
            task_tok     = model.task_embed(task_ids).unsqueeze(1)
            x            = torch.cat([task_tok, x], dim=1)
            task_mask    = torch.ones(x.size(0), 1,
                                      dtype=padding_mask.dtype, device=device)
            padding_mask = torch.cat([task_mask, padding_mask], dim=1)

        for layer in model.layers:
            x = layer(x, padding_mask)
        x = model.norm(x)
        all_cls.append(x[:, 0, :].cpu())

        print(f"  CLS extraction: {b + 1}/{n_batches} batches\r", end="", flush=True)

    print()
    return torch.cat(all_cls, dim=0)


@torch.no_grad()
def get_predictions(
    model: EHR_Encoder,
    tensors: dict[str, torch.Tensor],
    indices: list[int],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Return P(cardiotoxic) for the given row indices. Shape (N,)."""
    all_probs: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        rows = indices[start : start + batch_size]
        concept_ids  = tensors["concept_ids"][rows].to(device)
        type_ids     = tensors["type_ids"][rows].to(device)
        visit_ids    = tensors["visit_ids"][rows].to(device)
        position_ids = tensors["position_ids"][rows].to(device)
        age_ids      = tensors["age_ids"][rows].to(device)
        dates     = tensors["dates"][rows].to(device)     if "dates"     in tensors else None
        age_years = tensors["age_years"][rows].to(device) if "age_years" in tensors else None
        task_ids  = tensors["task_ids"][rows].to(device)  if "task_ids"  in tensors else None

        logits = model(concept_ids, type_ids, visit_ids, position_ids,
                       age_ids, dates, age_years, task_ids)
        all_probs.append(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())

    return np.concatenate(all_probs)


# ── Dimensionality reduction ───────────────────────────────────────────────────

def reduce_dim(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    random_state: int = 42,
    **kwargs,
) -> tuple[np.ndarray, str]:
    """
    Reduce a (N, d) embedding array to (N, n_components).

    method: "umap" (preferred), "pca", or "tsne".
    Falls back to PCA if umap-learn is not installed.
    Returns (coords, method_name_used).
    """
    if method == "umap":
        try:
            import umap as umap_lib
            reducer = umap_lib.UMAP(n_components=n_components,
                                    random_state=random_state, **kwargs)
            return reducer.fit_transform(embeddings), "UMAP"
        except ImportError:
            warnings.warn("umap-learn not installed — falling back to PCA. "
                          "Install with: pip install umap-learn")
            method = "pca"

    if method == "tsne":
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        n_pca   = min(50, embeddings.shape[1], embeddings.shape[0] - 1)
        reduced = PCA(n_components=n_pca, random_state=random_state).fit_transform(embeddings)
        return TSNE(n_components=n_components, random_state=random_state,
                    **kwargs).fit_transform(reduced), "t-SNE"

    from sklearn.decomposition import PCA
    return (PCA(n_components=n_components, random_state=random_state)
            .fit_transform(embeddings), "PCA")


def project_together(
    background_embeddings: torch.Tensor,
    patient_embeddings: torch.Tensor,
    perturb_embeddings: dict[str, torch.Tensor] | None = None,
    method: str = "umap",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], str]:
    """
    Project background, patient, and perturbation CLS embeddings jointly
    so they share the same 2-D coordinate space.

    Returns (bg_coords, patient_coords, perturb_coords_dict, method_name).
    """
    bg_np  = background_embeddings.float().numpy()
    pat_np = patient_embeddings.float().numpy()
    n_bg   = len(bg_np)
    n_pat  = len(pat_np)

    parts: list[np.ndarray] = [bg_np, pat_np]
    perturb_names: list[str] = []
    if perturb_embeddings:
        for pname, pemb in perturb_embeddings.items():
            perturb_names.append(pname)
            parts.append(pemb.float().numpy())

    coords, method_name = reduce_dim(np.vstack(parts), method=method,
                                     random_state=random_state)

    bg_coords  = coords[:n_bg]
    pat_coords = coords[n_bg : n_bg + n_pat]
    ptr = n_bg + n_pat
    perturb_coords: dict[str, np.ndarray] = {}
    for pname in perturb_names:
        n_p = len(perturb_embeddings[pname])
        perturb_coords[pname] = coords[ptr : ptr + n_p]
        ptr += n_p

    return bg_coords, pat_coords, perturb_coords, method_name
