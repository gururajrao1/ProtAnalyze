"""ESM-2 embedding extraction with graceful offline fallback."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass

import numpy as np

# Mean-pooled embedding dimensions per supported ESM-2 checkpoint.
ESM_DIMS: dict[str, int] = {"esm2_t6_8M_UR50D": 320, "esm2_t33_650M_UR50D": 1280}

DEFAULT_ESM_MODEL: str = "esm2_t6_8M_UR50D"
SUPPORTED_ESM_MODELS: tuple[str, str] = ("esm2_t6_8M_UR50D", "esm2_t33_650M_UR50D")


@dataclass
class EmbeddingResult:
    """Batched embeddings plus provenance of how they were produced."""

    embeddings: np.ndarray  # shape (n_sequences, dim)
    source: str  # e.g. "esm2:esm2_t6_8M_UR50D" or "hashed-fallback"
    dim: int


def embedding_dim(model_name: str) -> int:
    """Return the embedding width for a supported ESM-2 checkpoint."""
    return int(ESM_DIMS.get(model_name, 320))


def hashed_embedding(sequence: str, dim: int = 320) -> np.ndarray:
    """Deterministic k-mer hash embedding used when ESM-2 is unavailable.

    Each overlapping 3-mer is hashed (md5) into one of `dim` buckets and
    the resulting count vector is L2-normalised. Fully offline and stable.

    Args:
        sequence: Cleaned (standard-AA-only) protein string.
        dim: Output width; defaults to 320 to match esm2_t6_8M_UR50D.

    Returns:
        L2-normalised float32 vector of shape (dim,).
    """
    vec = np.zeros(dim, dtype=np.float64)
    seq: str = sequence.upper() if sequence else "A"
    # Degenerate short sequences still yield a nonzero vector via full-seq hash.
    kmers: list[str] = [seq[i : i + 3] for i in range(max(len(seq) - 2, 1))]
    for kmer in kmers:
        digest: str = hashlib.md5(kmer.encode("ascii", errors="replace")).hexdigest()
        vec[int(digest, 16) % dim] += 1.0
    norm: float = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.astype(np.float32)


def _try_esm_embed(
    sequences: list[str], masks: list[list[bool]], model_name: str
) -> np.ndarray | None:
    """Attempt real ESM-2 mean-pooled embedding; return None on any failure.

    Mean pooling honours the validity mask so substituted ambiguous positions
    contribute less: masked-out positions are excluded from the average.
    """
    try:
        import esm  # type: ignore[import-not-found]  # fair-esm package
        import torch
    except Exception as exc:  # torch/esm missing -> caller falls back
        warnings.warn(f"ESM-2 unavailable ({exc}); using hashed fallback.")
        return None
    try:
        # Resolve the constructor dynamically: esm.pretrained.<model_name>().
        constructor = getattr(esm.pretrained, model_name)
        model, alphabet = constructor()
        model.eval()
        batch_converter = alphabet.get_batch_converter()
        # ESM truncates at 1022 residues; clip defensively.
        clipped: list[str] = [s[:1022] if s else "A" for s in sequences]
        data: list[tuple[str, str]] = [(f"seq{i}", s) for i, s in enumerate(clipped)]
        _, _, tokens = batch_converter(data)
        n_layers: int = int(model.num_layers)
        with torch.no_grad():
            results = model(tokens, repr_layers=[n_layers], return_contacts=False)
        reps = results["representations"][n_layers].cpu().numpy()  # (B, L, D)
        pooled: list[np.ndarray] = []
        for i, seq in enumerate(clipped):
            # Token 0 is <cls>, tokens 1..len are residues; exclude <eos>.
            residue_reps: np.ndarray = reps[i, 1 : len(seq) + 1, :]
            if i < len(masks) and len(masks[i]) >= len(seq) and any(masks[i][: len(seq)]):
                # Mask-aware mean: average valid positions only.
                valid = np.array(masks[i][: len(seq)], dtype=bool)
                pooled.append(residue_reps[valid].mean(axis=0))
            else:
                pooled.append(residue_reps.mean(axis=0))
        return np.stack(pooled).astype(np.float32)
    except Exception as exc:  # offline weights / hub failure -> fall back
        warnings.warn(f"ESM-2 inference failed ({exc}); using hashed fallback.")
        return None


def embed_sequences(
    sequences: list[str],
    masks: list[list[bool]] | None = None,
    model_name: str = DEFAULT_ESM_MODEL,
    force_fallback: bool = False,
) -> EmbeddingResult:
    """Embed sequences with ESM-2 (mean-pooled) or hashed fallback.

    Args:
        sequences: Cleaned protein strings.
        masks: Optional per-residue validity masks for masked mean pooling.
        model_name: One of SUPPORTED_ESM_MODELS.
        force_fallback: Skip ESM-2 entirely (offline/CI mode).

    Returns:
        EmbeddingResult with (n, dim) array and source provenance string.
    """
    dim: int = embedding_dim(model_name)
    masks = masks if masks is not None else [[True] * len(s) for s in sequences]
    if not force_fallback:
        esm_out: np.ndarray | None = _try_esm_embed(sequences, masks, model_name)
        if esm_out is not None:
            return EmbeddingResult(embeddings=esm_out, source=f"esm2:{model_name}", dim=int(esm_out.shape[1]))
    # Graceful fallback: ProtParam-independent deterministic hash vectors.
    fallback = np.stack([hashed_embedding(s, dim) for s in sequences]).astype(np.float32)
    return EmbeddingResult(embeddings=fallback, source="hashed-fallback", dim=dim)
