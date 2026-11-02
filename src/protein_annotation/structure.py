"""Per-residue SS3 secondary-structure prediction (H/E/C).

Two-tier design mirroring the repo's embedding fallback pattern:

1. ESM-2 head (``SSHeadWeights``): a windowed linear probe over frozen
   ESM-2 per-residue representations (``repr_layers`` kept instead of
   mean-pooled). Weights live in ``data/ss_head_8m.json`` (KBs) and the
   forward pass is pure numpy, so inference needs no torch once the
   representations exist.
2. Heuristic fallback (``predict_ss3_heuristic``): a Chou-Fasman-style
   propensity smoother. Zero dependencies, fully offline, always
   available — used whenever the head file or ESM-2 is missing.

SS3 splits hypothetical proteins into mostly-helical (possible TM),
sheet-rich (possible porin/outer-membrane) and coil/disordered buckets,
so callers stop trusting GO scores equally on every orphan.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Canonical SS3 alphabet order used by heads and composition dicts.
SS3_LABELS: tuple[str, str, str] = ("H", "E", "C")

# Chou-Fasman conformational propensities (x100 scale in the literature;
# stored here as P/100). Helix lovers: E/M/A/L/K; sheet lovers: V/I/Y/F.
# Used only by the zero-dependency heuristic fallback.
HELIX_PROPENSITY: dict[str, float] = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "E": 1.51, "Q": 1.11, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}
SHEET_PROPENSITY: dict[str, float] = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "E": 0.37, "Q": 1.10, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}

# Heuristic tunables: smoothing window plus minimum run lengths for the
# cleanup pass (isolated H/E singlets are physically implausible).
HEURISTIC_WINDOW: int = 7
HEURISTIC_MIN_HELIX: int = 4
HEURISTIC_MIN_SHEET: int = 3

# Default head-file name; resolved against repo root, then the CWD.
SS_HEAD_FILENAME: str = "ss_head_8m.json"


def default_ss_head_path() -> Path:
    """Locate ``data/ss_head_8m.json`` (repo root first, CWD fallback)."""
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "data" / SS_HEAD_FILENAME
    if candidate.exists():
        return candidate
    return Path("data") / SS_HEAD_FILENAME


def ss_head_available(path: str | Path | None = None) -> bool:
    """True when a readable SS-head JSON exists at `path` (or the default)."""
    candidate = Path(path) if path is not None else default_ss_head_path()
    return candidate.is_file()


@dataclass
class SSHeadWeights:
    """Windowed linear SS3 probe: concat(reps[i-w..i+w]) @ W + b -> H/E/C."""

    dim: int  # Per-residue representation width (320 for esm2_t6_8M).
    window: int  # One-sided context (full span is 2*window+1 residues).
    labels: list[str]  # Class order; always ["H", "E", "C"].
    weights: list[list[float]]  # (dim*(2*window+1), 3) row-major matrix.
    bias: list[float]  # 3 class biases.
    esm_model: str = "esm2_t6_8M_UR50D"
    trained_on: str = ""  # Provenance note (dataset + date).

    def to_json(self) -> str:
        """Serialise the head (KBs) for ``data/ss_head_8m.json``."""
        return json.dumps(
            {
                "dim": int(self.dim),
                "window": int(self.window),
                "labels": list(self.labels),
                "weights": [[float(v) for v in row] for row in self.weights],
                "bias": [float(v) for v in self.bias],
                "esm_model": self.esm_model,
                "trained_on": self.trained_on,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> "SSHeadWeights":
        """Parse a head bundle; raises ValueError on shape/label mismatch."""
        payload = json.loads(text)
        labels = [str(s) for s in payload.get("labels", [])]
        if labels != list(SS3_LABELS):
            raise ValueError(f"SS head labels must be {list(SS3_LABELS)}, got {labels}")
        dim = int(payload["dim"])
        window = int(payload["window"])
        weights = [[float(v) for v in row] for row in payload["weights"]]
        bias = [float(v) for v in payload["bias"]]
        expect_rows = dim * (2 * window + 1)
        if len(weights) != expect_rows or any(len(r) != 3 for r in weights):
            raise ValueError(
                f"SS head weights must be ({expect_rows}, 3), got ({len(weights)}, "
                f"{len(weights[0]) if weights else 0})"
            )
        if len(bias) != 3:
            raise ValueError(f"SS head bias must have length 3, got {len(bias)}")
        return cls(
            dim=dim,
            window=window,
            labels=labels,
            weights=weights,
            bias=bias,
            esm_model=str(payload.get("esm_model", "esm2_t6_8M_UR50D")),
            trained_on=str(payload.get("trained_on", "")),
        )

    def save(self, path: str | Path) -> Path:
        """Write the bundle to `path` (creating parent dirs); return it."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out


def load_ss_head(path: str | Path | None = None) -> SSHeadWeights | None:
    """Load the SS head bundle; warn-and-None when missing/unreadable.

    Args:
        path: Explicit bundle path, or None for the default lookup.

    Returns:
        Parsed weights, or None when the file is absent or invalid (the
        caller then falls back to :func:`predict_ss3_heuristic`).
    """
    candidate = Path(path) if path is not None else default_ss_head_path()
    try:
        return SSHeadWeights.from_json(candidate.read_text(encoding="utf-8"))
    except FileNotFoundError:
        warnings.warn(f"SS head not found at {candidate}; using heuristic fallback.")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        warnings.warn(f"SS head at {candidate} unusable ({exc}); using heuristic fallback.")
    return None


def predict_ss3_heuristic(
    cleaned: str,
    window: int = HEURISTIC_WINDOW,
    min_helix: int = HEURISTIC_MIN_HELIX,
    min_sheet: int = HEURISTIC_MIN_SHEET,
) -> list[str]:
    """Chou-Fasman-style SS3 assignment: smoothed propensities + run cleanup.

    Each residue scores window-averaged helix/sheet propensity against a
    coil baseline; the argmax wins, then helix runs shorter than
    `min_helix` and sheet runs shorter than `min_sheet` collapse to coil.
    Deterministic, pure Python, no dependencies.

    Args:
        cleaned: Cleaned (standard-AA) protein string.
        window: Smoothing window width (odd preferred).
        min_helix: Minimum surviving helix-run length.
        min_sheet: Minimum surviving sheet-run length.

    Returns:
        One of "H"/"E"/"C" per residue (empty input -> empty list).
    """
    seq = (cleaned or "").upper()
    n = len(seq)
    if n == 0:
        return []
    half = max(window, 1) // 2
    helix_raw = [HELIX_PROPENSITY.get(aa, 1.0) for aa in seq]
    sheet_raw = [SHEET_PROPENSITY.get(aa, 1.0) for aa in seq]
    # Coil baseline: strong helix/sheet breakers (P/G) favour coil even
    # when the window average is lukewarm.
    raw: list[str] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        h = sum(helix_raw[lo:hi]) / (hi - lo) - 1.0
        e = sum(sheet_raw[lo:hi]) / (hi - lo) - 1.0
        c = 0.05 + (0.25 if seq[i] in ("P", "G") else 0.0)
        raw.append("HEC"[int(np.argmax([h, e, c]))])
    # Single cleanup pass: short H/E runs collapse to coil.
    out = list(raw)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        run, length = out[i], j - i
        if (run == "H" and length < min_helix) or (run == "E" and length < min_sheet):
            out[i:j] = ["C"] * length
        i = j
    return out


def fold_standardisation(
    weights: np.ndarray, bias: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fold feature standardisation into a linear head.

    The probe trains on ``(x - mu) / sd`` as ``z @ W + b``; folding lets
    inference stay a pure ``x @ W_folded + b_folded`` with no scaler::

        x @ (W / sd) + (b - (mu / sd) @ W)

    The bias MUST use the unscaled ``W`` — scaling ``W`` first and then
    adjusting ``b`` with it double-divides by ``sd`` and silently breaks
    every prediction (observed as Q3 collapsing to majority-class level).

    Args:
        weights: Trained ``(d, 3)`` matrix on standardised features.
        bias: Trained ``(3,)`` vector.
        mean: Per-feature training mean ``(d,)``.
        std: Per-feature training std ``(d,)`` (entries near zero are
            treated as 1.0 upstream, so no zeros reach this function).

    Returns:
        ``(W_folded, b_folded)`` ready to store in :class:`SSHeadWeights`.
    """
    w = np.asarray(weights, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(std, dtype=np.float64)
    b_folded = b - (mu / sd) @ w
    return w / sd[:, None], b_folded


def predict_ss3_head(res_repr: np.ndarray, weights: SSHeadWeights) -> list[str]:
    """Numpy forward of the windowed linear probe over per-residue reps.

    Edges replicate the terminal representation (no zero padding), so
    short sequences still produce full-length output.

    Args:
        res_repr: (L, dim) float array of per-residue representations.
        weights: Loaded head bundle (dim must match).

    Returns:
        One of "H"/"E"/"C" per residue.
    """
    arr = np.asarray(res_repr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != weights.dim:
        raise ValueError(
            f"res_repr must be (L, {weights.dim}), got {tuple(arr.shape)}"
        )
    n = int(arr.shape[0])
    if n == 0:
        return []
    w = int(weights.window)
    # Edge-replicated padding keeps every window full-width.
    padded = np.pad(arr, ((w, w), (0, 0)), mode="edge")
    span = 2 * w + 1
    feats = np.concatenate([padded[i : i + n, :] for i in range(span)], axis=1)
    W = np.asarray(weights.weights, dtype=np.float64)  # (dim*span, 3)
    b = np.asarray(weights.bias, dtype=np.float64)  # (3,)
    idx = np.argmax(feats @ W + b, axis=1)
    return [weights.labels[int(k)] for k in idx]


def predict_ss3(
    cleaned: str,
    res_repr: np.ndarray | None = None,
    weights: SSHeadWeights | None = None,
) -> list[str]:
    """Dispatch: ESM-2 head when reps + weights fit, else the heuristic.

    Positions past the ESM length cap (1022 residues) always use the
    heuristic, even when the head scores the prefix.
    """
    from .embeddings import ESM_MAX_RESIDUES  # Local import: no cycle.

    seq = cleaned or ""
    if weights is not None and res_repr is not None:
        arr = np.asarray(res_repr)
        usable = min(len(seq), int(arr.shape[0]) if arr.ndim == 2 else 0)
        usable = min(usable, ESM_MAX_RESIDUES)
        if usable > 0 and arr.ndim == 2 and arr.shape[1] == weights.dim:
            head_part = predict_ss3_head(arr[:usable], weights)
            if usable >= len(seq):
                return head_part
            return head_part + predict_ss3_heuristic(seq[usable:])
    return predict_ss3_heuristic(seq)


def ss_composition(ss3: list[str] | str) -> dict[str, float]:
    """Fraction of H/E/C states (empty input -> all 0.0)."""
    seq = list(ss3)
    if not seq:
        return {"H": 0.0, "E": 0.0, "C": 0.0}
    n = len(seq)
    return {s: sum(1 for x in seq if x == s) / n for s in SS3_LABELS}


def ss_source(
    weights: SSHeadWeights | None, res_repr: np.ndarray | None = None
) -> str:
    """Provenance label for a prediction: head vs heuristic."""
    if weights is not None and res_repr is not None:
        arr = np.asarray(res_repr)
        if arr.ndim == 2 and arr.shape[1] == weights.dim and arr.shape[0] > 0:
            return f"esm2-ss-head:{weights.esm_model}"
    return "heuristic:chou-fasman"


def ss_accuracy(predicted: list[str] | str, actual: list[str] | str) -> float:
    """Q3 accuracy: fraction of matching SS3 states (empty -> 0.0)."""
    pred, true = list(predicted), list(actual)
    if not true or len(pred) != len(true):
        if not true:
            return 0.0
        raise ValueError(f"length mismatch: {len(pred)} predicted vs {len(true)} actual")
    return sum(1 for p, t in zip(pred, true) if p == t) / len(true)
