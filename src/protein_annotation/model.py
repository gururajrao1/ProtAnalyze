"""PyTorch multilabel GO-term head plus per-term F1 threshold tuning."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

# Curated demo GO label space spanning BP / MF / CC.
GO_TERMS: dict[str, dict[str, str]] = {
    "GO:0003824": {"name": "catalytic activity", "namespace": "MF"},
    "GO:0003677": {"name": "DNA binding", "namespace": "MF"},
    "GO:0005524": {"name": "ATP binding", "namespace": "MF"},
    "GO:0004871": {"name": "signal transducer activity", "namespace": "MF"},
    "GO:0006355": {"name": "regulation of transcription", "namespace": "BP"},
    "GO:0006412": {"name": "translation", "namespace": "BP"},
    "GO:0006979": {"name": "response to oxidative stress", "namespace": "BP"},
    "GO:0006810": {"name": "transport", "namespace": "BP"},
    "GO:0016021": {"name": "integral component of membrane", "namespace": "CC"},
    "GO:0005737": {"name": "cytoplasm", "namespace": "CC"},
    "GO:0005840": {"name": "ribosome", "namespace": "CC"},
    "GO:0005634": {"name": "nucleus", "namespace": "CC"},
}
GO_IDS: list[str] = list(GO_TERMS.keys())


def _torch_available() -> bool:
    """Probe for torch without importing it at module scope."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def build_head(input_dim: int, n_terms: int, hidden_dim: int = 256, dropout: float = 0.3):
    """Build Linear -> BatchNorm -> GELU -> Dropout(0.3) -> Linear head.

    Raises ImportError when torch is unavailable; callers fall back to the
    numpy dummy classifier in that case.
    """
    import torch.nn as nn  # Local import keeps module importable w/o torch.

    class MultilabelHead(nn.Module):
        """Two-layer multilabel classifier returning raw logits."""

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),  # Project embeddings up.
                nn.BatchNorm1d(hidden_dim),  # Stabilise small-batch training.
                nn.GELU(),  # Smooth non-linearity.
                nn.Dropout(dropout),  # Regulariser (spec: 0.3).
                nn.Linear(hidden_dim, n_terms),  # One logit per GO term.
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            """Map (batch, input_dim) features to (batch, n_terms) logits."""
            return self.net(x)

    return MultilabelHead()


@dataclass
class GOClassifier:
    """Multilabel GO predictor: torch head when available, dummy otherwise."""

    input_dim: int
    go_ids: list[str] = field(default_factory=lambda: list(GO_IDS))
    hidden_dim: int = 256
    dropout: float = 0.3
    thresholds: list[float] = field(default_factory=list)
    _model: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialise per-term thresholds to 0.5 and build the head lazily."""
        if not self.thresholds:
            self.thresholds = [0.5] * len(self.go_ids)
        if _torch_available():
            # Build torch head eagerly so inference works out of the box.
            self._model = build_head(self.input_dim, len(self.go_ids), self.hidden_dim, self.dropout)
        else:
            warnings.warn("torch missing; GOClassifier uses dummy uniform scores.")

    @property
    def n_terms(self) -> int:
        """Number of GO terms in the label space."""
        return len(self.go_ids)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return (n, n_terms) probabilities via sigmoid (or dummy 1/n)."""
        if self._model is None:
            # Dummy fallback: uniform low scores, deterministic shape.
            return np.full((features.shape[0], self.n_terms), 1.0 / self.n_terms, dtype=np.float32)
        import torch  # Local import: only runs when torch exists.

        self._model.eval()  # type: ignore[union-attr]  # nn.Module at runtime.
        with torch.no_grad():
            tensor = torch.as_tensor(features, dtype=torch.float32)
            import torch.nn.functional as f

            logits = self._model(tensor)  # type: ignore[operator]
            return f.sigmoid(logits).cpu().numpy().astype(np.float32)

    def predict(self, features: np.ndarray, top_k: int = 3) -> tuple[list[list[str]], list[list[float]]]:
        """Top-k GO ids + scores per sample, ranked by probability."""
        proba: np.ndarray = self.predict_proba(features)
        top_terms: list[list[str]] = []
        top_scores: list[list[float]] = []
        for row in proba:
            # argsort descending, take top_k.
            idx: np.ndarray = np.argsort(row)[::-1][:top_k]
            top_terms.append([self.go_ids[i] for i in idx])
            top_scores.append([float(row[i]) for i in idx])
        return top_terms, top_scores

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 8,
    ) -> "GOClassifier":
        """Train head with BCEWithLogitsLoss; tune thresholds on validation.

        Falls back to a no-op (keeping 0.5 thresholds) when torch is missing
        or no validation split is supplied.
        """
        if self._model is None or X_val is None or Y_val is None:
            return self  # Nothing trainable; keep defaults.
        import torch  # Local import: guarded by _model is not None.

        model = self._model
        model.train()  # type: ignore[union-attr]
        opt = torch.optim.Adam(model.parameters(), lr=lr)  # type: ignore[union-attr]
        loss_fn = torch.nn.BCEWithLogitsLoss()
        n: int = X_train.shape[0]
        for _ in range(epochs):
            # Simple sequential mini-batches (demo scale; no shuffling need).
            for start in range(0, n, batch_size):
                xb = torch.as_tensor(X_train[start : start + batch_size], dtype=torch.float32)
                yb = torch.as_tensor(Y_train[start : start + batch_size], dtype=torch.float32)
                # BatchNorm needs batch > 1; duplicate singletons defensively.
                if xb.shape[0] < 2:
                    xb = xb.repeat(2, 1)
                    yb = yb.repeat(2, 1)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)  # type: ignore[operator]
                loss.backward()
                opt.step()
        # Tune per-term thresholds on the held-out validation split.
        val_proba: np.ndarray = self.predict_proba(X_val)
        self.thresholds = tune_thresholds(Y_val, val_proba)
        return self


def f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    """Binary F1 of one term's scores binarised at `threshold`."""
    pred: np.ndarray = (y_score >= threshold).astype(int)
    tp: int = int(((pred == 1) & (y_true == 1)).sum())
    fp: int = int(((pred == 1) & (y_true == 0)).sum())
    fn: int = int(((pred == 0) & (y_true == 1)).sum())
    if tp == 0:
        return 0.0  # Avoid 0/0; no correct positives means F1 = 0.
    precision: float = tp / (tp + fp)
    recall: float = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def tune_thresholds(
    Y_true: np.ndarray, Y_score: np.ndarray, grid: np.ndarray | None = None
) -> list[float]:
    """Pick the F1-best threshold per GO term over a 0.1..0.9 grid.

    Args:
        Y_true: Binary (n, n_terms) validation labels.
        Y_score: Predicted (n, n_terms) probabilities.
        grid: Threshold candidates; defaults to 0.10..0.90 step 0.05.

    Returns:
        One tuned threshold per term (ties break toward lower threshold).
    """
    if grid is None:
        grid = np.arange(0.10, 0.901, 0.05)
    tuned: list[float] = []
    for j in range(Y_true.shape[1]):
        best_t: float = 0.5
        best_f1: float = -1.0
        for t in grid:
            f1: float = f1_at_threshold(Y_true[:, j], Y_score[:, j], float(t))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        tuned.append(best_t)
    return tuned
