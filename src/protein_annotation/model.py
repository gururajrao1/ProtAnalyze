"""PyTorch multilabel GO-term head plus per-term F1 threshold tuning."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # Annotation-only; avoids any import-cycle risk.
    from .hmmer import HmmerHit

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

    def save_weights(self, path: str) -> str:
        """Persist head weights + label space to disk for --go-weights reuse.

        Writes two files sharing the stem of `path`: `<stem>.pt` (torch
        state dict; skipped with a warning when torch is absent) and
        `<stem>.json` (go_ids, thresholds, dims). Returns the JSON path.
        """
        import json
        from pathlib import Path as _Path

        stem: str = str(_Path(str(path)).with_suffix(""))
        meta: dict = {
            "go_ids": list(self.go_ids),
            "thresholds": [float(t) for t in self.thresholds],
            "input_dim": int(self.input_dim),
            "hidden_dim": int(self.hidden_dim),
            "dropout": float(self.dropout),
        }
        json_path: str = stem + ".json"
        _Path(json_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if self._model is not None:
            import torch  # Local import: guarded by _model presence.

            torch.save(self._model.state_dict(), stem + ".pt")  # type: ignore[union-attr]
        else:
            warnings.warn("No torch head to save; wrote thresholds/labels JSON only.")
        return json_path

    @classmethod
    def load_weights(cls, path: str, input_dim: int | None = None) -> "GOClassifier":
        """Load a classifier saved with save_weights() (or a terms JSON).

        Args:
            path: Either the `.json` sidecar or the shared stem path.
            input_dim: Override the stored input width (e.g. ESM-2 650M
                vectors are wider than the 8M default the head trained on).

        Terms-only JSON (a bare list of GO ids, or {go_id: {name,...}})
        is also accepted; the head keeps random-init weights in that case.
        """
        import json
        from pathlib import Path as _Path

        json_path = _Path(str(path))
        if json_path.suffix != ".json":
            json_path = _Path(str(json_path) + ".json")
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(meta, list):
            go_ids: list[str] = [str(g) for g in meta]
            thresholds: list[float] = [0.5] * len(go_ids)
            dims: dict = {}
        elif isinstance(meta, dict) and "go_ids" not in meta:
            # Assume {go_id: {name, namespace}} label-space dict.
            go_ids = [str(g) for g in meta.keys()]
            thresholds = [0.5] * len(go_ids)
            dims = {}
        else:
            go_ids = [str(g) for g in meta.get("go_ids", GO_IDS)]
            thresholds = [float(t) for t in meta.get("thresholds", [0.5] * len(go_ids))]
            dims = meta
        dim: int = int(input_dim) if input_dim is not None else int(dims.get("input_dim", 347))
        clf = cls(
            input_dim=dim,
            go_ids=go_ids,
            hidden_dim=int(dims.get("hidden_dim", 256)),
            dropout=float(dims.get("dropout", 0.3)),
            thresholds=thresholds,
        )
        pt_path = str(json_path.with_suffix(".pt"))
        # Load torch weights only when the file exists and torch is present.
        if clf._model is not None and _Path(pt_path).exists():
            import torch  # Local import: guarded by _model presence.

            state = torch.load(pt_path, map_location="cpu", weights_only=True)
            clf._model.load_state_dict(state)  # type: ignore[union-attr]
        elif not _Path(pt_path).exists():
            warnings.warn(f"No .pt weights at {pt_path}; using random-init head.")
        return clf


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


def information_content(Y_true: np.ndarray) -> np.ndarray:
    """Per-term information content from annotation frequencies.

    IC(t) = -log2(P(t)) where P(t) is the fraction of proteins annotated
    with term t (post-propagation frequencies in CAFA practice). Terms that
    never occur get IC 0 (no information). Used by Smin.
    """
    n: int = max(int(Y_true.shape[0]), 1)
    freq: np.ndarray = Y_true.sum(axis=0).astype(float) / n
    with np.errstate(divide="ignore"):
        ic: np.ndarray = -np.log2(np.clip(freq, 1e-12, 1.0))
    ic[freq <= 0.0] = 0.0
    return ic


def _protein_centric_pr(
    Y_true: np.ndarray, Y_score: np.ndarray, threshold: float
) -> tuple[float, float]:
    """Micro-averaged protein-centric precision/recall at one threshold.

    Proteins with no predicted terms are skipped for precision; proteins
    with no true terms are skipped for recall (CAFA convention).
    """
    pred: np.ndarray = (Y_score >= threshold).astype(int)
    precisions: list[float] = []
    recalls: list[float] = []
    for i in range(Y_true.shape[0]):
        n_pred: int = int(pred[i].sum())
        n_true: int = int(Y_true[i].sum())
        tp: int = int(((pred[i] == 1) & (Y_true[i] == 1)).sum())
        if n_pred > 0:
            precisions.append(tp / n_pred)
        if n_true > 0:
            recalls.append(tp / n_true)
    precision: float = float(sum(precisions) / len(precisions)) if precisions else 0.0
    recall: float = float(sum(recalls) / len(recalls)) if recalls else 0.0
    return precision, recall


def _semantic_distance(
    Y_true: np.ndarray, Y_score: np.ndarray, threshold: float, ic: np.ndarray
) -> float:
    """S(t) = sqrt(remaining-uncertainty^2 + misinformation^2) at threshold."""
    pred: np.ndarray = (Y_score >= threshold).astype(int)
    # False negatives weighted by IC (missed information); false positives
    # weighted by IC (wrongly added information); averaged over proteins.
    fn_ic: float = float((((Y_true == 1) & (pred == 0)) * ic).sum(axis=1).mean())
    fp_ic: float = float((((Y_true == 0) & (pred == 1)) * ic).sum(axis=1).mean())
    return float(np.sqrt(fn_ic**2 + fp_ic**2))


def _micro_aupr(Y_true: np.ndarray, Y_score: np.ndarray) -> float:
    """Micro-averaged area under the precision-recall curve (trapezoid)."""
    y: np.ndarray = Y_true.ravel().astype(int)
    s: np.ndarray = Y_score.ravel().astype(float)
    if y.sum() == 0 or y.sum() == y.size:
        return 0.0  # Degenerate: no positives or no negatives.
    order: np.ndarray = np.argsort(s)[::-1]
    y_sorted: np.ndarray = y[order]
    tp_cum: np.ndarray = np.cumsum(y_sorted).astype(float)
    fp_cum: np.ndarray = np.cumsum(1 - y_sorted).astype(float)
    precision: np.ndarray = tp_cum / np.clip(tp_cum + fp_cum, 1, None)
    recall: np.ndarray = tp_cum / y.sum()
    # Prepend (0, p0) anchor then integrate precision over recall.
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0]], precision])
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(precision, recall))


def cafa_eval_report(
    Y_true: np.ndarray,
    Y_score: np.ndarray,
    go_ids: list[str] | None = None,
    grid: np.ndarray | None = None,
) -> dict:
    """CAFA-style evaluation report: per-term F1, Fmax, Smin, AUPR.

    Args:
        Y_true: Binary (n, n_terms) ground-truth labels.
        Y_score: Predicted (n, n_terms) probabilities.
        go_ids: Term ids for per-term rows (defaults to T0..Tn-1).
        grid: Threshold candidates; defaults to 0.01..0.99 step 0.01.

    Returns:
        JSON-serialisable dict with per_term[{go_id, f1, threshold}],
        fmax/fmax_threshold, smin/smin_threshold, aupr_micro and n_proteins.
    """
    if grid is None:
        grid = np.arange(0.01, 1.0, 0.01)
    n_terms: int = int(Y_true.shape[1])
    ids: list[str] = list(go_ids) if go_ids else [f"T{i}" for i in range(n_terms)]
    # Per-term best F1 over the grid (CAFA-evaluator-compatible term-centric view).
    per_term: list[dict] = []
    for j in range(n_terms):
        best_f1: float = 0.0
        best_t: float = 0.5
        for t in grid:
            f1: float = f1_at_threshold(Y_true[:, j], Y_score[:, j], float(t))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        per_term.append({"go_id": ids[j], "f1": round(best_f1, 4), "threshold": round(best_t, 3)})
    # Protein-centric sweep for Fmax / Smin.
    ic: np.ndarray = information_content(Y_true)
    fmax: float = 0.0
    fmax_t: float = 0.5
    smin: float = float("inf")
    smin_t: float = 0.5
    for t in grid:
        precision, recall = _protein_centric_pr(Y_true, Y_score, float(t))
        f1: float = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > fmax:
            fmax, fmax_t = f1, float(t)
        dist: float = _semantic_distance(Y_true, Y_score, float(t), ic)
        if dist < smin:
            smin, smin_t = dist, float(t)
    return {
        "n_proteins": int(Y_true.shape[0]),
        "n_terms": n_terms,
        "per_term": per_term,
        "fmax": round(float(fmax), 4),
        "fmax_threshold": round(float(fmax_t), 3),
        "smin": round(float(smin), 4),
        "smin_threshold": round(float(smin_t), 3),
        "aupr_micro": round(float(_micro_aupr(Y_true, Y_score)), 4),
    }


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


def parse_go_map(text: str) -> dict[str, set[str]]:
    """Parse a target->GO map TSV (`target_id <TAB> GO:1,GO:2`) for blending.

    Skips blank lines and '#' comments; malformed lines are ignored with a
    warning so a hand-edited map never crashes a run.
    """
    mapping: dict[str, set[str]] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts: list[str] = line.rstrip("\n").split("\t")
        if len(parts) < 2 or not parts[0].strip():
            warnings.warn(f"Ignoring malformed go-map line {lineno}.")
            continue
        terms: set[str] = {t.strip() for t in parts[1].split(",") if t.strip()}
        if terms:
            mapping.setdefault(parts[0].strip(), set()).update(terms)
    return mapping


def homolog_scores_from_hits(
    hits: list["HmmerHit"], go_map: dict[str, set[str]], go_ids: list[str]
) -> dict[str, dict[str, float]]:
    """Transfer GO terms from HMMER hits, weighted by normalised bitscore.

    DiamondScore-portable idea: each mapped hit votes for its target's GO
    terms with weight bitscore/max_bitscore (per query, so the best hit
    votes 1.0); a term's score is the max vote across hits. Terms outside
    `go_ids` and unmapped targets are ignored. Returns {query: {term: score}}.
    """
    want: set[str] = set(go_ids)
    by_query: dict[str, list["HmmerHit"]] = {}
    for hit in hits:
        by_query.setdefault(hit.query_id, []).append(hit)
    scores: dict[str, dict[str, float]] = {}
    for query, qhits in by_query.items():
        mapped: list["HmmerHit"] = [h for h in qhits if h.target_id in go_map]
        if not mapped:
            continue  # No mappable hits; homolog arm abstains for this query.
        best: float = max(h.bitscore for h in mapped)
        if best <= 0:
            continue  # Degenerate scores; abstain rather than divide by zero.
        term_scores: dict[str, float] = {}
        for hit in mapped:
            weight: float = hit.bitscore / best
            for term in go_map[hit.target_id]:
                if term in want and weight > term_scores.get(term, 0.0):
                    term_scores[term] = weight
        if term_scores:
            scores[query] = term_scores
    return scores


def blend_scores(
    head: np.ndarray, homolog: dict[str, float], go_ids: list[str], alpha: float
) -> np.ndarray:
    """Convex blend of head sigmoids with homolog votes for one protein.

    blended[j] = alpha * head[j] + (1 - alpha) * homolog(term_j),
    with absent homolog terms treated as 0. alpha=1.0 reproduces the head.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"blend alpha must be in [0, 1], got {alpha}")
    homolog_vec: np.ndarray = np.array(
        [homolog.get(term, 0.0) for term in go_ids], dtype=float
    )
    return alpha * np.asarray(head, dtype=float) + (1.0 - alpha) * homolog_vec


def smooth_over_ancestors(
    scores: np.ndarray, go_ids: list[str], parents: dict[str, set[str]]
) -> np.ndarray:
    """Enforce true-path consistency: no ancestor scores below its descendants.

    smoothed(t) = max(score(t), max over children of smoothed(child)),
    computed leaves-first via memoised DFS over the child map derived from
    `parents`. Terms outside the map pass through unchanged.
    """
    children: dict[str, set[str]] = {}
    for child, pars in parents.items():
        for par in pars:
            children.setdefault(par, set()).add(child)
    base: dict[str, float] = {term: float(scores[j]) for j, term in enumerate(go_ids)}
    memo: dict[str, float] = {}

    def _smooth(term: str) -> float:
        if term in memo:
            return memo[term]
        memo[term] = base.get(term, 0.0)  # Pre-seed: cycle-safe.
        best: float = memo[term]
        for child in children.get(term, ()):
            child_score: float = _smooth(child)
            if child_score > best:
                best = child_score
        memo[term] = best
        return best

    for term in go_ids:
        _smooth(term)
    return np.array([memo.get(term, float(scores[j])) for j, term in enumerate(go_ids)])
