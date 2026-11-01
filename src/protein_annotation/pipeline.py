"""End-to-end annotation pipeline: FASTA in, TSV annotations out."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio import SeqIO

from .amr import AmrHit, run_amrfinder
from .embeddings import DEFAULT_ESM_MODEL, EmbeddingResult, embed_sequences
from .go_labels import parse_obo_ancestors
from .hmmer import HmmerHit, run_phmmer, top_hit_per_query
from .model import (
    GO_TERMS,
    GOClassifier,
    blend_scores,
    cafa_eval_report,
    homolog_scores_from_hits,
    parse_go_map,
    smooth_over_ancestors,
)
from .sequence import (
    CleanedSequence,
    ProtParamFeatures,
    clean_sequence,
    compute_protparam_features,
    disorder_fraction,
)

# Exact TSV columns produced by the pipeline (CLI contract).
TSV_COLUMNS: list[str] = [
    "Protein_ID",
    "Sequence_Length",
    "Molecular_Weight",
    "Top_GO_Terms",
    "Confidence_Scores",
    "HMMER_Top_Hit",
    "Predicted_Function",
]

# Opt-in extended columns (--extended-tsv): disorder signal + embedding provenance.
EXTENDED_COLUMNS: list[str] = ["Disordered_Fraction", "Embedding_Source"]
EXTENDED_TSV_COLUMNS: list[str] = TSV_COLUMNS + EXTENDED_COLUMNS


@dataclass
class AnnotationRecord:
    """One annotated protein, ready for TSV serialisation."""

    protein_id: str
    length: int
    molecular_weight: float
    top_go_terms: list[str]
    confidence_scores: list[float]
    hmmer_hit: str
    predicted_function: str
    # Optional offline signals (None/"" when skipped); excluded from default TSV.
    disordered_fraction: float | None = None
    embedding_source: str = ""
    # True when Top_GO_Terms came from the homology-blended/smoothed scores.
    blended: bool = False

    def to_row(self, extended: bool = False) -> list[str]:
        """Serialise to TSV_COLUMNS order (plus extended cols when asked)."""
        row: list[str] = [
            self.protein_id,
            str(self.length),
            f"{self.molecular_weight:.2f}",
            ";".join(self.top_go_terms),
            ";".join(f"{s:.3f}" for s in self.confidence_scores),
            self.hmmer_hit,
            self.predicted_function,
        ]
        if extended:  # Opt-in 9-column TSV; default stays 7-column compatible.
            dis: str = f"{self.disordered_fraction:.3f}" if self.disordered_fraction is not None else "-"
            row += [dis, self.embedding_source or "-"]
        return row


def describe_function(top_terms: list[str], top_scores: list[float], hmmer_hit: str) -> str:
    """Human-readable function line from top GO term + HMMER evidence."""
    if top_terms:
        go_id: str = top_terms[0]
        name: str = GO_TERMS.get(go_id, {}).get("name", "unknown function")
        base: str = f"Putative {name} ({go_id})"
    else:
        base = "Hypothetical protein (no GO prediction)"
    # Append homology evidence when available.
    if hmmer_hit and hmmer_hit != "-":
        return f"{base}; HMMER homolog: {hmmer_hit}"
    return f"{base}; no HMMER homolog"


def annotate_records(
    cleaned: list[CleanedSequence],
    features: list[ProtParamFeatures],
    embedding: EmbeddingResult,
    hmmer_best: dict[str, HmmerHit],
    classifier: GOClassifier,
    top_k: int = 3,
    include_disorder: bool = True,
    homolog_go: dict[str, dict[str, float]] | None = None,
    blend_alpha: float = 1.0,
    obo_parents: dict[str, set[str]] | None = None,
) -> list[AnnotationRecord]:
    """Combine ProtParam + embeddings + GO head + HMMER into records.

    Optional post-processing (all off by default, hence backward compatible):
    homolog-arm blending (alpha < 1 with `homolog_go` votes) and GO-DAG
    ancestor smoothing (`obo_parents`). Either sets record.blended = True.
    """
    # Model input = ESM/hashed embedding concatenated with ProtParam vector.
    prot_vectors = np.array([f.to_vector() for f in features], dtype=np.float32)
    combined = np.concatenate([embedding.embeddings.astype(np.float32), prot_vectors], axis=1)
    # Rebuild classifier head if the fused width differs (e.g. 650M model).
    if classifier.input_dim != combined.shape[1]:
        classifier = GOClassifier(
            input_dim=combined.shape[1], go_ids=classifier.go_ids, thresholds=classifier.thresholds
        )
    proba: np.ndarray = classifier.predict_proba(combined)
    go_ids: list[str] = classifier.go_ids
    # Per-protein final scores: head -> optional blend -> optional smoothing.
    blended_any: bool = bool(homolog_go) and blend_alpha < 1.0
    smoothed_any: bool = obo_parents is not None
    final_scores: list[np.ndarray] = []
    for i, seq in enumerate(cleaned):
        row: np.ndarray = np.asarray(proba[i], dtype=float)
        if blended_any and homolog_go is not None and seq.record_id in homolog_go:
            row = blend_scores(row, homolog_go[seq.record_id], go_ids, blend_alpha)
        if smoothed_any and obo_parents is not None:
            row = smooth_over_ancestors(row, go_ids, obo_parents)
        final_scores.append(row)
    records: list[AnnotationRecord] = []
    for i, seq in enumerate(cleaned):
        # Top-k from final scores (identical to classifier.predict by default).
        idx: np.ndarray = np.argsort(final_scores[i])[::-1][:top_k]
        terms: list[str] = [go_ids[j] for j in idx]
        scores: list[float] = [float(final_scores[i][j]) for j in idx]
        hit: HmmerHit | None = hmmer_best.get(seq.record_id)
        hit_str: str = hit.short() if hit else "-"
        # Offline disorder signal (pure CPU); None when --no-disorder skips it.
        dis: float | None = disorder_fraction(seq.cleaned) if include_disorder else None
        records.append(
            AnnotationRecord(
                protein_id=seq.record_id,
                length=features[i].length,
                molecular_weight=features[i].molecular_weight,
                top_go_terms=terms,
                confidence_scores=scores,
                hmmer_hit=hit_str,
                predicted_function=describe_function(terms, scores, hit_str),
                disordered_fraction=dis,
                embedding_source=embedding.source,
                blended=bool(blended_any or smoothed_any),
            )
        )
    return records


def load_eval_labels(path: str | Path, protein_ids: list[str], go_ids: list[str]) -> np.ndarray:
    """Load eval labels TSV (protein_id <TAB> GO:1,GO:2) into a binary matrix.

    Unknown proteins/terms are ignored with a warning-free skip (labels may
    cover a subset); rows follow `protein_ids` order, columns `go_ids` order.
    """
    import warnings as _warnings

    want_proteins: set[str] = set(protein_ids)
    want_terms: dict[str, int] = {g: j for j, g in enumerate(go_ids)}
    label_sets: dict[str, set[str]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue  # Allow comments/blank lines in label files.
            parts: list[str] = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                _warnings.warn(f"Ignoring malformed eval-labels line {lineno}.")
                continue
            pid: str = parts[0].strip()
            if pid in want_proteins:
                label_sets[pid] = {t.strip() for t in parts[1].split(",") if t.strip()}
    Y: np.ndarray = np.zeros((len(protein_ids), len(go_ids)), dtype=int)
    for i, pid in enumerate(protein_ids):
        for term in label_sets.get(pid, ()):
            if term in want_terms:
                Y[i, want_terms[term]] = 1
    return Y


def write_amr_report(hits: list[AmrHit], path: str | Path) -> None:
    """Write AMR hits as a small sidecar TSV (query, gene, class, id%, cov%)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["Protein_ID", "Gene_Symbol", "Sequence_Name", "Element_Type",
             "Class", "Pct_Identity", "Pct_Coverage", "Method"]
        )
        for hit in hits:
            writer.writerow(
                [hit.query_id, hit.gene_symbol, hit.sequence_name, hit.element_type,
                 hit.gene_class, f"{hit.pct_identity:.2f}", f"{hit.pct_coverage:.2f}",
                 hit.method]
            )


def run_pipeline(
    input_fasta: str | Path,
    output_tsv: str | Path,
    esm_model: str = DEFAULT_ESM_MODEL,
    hmmer_db: str | Path | None = None,
    top_k: int = 3,
    force_fallback_embeddings: bool = False,
    go_weights: str | Path | None = None,
    go_terms: str | Path | None = None,
    extended_tsv: bool = False,
    include_disorder: bool = True,
    eval_labels: str | Path | None = None,
    eval_report: str | Path | None = None,
    go_map: str | Path | None = None,
    blend_alpha: float = 1.0,
    obo_path: str | Path | None = None,
    run_amrfinder_flag: bool = False,
    amrfinder_db: str | Path | None = None,
    amr_report: str | Path | None = None,
) -> list[AnnotationRecord]:
    """Full pipeline: parse -> clean -> ProtParam -> embed -> GO -> HMMER -> TSV.

    Args:
        input_fasta: Query proteins in FASTA format.
        output_tsv: Destination TSV path (parent dirs created as needed).
        esm_model: ESM-2 checkpoint name (default 8M, optional 650M).
        hmmer_db: Reference database FASTA for phmmer; None skips HMMER.
        top_k: Number of GO terms reported per protein.
        force_fallback_embeddings: Skip ESM-2 (offline/CI mode).
        go_weights: Trained head bundle (save_weights stem/.json); None = demo.
        go_terms: Terms-only JSON label space; ignored when go_weights given.
        extended_tsv: Write 9-column TSV (+disorder, embedding source).
        include_disorder: Compute the offline disorder signal (False = skip).
        eval_labels: Labels TSV for the CAFA eval report (None = no eval).
        eval_report: JSON sidecar path for the CAFA report (needs eval_labels).
        go_map: Target->GO map TSV enabling homology blending (needs hmmer_db).
        blend_alpha: Head weight in [0, 1]; 1.0 = head only (default).
        obo_path: go-basic.obo for GO-DAG ancestor smoothing (None = off).
        run_amrfinder_flag: Run the AMRFinderPlus AMR scan (graceful skip).
        amrfinder_db: AMRFinderPlus database dir (None = installed default).
        amr_report: Sidecar TSV path for AMR hits (needs run_amrfinder_flag).

    Returns:
        Annotation records (also written to output_tsv).
    """
    import json as _json
    import warnings as _warnings

    if not 0.0 <= blend_alpha <= 1.0:
        raise ValueError(f"blend_alpha must be in [0, 1], got {blend_alpha}")
    # 1. Parse + clean/mask every record.
    parsed = list(SeqIO.parse(str(input_fasta), "fasta"))
    if not parsed:
        raise ValueError(f"No records found in {input_fasta}")
    cleaned: list[CleanedSequence] = [clean_sequence(r.id, str(r.seq)) for r in parsed]
    # 2. ProtParam physicochemical features per sequence.
    features: list[ProtParamFeatures] = [compute_protparam_features(c) for c in cleaned]
    # 3. ESM-2 (or hashed fallback) mean-pooled embeddings.
    embedding: EmbeddingResult = embed_sequences(
        [c.cleaned for c in cleaned],
        [c.mask for c in cleaned],
        model_name=esm_model,
        force_fallback=force_fallback_embeddings,
    )
    fused_dim: int = embedding.dim + len(features[0].to_vector())
    # 4. Optional phmmer homology search (graceful skip when unavailable).
    # Keep the full hit list: top hits feed the TSV, all hits feed blending.
    all_hits: list[HmmerHit] = []
    if hmmer_db is not None:
        all_hits = run_phmmer(input_fasta, hmmer_db)
    hmmer_best: dict[str, HmmerHit] = top_hit_per_query(all_hits)
    # 5. GO multilabel head inference (random-init demo weights by default).
    classifier: GOClassifier
    if go_weights is not None:
        # Trained bundle wins; override width so 650M vectors fit stored heads.
        classifier = GOClassifier.load_weights(str(go_weights), input_dim=fused_dim)
    elif go_terms is not None:
        classifier = GOClassifier.load_weights(str(go_terms), input_dim=fused_dim)
    else:
        classifier = GOClassifier(input_dim=fused_dim)
    # 5b. Optional homology blend inputs (warn-and-skip on missing files).
    homolog_go: dict[str, dict[str, float]] | None = None
    if go_map is not None:
        try:
            mapping = parse_go_map(Path(go_map).read_text(encoding="utf-8"))
            homolog_go = homolog_scores_from_hits(all_hits, mapping, classifier.go_ids)
        except OSError as exc:
            _warnings.warn(f"Cannot read --go-map ({exc}); blending disabled.")
    obo_parents: dict[str, set[str]] | None = None
    if obo_path is not None:
        try:
            obo_parents = parse_obo_ancestors(obo_path)
        except OSError as exc:
            _warnings.warn(f"Cannot read --obo ({exc}); DAG smoothing disabled.")
    # 5c. Optional AMRFinderPlus scan (binary-gated inside run_amrfinder).
    amr_hits: list[AmrHit] = []
    if run_amrfinder_flag:
        amr_hits = run_amrfinder(input_fasta, amrfinder_db)
    records: list[AnnotationRecord] = annotate_records(
        cleaned,
        features,
        embedding,
        hmmer_best,
        classifier,
        top_k,
        include_disorder,
        homolog_go=homolog_go,
        blend_alpha=blend_alpha,
        obo_parents=obo_parents,
    )
    # 5d. Optional AMR sidecar (warn-and-skip without the scan flag).
    if amr_report is not None:
        if not run_amrfinder_flag:
            _warnings.warn("--amr-report needs --run-amrfinder; skipping AMR report.")
        else:
            write_amr_report(amr_hits, amr_report)
    # 6. Optional CAFA eval sidecar (needs labels; else warn-and-skip).
    if eval_report is not None:
        if eval_labels is None:
            _warnings.warn("--eval-report needs --eval-labels; skipping eval report.")
        else:
            prot_vectors = np.array([f.to_vector() for f in features], dtype=np.float32)
            combined = np.concatenate([embedding.embeddings.astype(np.float32), prot_vectors], axis=1)
            # Reuse the (possibly dim-rebuilt) head's label space for scoring.
            head: GOClassifier = classifier
            if head.input_dim != combined.shape[1]:
                head = GOClassifier(input_dim=combined.shape[1], go_ids=head.go_ids)
            Y_score: np.ndarray = head.predict_proba(combined)
            Y_true: np.ndarray = load_eval_labels(eval_labels, [c.record_id for c in cleaned], head.go_ids)
            report: dict = cafa_eval_report(Y_true, Y_score, go_ids=head.go_ids)
            report_path = Path(eval_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(_json.dumps(report, indent=2), encoding="utf-8")
    # 7. Write TSV (default 7 columns; 9 with --extended-tsv).
    out_path = Path(output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(EXTENDED_TSV_COLUMNS if extended_tsv else TSV_COLUMNS)
        for record in records:
            writer.writerow(record.to_row(extended=extended_tsv))
    return records
