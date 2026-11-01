"""End-to-end annotation pipeline: FASTA in, TSV annotations out."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio import SeqIO

from .embeddings import DEFAULT_ESM_MODEL, EmbeddingResult, embed_sequences
from .hmmer import HmmerHit, run_phmmer, top_hit_per_query
from .model import GO_TERMS, GOClassifier
from .sequence import CleanedSequence, ProtParamFeatures, clean_sequence, compute_protparam_features

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

    def to_row(self) -> list[str]:
        """Serialise to TSV_COLUMNS order with 2/3-decimal formatting."""
        return [
            self.protein_id,
            str(self.length),
            f"{self.molecular_weight:.2f}",
            ";".join(self.top_go_terms),
            ";".join(f"{s:.3f}" for s in self.confidence_scores),
            self.hmmer_hit,
            self.predicted_function,
        ]


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
) -> list[AnnotationRecord]:
    """Combine ProtParam + embeddings + GO head + HMMER into records."""
    # Model input = ESM/hashed embedding concatenated with ProtParam vector.
    prot_vectors = np.array([f.to_vector() for f in features], dtype=np.float32)
    combined = np.concatenate([embedding.embeddings.astype(np.float32), prot_vectors], axis=1)
    # Rebuild classifier head if the fused width differs (e.g. 650M model).
    if classifier.input_dim != combined.shape[1]:
        classifier = GOClassifier(input_dim=combined.shape[1], thresholds=classifier.thresholds)
    top_terms, top_scores = classifier.predict(combined, top_k=top_k)
    records: list[AnnotationRecord] = []
    for i, seq in enumerate(cleaned):
        hit: HmmerHit | None = hmmer_best.get(seq.record_id)
        hit_str: str = hit.short() if hit else "-"
        records.append(
            AnnotationRecord(
                protein_id=seq.record_id,
                length=features[i].length,
                molecular_weight=features[i].molecular_weight,
                top_go_terms=top_terms[i],
                confidence_scores=top_scores[i],
                hmmer_hit=hit_str,
                predicted_function=describe_function(top_terms[i], top_scores[i], hit_str),
            )
        )
    return records


def run_pipeline(
    input_fasta: str | Path,
    output_tsv: str | Path,
    esm_model: str = DEFAULT_ESM_MODEL,
    hmmer_db: str | Path | None = None,
    top_k: int = 3,
    force_fallback_embeddings: bool = False,
) -> list[AnnotationRecord]:
    """Full pipeline: parse -> clean -> ProtParam -> embed -> GO -> HMMER -> TSV.

    Args:
        input_fasta: Query proteins in FASTA format.
        output_tsv: Destination TSV path (parent dirs created as needed).
        esm_model: ESM-2 checkpoint name (default 8M, optional 650M).
        hmmer_db: Reference database FASTA for phmmer; None skips HMMER.
        top_k: Number of GO terms reported per protein.
        force_fallback_embeddings: Skip ESM-2 (offline/CI mode).

    Returns:
        Annotation records (also written to output_tsv).
    """
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
    # 4. Optional phmmer homology search (graceful skip when unavailable).
    hmmer_best: dict[str, HmmerHit] = {}
    if hmmer_db is not None:
        hmmer_best = top_hit_per_query(run_phmmer(input_fasta, hmmer_db))
    # 5. GO multilabel head inference (random-init demo weights by default).
    classifier = GOClassifier(input_dim=embedding.dim + len(features[0].to_vector()))
    records: list[AnnotationRecord] = annotate_records(cleaned, features, embedding, hmmer_best, classifier, top_k)
    # 6. Write TSV with the fixed contract columns.
    out_path = Path(output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(TSV_COLUMNS)
        for record in records:
            writer.writerow(record.to_row())
    return records
