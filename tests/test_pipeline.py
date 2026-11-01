"""Smoke tests for the hypothetical-protein annotation pipeline."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# Make src/ importable without packaging (repo root is parent of tests/).
ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_annotation import embeddings, hmmer, model, sequence  # noqa: E402
from protein_annotation.cli import build_parser  # noqa: E402
from protein_annotation.pipeline import TSV_COLUMNS, run_pipeline  # noqa: E402

SAMPLE_FASTA: Path = ROOT / "hypothetical_proteins.fasta"


def test_clean_sequence_masks_ambiguous() -> None:
    """X/B/Z/U/O positions are substituted and marked False in the mask."""
    cleaned = sequence.clean_sequence("t1", "ACDXBZUO")
    # All output residues must be standard amino acids.
    assert set(cleaned.cleaned) <= set(sequence.STANDARD_AA)
    assert len(cleaned.cleaned) == len(cleaned.mask) == 8
    # First three (ACD) valid; last five (X/B/Z/U/O) masked.
    assert cleaned.mask[:3] == [True, True, True]
    assert cleaned.mask[3:] == [False] * 5
    assert cleaned.n_masked == 5


def test_protparam_features_keys() -> None:
    """ProtParam bundle exposes length, MW, pI, aliphatic and 27-dim vector."""
    cleaned = sequence.clean_sequence("t2", "MKTAYIAKQRQISFVKSHFSRQDILDLW")
    feats = sequence.compute_protparam_features(cleaned)
    assert feats.length == len(cleaned.cleaned)
    assert feats.molecular_weight > 0.0
    assert 0.0 < feats.isoelectric_point < 14.0
    assert feats.aliphatic >= 0.0
    assert len(feats.to_vector()) == 27  # 7 globals + 20 AA freqs.


def test_embeddings_fallback_shape() -> None:
    """Forced fallback yields deterministic (n, 320) L2-normalised vectors."""
    seqs: list[str] = ["MKTAYIAKQRQ", "GAGAGAGAGA"]
    first = embeddings.embed_sequences(seqs, force_fallback=True)
    second = embeddings.embed_sequences(seqs, force_fallback=True)
    assert first.source == "hashed-fallback"
    assert first.embeddings.shape == (2, 320)
    # Determinism: same input gives identical vectors.
    assert np.allclose(first.embeddings, second.embeddings)
    # L2 normalisation check per row.
    assert np.allclose(np.linalg.norm(first.embeddings, axis=1), 1.0)


def test_hmmer_parse_tblout() -> None:
    """tblout parser extracts query/target/e-value/bitscore, best-first."""
    tbl: str = (
        "# phmmer :: search a protein sequence against a protein database\n"
        "targetA - query1 - 1e-20 150.3 0.0 1.0 1 1 1 1 1 1 target desc\n"
        "targetB - query1 - 2e-05 45.1 0.0 1.0 1 1 1 1 1 1 other desc\n"
    )
    hits = hmmer.parse_tblout(tbl)
    assert len(hits) == 2
    assert hits[0].target_id == "targetA"  # Lowest e-value first.
    assert hits[0].evalue == 1e-20
    assert hits[0].bitscore == 45.1 or hits[1].bitscore == 45.1
    best = hmmer.top_hit_per_query(hits)
    assert set(best) == {"query1"}
    assert best["query1"].target_id == "targetA"


def test_hmmer_missing_binary_graceful(tmp_path: Path) -> None:
    """run_phmmer returns [] (no raise) when no database/binary is usable."""
    import warnings

    # Point at the sample fasta as both query and db; without phmmer this
    # must warn-and-skip. If phmmer exists, an empty/invalid db still parses
    # to a list (possibly hits) — either way a list is returned.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hits = hmmer.run_phmmer(SAMPLE_FASTA, tmp_path / "nodb.fasta")
    assert isinstance(hits, list)


def test_model_forward_and_threshold_tuning() -> None:
    """GO head emits (n, 12) probabilities; tuning returns 12 thresholds."""
    import torch  # noqa: F401  # Requires torch (installed per requirements).

    clf = model.GOClassifier(input_dim=32)
    X = np.random.RandomState(0).randn(6, 32).astype(np.float32)
    proba = clf.predict_proba(X)
    assert proba.shape == (6, len(model.GO_IDS))
    assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0
    # Synthetic labels for threshold tuning sweep.
    rng = np.random.RandomState(1)
    Y_true = (rng.rand(20, len(model.GO_IDS)) > 0.7).astype(int)
    Y_score = rng.rand(20, len(model.GO_IDS))
    tuned = model.tune_thresholds(Y_true, Y_score)
    assert len(tuned) == len(model.GO_IDS)
    assert all(0.09 <= t <= 0.91 for t in tuned)


def test_cli_parser_defaults() -> None:
    """argparse CLI exposes required input + ESM/HMMER options."""
    args = build_parser().parse_args(["in.fasta"])
    assert args.input_fasta == "in.fasta"
    assert args.esm_model == embeddings.DEFAULT_ESM_MODEL
    assert args.hmmer_db is None
    args2 = build_parser().parse_args(["in.fasta", "--esm-model", "esm2_t33_650M_UR50D", "--no-esm"])
    assert args2.esm_model == "esm2_t33_650M_UR50D"
    assert args2.no_esm is True


def test_end_to_end_offline(tmp_path: Path) -> None:
    """Full pipeline on the sample FASTA writes a valid 6-row TSV."""
    out_tsv: Path = tmp_path / "protein_annotations.tsv"
    records = run_pipeline(SAMPLE_FASTA, out_tsv, force_fallback_embeddings=True)
    assert len(records) == 6  # Sample file has six synthetic proteins.
    assert out_tsv.exists()
    with out_tsv.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert reader.fieldnames == TSV_COLUMNS
        rows = list(reader)
    assert len(rows) == 6
    # Spot-check ambiguous-sequence record survived cleaning.
    by_id = {r["Protein_ID"]: r for r in rows}
    assert "hypo_protein_4" in by_id
    assert float(by_id["hypo_protein_4"]["Molecular_Weight"]) > 0.0
    assert by_id["hypo_protein_4"]["Top_GO_Terms"] != ""
