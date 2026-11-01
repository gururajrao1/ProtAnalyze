"""Tests for the top-4 shortlist specs (VFDB, GO labels, CAFA eval, disorder)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

# Make src/ importable without packaging (repo root is parent of tests/).
ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_annotation import embeddings, go_labels, hmmer, model, sequence  # noqa: E402
from protein_annotation.cli import build_parser  # noqa: E402
from protein_annotation.pipeline import (  # noqa: E402
    EXTENDED_TSV_COLUMNS,
    TSV_COLUMNS,
    run_pipeline,
)

SAMPLE_FASTA: Path = ROOT / "hypothetical_proteins.fasta"

# Tiny GAF 2.2 fixture: 3 bacterial EXP rows (one NOT row dropped),
# 1 IEA row dropped, 1 human row dropped by the taxon filter.
GAF_FIXTURE: str = """!gaf-version: 2.2
UniProtKB\tP00001\tprotA\t\tGO:0003824\tREF\tIDA\t\tF\t\t\tprotein\ttaxon:243274\t20240101\tUniProt\t\t
UniProtKB\tP00001\tprotA\tNOT\tGO:0003677\tREF\tIDA\t\tF\t\t\tprotein\ttaxon:243274\t20240101\tUniProt\t\t
UniProtKB\tP00002\tprotB\t\tGO:0003824\tREF\tIDA\t\tF\t\t\tprotein\ttaxon:243277\t20240101\tUniProt\t\t
UniProtKB\tP00002\tprotB\t\tGO:0005524\tREF\tIEA\t\tF\t\t\tprotein\ttaxon:243277\t20240101\tUniProt\t\t
UniProtKB\tP00003\tprotC\t\tGO:0003824\tREF\tIDA\t\tF\t\t\tprotein\ttaxon:9606\t20240101\tUniProt\t\t
UniProtKB\tP00004\tprotD\t\tGO:0005524\tREF\tIDA\t\tF\t\t\tprotein\ttaxon:243274\t20240101\tUniProt\t\t
"""

# Tiny OBO fixture: 3824 is_a 3677; 5524 is obsolete (must not propagate).
OBO_FIXTURE: str = """format-version: 1.2
[Term]
id: GO:0003824
name: catalytic activity
is_a: GO:0003677 ! DNA binding
[Term]
id: GO:0003677
name: DNA binding
[Term]
id: GO:0005524
name: ATP binding
is_obsolete: true
"""


def test_disorder_ordering_and_bounds() -> None:
    """Poly-K/E/P (disordered) scores far above globular hydrophobic sequence."""
    disordered = sequence.disorder_fraction("KKKEEEPPPSSQQQKKKEEEPPPSSQQQ")
    globular = sequence.disorder_fraction("AILMFWYVAILMFWYVAILMFWYVAILMF")
    assert 0.0 <= globular <= 1.0
    assert disordered > 0.8
    assert globular < 0.2
    assert disordered > globular
    # Per-residue scores align 1:1 with the sequence.
    assert len(sequence.disorder_scores("MKTAY")) == 5
    assert sequence.disorder_fraction("") == 0.0


def test_embedding_fallback_flag() -> None:
    """Forced fallback carries the hashed-fallback provenance flag."""
    result = embeddings.embed_sequences(["MKTAYIAKQRQ"], force_fallback=True)
    assert result.source == "hashed-fallback"


def test_pipeline_records_carry_offline_signals(tmp_path: Path) -> None:
    """Records carry disordered_fraction + embedding_source; TSV stays 7-col."""
    out_tsv: Path = tmp_path / "ann.tsv"
    records = run_pipeline(SAMPLE_FASTA, out_tsv, force_fallback_embeddings=True)
    assert all(r.disordered_fraction is not None for r in records)
    assert all(0.0 <= r.disordered_fraction <= 1.0 for r in records)  # type: ignore[operator]
    assert all(r.embedding_source == "hashed-fallback" for r in records)
    with out_tsv.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert reader.fieldnames == TSV_COLUMNS  # Backward compatible default.


def test_extended_tsv_and_no_disorder(tmp_path: Path) -> None:
    """--extended-tsv writes 11 cols; --no-disorder blanks the disorder cell."""
    out_ext: Path = tmp_path / "ext.tsv"
    records = run_pipeline(SAMPLE_FASTA, out_ext, force_fallback_embeddings=True, extended_tsv=True)
    assert len(records) == 6
    with out_ext.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert reader.fieldnames == EXTENDED_TSV_COLUMNS
        rows = list(reader)
    assert rows[0]["Disordered_Fraction"] != "-"
    assert rows[0]["Embedding_Source"] == "hashed-fallback"
    # Skip path: disorder is None on records and "-" in the extended TSV.
    out_skip: Path = tmp_path / "skip.tsv"
    skipped = run_pipeline(
        SAMPLE_FASTA, out_skip, force_fallback_embeddings=True, extended_tsv=True, include_disorder=False
    )
    assert all(r.disordered_fraction is None for r in skipped)
    with out_skip.open(encoding="utf-8") as handle:
        rows2 = list(csv.DictReader(handle, delimiter="\t"))
    assert rows2[0]["Disordered_Fraction"] == "-"


def test_cafa_report_perfect_scores() -> None:
    """Perfect synthetic scores yield Fmax=1, Smin=0, AUPR=1."""
    rng = np.random.RandomState(7)
    Y_true = (rng.rand(30, 4) > 0.5).astype(int)
    Y_score = Y_true.astype(float) * 0.98 + 0.01  # Near-perfect, ordered correctly.
    report = model.cafa_eval_report(Y_true, Y_score, go_ids=["T0", "T1", "T2", "T3"])
    assert report["fmax"] == 1.0
    assert report["smin"] == 0.0
    assert report["aupr_micro"] == 1.0
    assert len(report["per_term"]) == 4
    assert all(t["f1"] == 1.0 for t in report["per_term"])
    # Random scores must score strictly worse than perfect.
    noisy = model.cafa_eval_report(Y_true, rng.rand(30, 4))
    assert noisy["fmax"] < 1.0


def test_eval_report_sidecar_via_pipeline(tmp_path: Path) -> None:
    """run_pipeline writes the CAFA JSON sidecar when labels are provided."""
    labels = tmp_path / "labels.tsv"
    labels.write_text(
        "hypo_protein_1\tGO:0003824,GO:0003677\n"
        "hypo_protein_2\tGO:0006412\n"
        "unknown_protein\tGO:0006810\n",  # Unknown id: skipped, no crash.
        encoding="utf-8",
    )
    out_tsv = tmp_path / "ann.tsv"
    sidecar = tmp_path / "eval.json"
    run_pipeline(
        SAMPLE_FASTA,
        out_tsv,
        force_fallback_embeddings=True,
        eval_labels=labels,
        eval_report=sidecar,
    )
    report = json.loads(sidecar.read_text(encoding="utf-8"))
    assert report["n_proteins"] == 6
    assert report["n_terms"] == len(model.GO_IDS)
    assert 0.0 <= report["fmax"] <= 1.0
    assert "smin" in report and "aupr_micro" in report


def test_eval_report_without_labels_warns_and_skips(tmp_path: Path) -> None:
    """--eval-report without --eval-labels warns gracefully (no crash/file)."""
    import warnings

    out_tsv = tmp_path / "ann.tsv"
    sidecar = tmp_path / "eval.json"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_pipeline(SAMPLE_FASTA, out_tsv, force_fallback_embeddings=True, eval_report=sidecar)
    assert any("eval-labels" in str(w.message) for w in caught)
    assert not sidecar.exists()


def test_go_labels_parse_propagate_prune(tmp_path: Path) -> None:
    """GAF filters (NOT/IEA/taxon) + OBO propagation + pruning on fixtures."""
    gaf = tmp_path / "mini.gaf"
    obo = tmp_path / "mini.obo"
    gaf.write_text(GAF_FIXTURE, encoding="utf-8")
    obo.write_text(OBO_FIXTURE, encoding="utf-8")
    labels = go_labels.parse_gaf(gaf)
    # NOT row, IEA row and human row dropped; obsolete term kept as direct only.
    assert labels == {"P00001": {"GO:0003824"}, "P00002": {"GO:0003824"}, "P00004": {"GO:0005524"}}
    parents = go_labels.parse_obo_ancestors(obo)
    assert parents["GO:0003824"] == {"GO:0003677"}  # is_a edge kept.
    assert "GO:0005524" not in parents  # Obsolete term excluded.
    full = go_labels.propagate(labels, parents)
    assert full["P00001"] == {"GO:0003824", "GO:0003677"}  # Ancestor added.
    kept = go_labels.prune_terms(full, min_count=2, max_terms=500)
    assert set(kept) == {"GO:0003824", "GO:0003677"}  # Both pass min_count=2; tie order free.
    accessions, matrix = go_labels.build_label_matrix(full, kept)
    assert accessions == ["P00001", "P00002", "P00004"]
    assert matrix[0] == [1, 1] and matrix[2] == [0, 0]
    # Terms JSON round-trips through the head loader (random-init head).
    terms_json = tmp_path / "terms.json"
    go_labels.save_terms_json(kept, terms_json, counts={"GO:0003824": 2, "GO:0003677": 2})
    clf = model.GOClassifier.load_weights(str(terms_json), input_dim=16)
    assert clf.go_ids == kept
    proba = clf.predict_proba(np.zeros((2, 16), dtype=np.float32))
    assert proba.shape == (2, 2)


def test_go_classifier_save_load_roundtrip(tmp_path: Path) -> None:
    """save_weights/load_weights preserves thresholds, label space and scores."""
    import torch  # noqa: F401  # Requires torch (installed per requirements).

    rng = np.random.RandomState(3)
    clf = model.GOClassifier(input_dim=16)
    X = rng.randn(12, 16).astype(np.float32)
    Y = (rng.rand(12, len(model.GO_IDS)) > 0.6).astype(np.float32)
    clf.fit(X, Y, X, Y, epochs=2)
    stem = tmp_path / "head"
    clf.save_weights(str(stem))
    assert (tmp_path / "head.json").exists()
    assert (tmp_path / "head.pt").exists()
    loaded = model.GOClassifier.load_weights(str(stem))
    assert loaded.go_ids == clf.go_ids
    assert loaded.thresholds == clf.thresholds
    assert np.allclose(loaded.predict_proba(X), clf.predict_proba(X))


def test_go_terms_flag_changes_label_space(tmp_path: Path) -> None:
    """--go-terms JSON flows through Top_GO_Terms in the pipeline TSV."""
    terms_json = tmp_path / "terms.json"
    terms_json.write_text(json.dumps(["GO:0003824", "GO:0003677"]), encoding="utf-8")
    out_tsv = tmp_path / "ann.tsv"
    records = run_pipeline(SAMPLE_FASTA, out_tsv, force_fallback_embeddings=True, go_terms=terms_json)
    assert all(set(r.top_go_terms) <= {"GO:0003824", "GO:0003677"} for r in records)


def test_vfdb_tblout_fixture_and_fetch_script() -> None:
    """VFDB-style phmmer hits parse + fetch scripts are documented."""
    tbl: str = (
        "# phmmer :: VFDB_setA_pro reference search\n"
        "VFG000001 - hypo_protein_1 - 3e-25 180.2 0.0 1.0 1 1 1 1 1 1 fimH adhesin\n"
        "VFG000042 - hypo_protein_1 - 1e-06 55.4 0.0 1.0 1 1 1 1 1 1 toxin subunit\n"
    )
    hits = hmmer.parse_tblout(tbl)
    assert len(hits) == 2
    best = hmmer.top_hit_per_query(hits)
    assert best["hypo_protein_1"].target_id == "VFG000001"  # Best e-value wins.
    assert "VFG000001" in best["hypo_protein_1"].short()
    # Fetch helpers exist for both shells (Windows host + bash).
    assert (ROOT / "scripts" / "fetch_vfdb.sh").exists()
    assert (ROOT / "scripts" / "fetch_vfdb.ps1").exists()


def test_cli_new_flags() -> None:
    """New CLI flags parse with backward-compatible defaults."""
    args = build_parser().parse_args(["in.fasta"])
    assert args.go_weights is None
    assert args.go_terms is None
    assert args.extended_tsv is False
    assert args.no_disorder is False
    assert args.eval_labels is None
    assert args.eval_report is None
    args2 = build_parser().parse_args(
        ["in.fasta", "--go-weights", "head", "--extended-tsv", "--no-disorder",
         "--eval-labels", "l.tsv", "--eval-report", "r.json"]
    )
    assert args2.go_weights == "head"
    assert args2.extended_tsv is True
    assert args2.no_disorder is True
    assert args2.eval_report == "r.json"
