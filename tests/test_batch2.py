"""Tests for batch-2 signals: AMR wrapper, homology blend, DAG smoothing."""

from __future__ import annotations

import csv
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

# Make src/ importable without packaging (repo root is parent of tests/).
ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_annotation import amr, embeddings, model, sequence  # noqa: E402
from protein_annotation.cli import build_parser  # noqa: E402
from protein_annotation.hmmer import HmmerHit  # noqa: E402
from protein_annotation.model import GO_IDS  # noqa: E402
from protein_annotation.pipeline import (  # noqa: E402
    TSV_COLUMNS,
    annotate_records,
    run_pipeline,
    write_amr_report,
)

SAMPLE_FASTA: Path = ROOT / "hypothetical_proteins.fasta"

# Tiny target->GO map: VFG-style target ids to demo GO terms.
GO_MAP_FIXTURE: str = """# target -> GO map
VFG000001\tGO:0003824,GO:0003677
VFG000002\tGO:0005524
"""

# Tiny OBO parents fixture: 3824 is_a 3677 (child -> parent).
OBO_PARENTS: dict[str, set[str]] = {"GO:0003824": {"GO:0003677"}}

# AMRFinderPlus-style tabular fixture (real column spellings, 2 hits).
AMR_FIXTURE: str = (
    "Protein id\tContig id\tStart\tStop\tStrand\tGene symbol\tSequence name\tScope\t"
    "Element type\tElement subtype\tClass\tSubclass\tMethod\tTarget length\t"
    "Reference sequence length\t% Coverage of reference sequence\t"
    "% Identity to reference sequence\tAlignment length\t"
    "Accession of closest sequence\tName of closest sequence\tHMM id\tHMM description\n"
    "protA\tcontig1\t10\t300\t+\tblaTEM-1\tblaTEM-1\tcore\tAMR\tAMR\tBETA-LACTAM\t"
    "CEPHALOSPORIN\tALIGNMENT\t290\t290\t100.00\t99.50\t290\tWP_1\tblaTEM\t\t\n"
    "protB\tcontig1\t400\t700\t+\ttet(A)\ttet(A)\tplus\tSTRESS\tBIOCIDE\tTETRACYCLINE\t"
    "\tEXACT\t300\t300\t100.00\t100.00\t300\tWP_2\ttetA\t\t\n"
)


def test_parse_go_map_skips_comments_and_bad_lines() -> None:
    """Map parser keeps good rows, warns (not raises) on malformed ones."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mapping = model.parse_go_map(GO_MAP_FIXTURE + "broken-line-no-tab\n")
    assert mapping == {
        "VFG000001": {"GO:0003824", "GO:0003677"},
        "VFG000002": {"GO:0005524"},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.parse_go_map("no-tab-here\n")
    assert len(caught) == 1


def test_homolog_transfer_normalises_bitscores() -> None:
    """Best mapped hit votes 1.0; weaker hits vote proportionally less."""
    hits = [
        HmmerHit("q1", "VFG000001", 1e-20, 200.0),
        HmmerHit("q1", "VFG000002", 1e-5, 100.0),
        HmmerHit("q1", "UNMAPPED", 1e-30, 999.0),  # Ignored: not in map.
        HmmerHit("q2", "UNMAPPED", 1e-30, 999.0),  # No mapped hits: abstains.
    ]
    mapping = model.parse_go_map(GO_MAP_FIXTURE)
    scores = model.homolog_scores_from_hits(hits, mapping, GO_IDS)
    assert set(scores) == {"q1"}  # q2 abstained.
    assert scores["q1"]["GO:0003824"] == 1.0  # Best hit votes full weight.
    assert scores["q1"]["GO:0003677"] == 1.0  # Same hit, second term.
    assert scores["q1"]["GO:0005524"] == 0.5  # Half the best bitscore.
    # Out-of-space terms never leak in.
    mapping["VFG000001"].add("GO:9999999")
    assert "GO:9999999" not in model.homolog_scores_from_hits(hits, mapping, GO_IDS)["q1"]


def test_blend_math_and_alpha_validation() -> None:
    """Convex combination is exact; alpha outside [0,1] raises."""
    head = np.array([0.2, 0.8] + [0.0] * (len(GO_IDS) - 2))
    homolog = {GO_IDS[0]: 1.0, GO_IDS[1]: 0.0}
    blended = model.blend_scores(head, homolog, GO_IDS, alpha=0.5)
    assert blended[0] == 0.6  # 0.5*0.2 + 0.5*1.0
    assert blended[1] == 0.4  # 0.5*0.8 + 0.5*0.0
    untouched = model.blend_scores(head, {}, GO_IDS, alpha=1.0)
    assert np.allclose(untouched, head)  # alpha=1 reproduces the head.
    try:
        model.blend_scores(head, homolog, GO_IDS, alpha=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("alpha=1.5 must raise ValueError")


def test_smooth_over_ancestors_raises_parents() -> None:
    """Child score propagates up; unrelated terms pass through unchanged."""
    scores = np.zeros(len(GO_IDS))
    scores[GO_IDS.index("GO:0003824")] = 0.9  # Child high...
    scores[GO_IDS.index("GO:0003677")] = 0.1  # ...parent low.
    scores[GO_IDS.index("GO:0005524")] = 0.3  # Unrelated control.
    smoothed = model.smooth_over_ancestors(scores, GO_IDS, OBO_PARENTS)
    assert smoothed[GO_IDS.index("GO:0003677")] == 0.9  # Raised to child.
    assert smoothed[GO_IDS.index("GO:0003824")] == 0.9  # Child kept.
    assert smoothed[GO_IDS.index("GO:0005524")] == 0.3  # Untouched.
    # Empty parents map is a strict no-op.
    assert np.allclose(model.smooth_over_ancestors(scores, GO_IDS, {}), scores)


def test_annotate_records_blend_is_deterministic() -> None:
    """alpha=0 makes homolog votes decide the top term (no RNG involved)."""
    cleaned_seq = sequence.clean_sequence("p1", "MKTAYIAKQRQISFVKSHFSRQDILDL")
    feats = sequence.compute_protparam_features(cleaned_seq)
    emb = embeddings.embed_sequences([cleaned_seq.cleaned], force_fallback=True)
    clf = model.GOClassifier(input_dim=emb.dim + len(feats.to_vector()))
    homolog = {"p1": {"GO:0005524": 1.0}}
    records = annotate_records(
        [cleaned_seq], [feats], emb, {}, clf, top_k=1, homolog_go=homolog, blend_alpha=0.0
    )
    assert records[0].top_go_terms == ["GO:0005524"]
    assert records[0].confidence_scores == [1.0]
    assert records[0].blended is True
    # Default path stays unblended.
    plain = annotate_records([cleaned_seq], [feats], emb, {}, clf, top_k=1)
    assert plain[0].blended is False


def test_amr_parser_reads_fixture() -> None:
    """Header-driven parser extracts gene/class/identity/coverage."""
    hits = amr.parse_amr_tsv(AMR_FIXTURE)
    assert len(hits) == 2
    assert hits[0].query_id == "protA"
    assert hits[0].gene_symbol == "blaTEM-1"
    assert hits[0].gene_class == "BETA-LACTAM"
    assert hits[0].pct_identity == 99.5
    assert hits[0].pct_coverage == 100.0
    assert hits[1].element_type == "STRESS"
    assert amr.parse_amr_tsv("") == []  # Empty output -> no hits.
    grouped = amr.hits_per_query(hits)
    assert set(grouped) == {"protA", "protB"}


def test_amr_graceful_skip_without_binary(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Missing binary warns and returns [] instead of raising."""
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    assert amr.amr_available() is False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert amr.run_amrfinder(SAMPLE_FASTA) == []
    assert any("amrfinder" in str(w.message).lower() for w in caught)


def test_amr_report_sidecar_roundtrip(tmp_path: Path) -> None:
    """Sidecar writer emits the 8-column header + one row per hit."""
    out: Path = tmp_path / "amr.tsv"
    write_amr_report(amr.parse_amr_tsv(AMR_FIXTURE), out)
    with out.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    assert rows[0][0] == "Protein_ID" and len(rows[0]) == 8
    assert rows[1][1] == "blaTEM-1"
    assert len(rows) == 3  # Header + 2 hits.


def test_pipeline_blend_and_obo_flags(tmp_path: Path) -> None:
    """--go-map/--obo run offline; smoothing marks records blended."""
    gomap: Path = tmp_path / "map.tsv"
    gomap.write_text(GO_MAP_FIXTURE, encoding="utf-8")
    obo: Path = tmp_path / "mini.obo"
    obo.write_text(
        "format-version: 1.2\n[Term]\nid: GO:0003824\nis_a: GO:0003677\n"
        "[Term]\nid: GO:0003677\n",
        encoding="utf-8",
    )
    out: Path = tmp_path / "ann.tsv"
    records = run_pipeline(
        SAMPLE_FASTA,
        out,
        force_fallback_embeddings=True,
        go_map=gomap,  # No phmmer here: homolog arm abstains, must not crash.
        blend_alpha=0.5,
        obo_path=obo,
    )
    assert len(records) == 6
    assert all(r.blended for r in records)  # Smoothing alone sets the flag.
    with out.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    assert header == TSV_COLUMNS  # Default TSV contract untouched.


def test_pipeline_amr_warn_and_skip(tmp_path: Path) -> None:
    """--amr-report without the scan flag warns; scan without binary still runs."""
    out: Path = tmp_path / "ann.tsv"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_pipeline(
            SAMPLE_FASTA, out, force_fallback_embeddings=True,
            amr_report=tmp_path / "amr.tsv",
        )
    assert any("--run-amrfinder" in str(w.message) for w in caught)
    # Scan requested but binary absent: pipeline still completes.
    out2: Path = tmp_path / "ann2.tsv"
    sidecar: Path = tmp_path / "amr2.tsv"
    records = run_pipeline(
        SAMPLE_FASTA, out2, force_fallback_embeddings=True,
        run_amrfinder_flag=True, amr_report=sidecar,
    )
    assert len(records) == 6
    assert sidecar.exists()  # Header-only sidecar, honest empty result.


def test_cli_exposes_batch2_flags() -> None:
    """All batch-2 flags parse with the documented defaults."""
    args = build_parser().parse_args(["in.fasta"])
    assert args.go_map is None and args.blend_alpha == 1.0 and args.obo_path is None
    assert args.run_amrfinder is False and args.amr_report is None
    args = build_parser().parse_args(
        ["in.fasta", "--go-map", "m.tsv", "--blend-alpha", "0.3", "--obo", "g.obo",
         "--run-amrfinder", "--amr-report", "a.tsv"]
    )
    assert (args.go_map, args.blend_alpha, args.obo_path) == ("m.tsv", 0.3, "g.obo")
    assert args.run_amrfinder is True and args.amr_report == "a.tsv"
