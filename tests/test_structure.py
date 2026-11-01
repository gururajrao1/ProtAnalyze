"""Tests for the SS3 secondary-structure niche (structure.py + wiring)."""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable without packaging (repo root is parent of tests/).
ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_annotation import sequence, structure  # noqa: E402
from protein_annotation.cli import build_parser  # noqa: E402
from protein_annotation.pipeline import EXTENDED_TSV_COLUMNS, run_pipeline  # noqa: E402
from protein_annotation.structure import (  # noqa: E402
    SS3_LABELS,
    SSHeadWeights,
    load_ss_head,
    predict_ss3,
    predict_ss3_head,
    predict_ss3_heuristic,
    ss_accuracy,
    ss_composition,
    ss_source,
)

SS_MINI_FASTA: Path = Path(__file__).resolve().parent / "data" / "ss_mini.fasta"
PINNED_JSON: Path = Path(__file__).resolve().parent / "data" / "ss_pinned20.json"
PINNED_NPZ: Path = Path(__file__).resolve().parent / "data" / "ss_pinned20_reps.npz"
HEAD_JSON: Path = ROOT / "data" / "ss_head_8m.json"

# Falsifiable bar for the shipped head on the pinned held-out fixture.
Q3_BAR: float = 0.70


def test_heuristic_length_alphabet_and_determinism() -> None:
    """Heuristic emits one H/E/C per residue, deterministically."""
    seq = "MKTAYIAKQRQISFVKSHFSRQDILDLWQKETLQKAVAGAPYKVTVLLVGDGAGKTTLLKRL"
    first = predict_ss3_heuristic(seq)
    second = predict_ss3_heuristic(seq)
    assert len(first) == len(seq)
    assert set(first) <= set(SS3_LABELS)
    assert first == second
    assert predict_ss3_heuristic("") == []
    # Helix-prone poly-alanine/glutamate stretch should not be all coil.
    helix_prone = predict_ss3_heuristic("AAAAAEEEEAAAAAEEEEAAAAA")
    assert "H" in helix_prone


def test_heuristic_fallback_when_head_missing() -> None:
    """predict_ss3 without weights/reps equals the heuristic; bad path warns."""
    seq = "MKTAYIAKQRQISFVKSHFSRQDILDLWQK"
    assert predict_ss3(seq) == predict_ss3_heuristic(seq)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert load_ss_head("/nonexistent/ss_head.json") is None
    assert any("heuristic" in str(w.message) for w in caught)
    assert ss_source(None) == "heuristic:chou-fasman"
    comp = ss_composition(predict_ss3(seq))
    assert set(comp) == {"H", "E", "C"}
    assert comp["H"] + comp["E"] + comp["C"] == pytest.approx(1.0)


def test_head_bundle_roundtrip_and_forward_shape(tmp_path: Path) -> None:
    """Head JSON validates shapes; numpy forward matches residue count."""
    rng = np.random.RandomState(0)
    head = SSHeadWeights(
        dim=8, window=1, labels=list(SS3_LABELS),
        weights=rng.randn(8 * 3, 3).tolist(), bias=[0.1, 0.2, 0.3],
    )
    path = head.save(tmp_path / "head.json")
    loaded = SSHeadWeights.from_json(path.read_text(encoding="utf-8"))
    assert (loaded.dim, loaded.window) == (8, 1)
    out = predict_ss3_head(rng.randn(25, 8).astype(np.float32), loaded)
    assert len(out) == 25
    assert set(out) <= set(SS3_LABELS)
    # Corrupt bundles raise instead of silently changing the label space.
    with pytest.raises(ValueError):
        SSHeadWeights.from_json('{"labels": ["H", "E"], "dim": 8, "window": 1,'
                                ' "weights": [], "bias": []}')


def test_q3_on_pinned_fixture() -> None:
    """Shipped head scores Q3 >= 0.70 on the pinned held-out fixture.

    Hermetic: numpy-only forward over stored fp16 ESM reps (no torch/ESM
    needed), so this runs CPU-only in seconds.
    """
    head = load_ss_head(HEAD_JSON)
    assert head is not None
    pinned = json.loads(PINNED_JSON.read_text(encoding="utf-8"))
    assert len(pinned) == 20
    archive = np.load(PINNED_NPZ)
    pred_all: list[str] = []
    true_all: list[str] = []
    for k, row in enumerate(pinned):
        rep = archive[f"rep_{k}"].astype(np.float64)
        assert rep.shape[0] == len(row["seq"])  # Reps cover the sequence.
        pred = predict_ss3_head(rep, head)
        assert len(pred) == len(row["ss3"])
        pred_all.extend(pred)
        true_all.extend(list(row["ss3"]))
    q3 = ss_accuracy(pred_all, true_all)
    print(f"\npinned20 Q3 = {q3:.4f} over {len(true_all)} residues")
    assert q3 >= Q3_BAR


def test_extended_tsv_ss_columns(tmp_path: Path) -> None:
    """--extended-tsv carries parseable SS helix/sheet fractions."""
    out: Path = tmp_path / "ss.tsv"
    records = run_pipeline(SS_MINI_FASTA, out, force_fallback_embeddings=True, extended_tsv=True)
    assert len(records) == 3
    assert "SS_Helix_Fraction" in EXTENDED_TSV_COLUMNS
    assert "SS_Sheet_Fraction" in EXTENDED_TSV_COLUMNS
    with out.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows[0]) == len(EXTENDED_TSV_COLUMNS) == 11
    for row, rec in zip(rows, records):
        assert float(row["SS_Helix_Fraction"]) == pytest.approx(rec.ss_helix)
        assert float(row["SS_Sheet_Fraction"]) == pytest.approx(rec.ss_sheet)
        assert 0.0 <= float(row["SS_Helix_Fraction"]) <= 1.0
        assert 0.0 <= float(row["SS_Sheet_Fraction"]) <= 1.0
    # Skip path: --no-ss leaves fractions None and TSV cells "-".
    out_skip: Path = tmp_path / "skip.tsv"
    skipped = run_pipeline(
        SS_MINI_FASTA, out_skip, force_fallback_embeddings=True,
        extended_tsv=True, include_ss=False,
    )
    assert all(r.ss_helix is None and r.ss_sheet is None for r in skipped)
    with out_skip.open(encoding="utf-8") as handle:
        rows2 = list(csv.DictReader(handle, delimiter="\t"))
    assert rows2[0]["SS_Helix_Fraction"] == "-"
    assert rows2[0]["SS_Sheet_Fraction"] == "-"


def test_cli_ss_flags() -> None:
    """CLI exposes --ss-head (default None) and --no-ss (default off)."""
    args = build_parser().parse_args(["in.fasta"])
    assert args.ss_head is None
    assert args.no_ss is False
    args2 = build_parser().parse_args(["in.fasta", "--ss-head", "custom.json", "--no-ss"])
    assert args2.ss_head == "custom.json"
    assert args2.no_ss is True


def test_annotate_schema_ss_keys() -> None:
    """/annotate detail carries tracks.ss3 + ss dict; meta.ss.active True."""
    sys.path.insert(0, str(ROOT))  # server.py lives at the repo root.
    import server  # noqa: E402

    fasta = SS_MINI_FASTA.read_text(encoding="utf-8")
    _records, detail, _eval, extras = server.annotate_fasta_text(fasta, force_fallback=True)
    assert len(detail) == 3
    for entry in detail:
        ss3 = entry["tracks"]["ss3"]
        assert len(ss3) == len(entry["sequence"])
        assert set(ss3) <= set(SS3_LABELS)
        assert set(entry["ss"]) == {"helix_f", "sheet_f", "coil_f", "source"}
        assert entry["ss"]["helix_f"] + entry["ss"]["sheet_f"] + entry["ss"]["coil_f"] == pytest.approx(1.0)
    assert extras["ss"]["active"] is True
    assert extras["ss"]["head_file"] is True  # data/ss_head_8m.json ships.


def test_live_esm_head_path_when_available() -> None:
    """With ESM-2 reachable, reps flow into head predictions (else skip)."""
    pytest.importorskip("esm")
    pytest.importorskip("torch")
    from protein_annotation.embeddings import _try_esm_forward  # noqa: E402

    head = load_ss_head(HEAD_JSON)
    assert head is not None
    seq = sequence.clean_sequence("live", "MKTAYIAKQRQISFVKSHFSRQDILDLWQK").cleaned
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Offline weights -> skip, not fail.
        forwarded = _try_esm_forward([seq], "esm2_t6_8M_UR50D")
    if forwarded is None:
        pytest.skip("ESM-2 weights unreachable from this host")
    reps, _ = forwarded
    pred = predict_ss3(seq, reps[0], head)
    assert len(pred) == len(seq)
    assert ss_source(head, reps[0]).startswith("esm2-ss-head:")
