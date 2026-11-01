"""Build a bacterial GO label space + Y matrix from GOA GAF + go-basic.obo.

Offline preprocessing (TOP2): replaces the 12-term demo dict with 100-500
real bacterial terms. Outputs a terms JSON (for --go-terms) and optionally
a labels NPZ (accessions + Y matrix) ready for GOClassifier.fit.

Example:
  python scripts/build_go_labels.py goa_uniprot_all.gaf.gz go-basic.obo \\
      --out-terms data/go_bacterial_terms.json --out-labels data/go_labels.npz \\
      --min-count 100 --max-terms 300

Laptop alternative: use a per-species GAF slice or the CAFA5 Kaggle mirrors
(see docs) instead of the full ~10GB GAF.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Repo root on sys.path so the script works without packaging.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_annotation.go_labels import (  # noqa: E402
    build_label_matrix,
    parse_gaf,
    parse_obo_ancestors,
    propagate,
    prune_terms,
    save_terms_json,
)


def build_parser() -> argparse.ArgumentParser:
    """CLI for the GO label-space build."""
    parser = argparse.ArgumentParser(description="Build bacterial GO labels from GAF + OBO.")
    parser.add_argument("gaf", help="GOA GAF 2.2 file (.gaf or .gaf.gz).")
    parser.add_argument("obo", help="go-basic.obo ontology file.")
    parser.add_argument("--out-terms", required=True, help="Output terms JSON path.")
    parser.add_argument("--out-labels", default=None, help="Optional output labels .npz path.")
    parser.add_argument("--taxon", default="taxon:2", help="Taxon prefix filter ('' = no filter).")
    parser.add_argument("--min-count", type=int, default=50, help="Min proteins per kept term.")
    parser.add_argument("--max-terms", type=int, default=500, help="Max terms to keep.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse -> propagate -> prune -> write terms JSON (+ optional NPZ)."""
    args: argparse.Namespace = build_parser().parse_args(argv)
    direct = parse_gaf(args.gaf, taxon_prefix=args.taxon)
    print(f"Parsed {len(direct)} proteins with direct annotations.")
    parents = parse_obo_ancestors(args.obo)
    print(f"Loaded {len(parents)} OBO terms.")
    full = propagate(direct, parents)
    kept = prune_terms(full, min_count=args.min_count, max_terms=args.max_terms)
    print(f"Kept {len(kept)} terms after pruning.")
    if not kept:
        return 1
    counts: Counter = Counter()
    for terms in full.values():
        counts.update(t for t in terms if t in set(kept))
    save_terms_json(kept, args.out_terms, counts=dict(counts))
    print(f"Wrote terms -> {args.out_terms}")
    if args.out_labels:
        import numpy as np  # Local import: only needed for the NPZ output.

        accessions, matrix = build_label_matrix(full, kept)
        np.savez_compressed(
            args.out_labels,
            accessions=np.array(accessions),
            Y=np.array(matrix, dtype=np.int8),
            go_ids=np.array(kept),
        )
        print(f"Wrote labels -> {args.out_labels} ({len(accessions)} x {len(kept)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
