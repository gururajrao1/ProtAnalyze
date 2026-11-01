"""argparse CLI entry point (no streamlit; core stays UI-independent)."""

from __future__ import annotations

import argparse
import sys

from .embeddings import DEFAULT_ESM_MODEL, SUPPORTED_ESM_MODELS
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface for the annotation pipeline."""
    parser = argparse.ArgumentParser(
        prog="protein-annotation",
        description="Annotate hypothetical proteins: ProtParam + ESM-2 + HMMER + GO multilabel head.",
    )
    # Required input FASTA of hypothetical proteins.
    parser.add_argument("input_fasta", help="Input protein FASTA file.")
    # Output TSV path (default in current directory).
    parser.add_argument("-o", "--output", default="protein_annotations.tsv", help="Output TSV path.")
    # ESM-2 checkpoint choice: light 8M default, heavy 650M optional.
    parser.add_argument(
        "--esm-model",
        choices=list(SUPPORTED_ESM_MODELS),
        default=DEFAULT_ESM_MODEL,
        help="ESM-2 checkpoint for embeddings.",
    )
    # Skip ESM-2 entirely (offline/CI): hashed fallback embeddings.
    parser.add_argument("--no-esm", action="store_true", help="Force hashed fallback embeddings.")
    # Optional reference database FASTA enabling the phmmer step.
    parser.add_argument("--hmmer-db", default=None, help="Reference FASTA for phmmer search (omit to skip).")
    # Number of GO terms reported per protein.
    parser.add_argument("--top-k", type=int, default=3, help="Top GO terms per protein.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI main: parse args, run pipeline, report output path."""
    args: argparse.Namespace = build_parser().parse_args(argv)
    try:
        records = run_pipeline(
            input_fasta=args.input_fasta,
            output_tsv=args.output,
            esm_model=args.esm_model,
            hmmer_db=args.hmmer_db,
            top_k=args.top_k,
            force_fallback_embeddings=args.no_esm,
        )
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Annotated {len(records)} proteins -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
