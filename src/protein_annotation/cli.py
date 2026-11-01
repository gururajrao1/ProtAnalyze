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
    # Trained GO head bundle (GOClassifier.save_weights stem or .json sidecar).
    parser.add_argument("--go-weights", default=None, help="Trained GO head bundle path (omit for demo head).")
    # Terms-only label-space JSON (alternative to --go-weights; ignored if both).
    parser.add_argument("--go-terms", default=None, help="GO terms JSON label space (random-init head).")
    # Opt-in 9-column TSV (+Disordered_Fraction, +Embedding_Source).
    parser.add_argument("--extended-tsv", action="store_true", help="Write extended 9-column TSV.")
    # Skip the offline disorder-fraction signal.
    parser.add_argument("--no-disorder", action="store_true", help="Skip disorder-fraction computation.")
    # CAFA eval: labels TSV (protein_id<TAB>GO:1,GO:2) + JSON report sidecar.
    parser.add_argument("--eval-labels", default=None, help="Eval labels TSV for the CAFA report.")
    parser.add_argument("--eval-report", default=None, help="CAFA eval report JSON path (needs --eval-labels).")
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
            go_weights=args.go_weights,
            go_terms=args.go_terms,
            extended_tsv=args.extended_tsv,
            include_disorder=not args.no_disorder,
            eval_labels=args.eval_labels,
            eval_report=args.eval_report,
        )
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Annotated {len(records)} proteins -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
