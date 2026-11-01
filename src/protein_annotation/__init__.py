"""Hypothetical-protein annotation pipeline (CLI core, UI-independent)."""

from .pipeline import TSV_COLUMNS, AnnotationRecord, run_pipeline

__all: list[str] = ["TSV_COLUMNS", "AnnotationRecord", "run_pipeline"]
__version__: str = "0.1.0"
