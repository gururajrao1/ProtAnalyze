"""NCBI AMRFinderPlus wrapper with graceful skip when the binary is missing.

AMRFinderPlus identifies AMR, stress-response, and virulence genes in protein
(or nucleotide) input. Public domain (NCBI-PD), installs via bioconda
(`conda install -c bioconda ncbi-amrfinderplus`, then `amrfinder -u` to fetch
the versioned DB). Linux/macOS only — on other hosts every entry point here
degrades to "not run", mirroring the hmmer.py pattern.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AmrHit:
    """One parsed AMRFinderPlus tabular row (protein-input mode)."""

    query_id: str  # Protein id column: our query sequence id.
    gene_symbol: str  # Gene symbol, e.g. "blaTEM-1".
    sequence_name: str  # Human-readable sequence name.
    scope: str  # e.g. "core", "plus".
    element_type: str  # "AMR", "STRESS", "VIRULENCE", ...
    gene_class: str  # Drug class, e.g. "BETA-LACTAM".
    pct_identity: float  # % identity to the reference sequence.
    pct_coverage: float  # % coverage of the reference sequence.
    method: str  # "ALIGNMENT", "EXACT", "HMM", ...

    def short(self) -> str:
        """Compact 'gene (class, id% identity)' string for reports."""
        return f"{self.gene_symbol} ({self.gene_class}, {self.pct_identity:.1f}% id)"


def amr_available() -> bool:
    """True when the `amrfinder` binary is on PATH."""
    return shutil.which("amrfinder") is not None


def _fnum(value: str | None) -> float:
    """Parse a float defensively (empty/missing -> 0.0)."""
    try:
        return float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def parse_amr_tsv(tsv_text: str) -> list[AmrHit]:
    """Parse AMRFinderPlus tabular output (header-driven, version-tolerant).

    Column lookup is by header name with several known spellings, so minor
    AMRFinderPlus version drift does not break parsing; unknown columns are
    ignored and missing ones default to ""/0.0.
    """
    lines: list[str] = [ln for ln in tsv_text.splitlines() if ln.strip()]
    if not lines:
        return []
    reader = csv.DictReader(lines, delimiter="\t")

    def _col(row: dict, *names: str) -> str:
        for name in names:
            if row.get(name) not in (None, ""):
                return str(row[name])
        return ""

    hits: list[AmrHit] = []
    for row in reader:
        query = _col(row, "Protein id", "Contig id")
        if not query:
            continue  # No query linkage; skip defensively.
        hits.append(
            AmrHit(
                query_id=query,
                gene_symbol=_col(row, "Gene symbol"),
                sequence_name=_col(row, "Sequence name"),
                scope=_col(row, "Scope"),
                element_type=_col(row, "Element type", "Type"),
                gene_class=_col(row, "Class"),
                pct_identity=_fnum(
                    _col(row, "% Identity to reference sequence", "% identity") or None
                ),
                pct_coverage=_fnum(
                    _col(row, "% Coverage of reference sequence", "% coverage") or None
                ),
                method=_col(row, "Method"),
            )
        )
    return hits


def run_amrfinder(
    query_fasta: str | Path,
    database: str | Path | None = None,
    plus: bool = True,
    timeout: int = 600,
) -> list[AmrHit]:
    """Run `amrfinder --protein` on a query FASTA and parse hits.

    Args:
        query_fasta: Protein FASTA of hypothetical proteins.
        database: AMRFinderPlus database dir (None = installed default).
        plus: Include --plus genes (stress/virulence, not just AMR).
        timeout: Subprocess timeout in seconds.

    Returns:
        Parsed hits (possibly empty). Returns [] with a warning when the
        binary is missing or the run fails — never raises for that.
    """
    if not amr_available():
        warnings.warn("AMRFinderPlus 'amrfinder' not found on PATH; skipping AMR scan.")
        return []
    with tempfile.NamedTemporaryFile(mode="r", suffix=".tsv", delete=False) as handle:
        out_path: str = handle.name
    cmd: list[str] = ["amrfinder", "--protein", str(query_fasta), "--output", out_path]
    if plus:
        cmd.append("--plus")  # Stress/virulence genes matter for pathogen scope.
    if database is not None:
        cmd += ["--database", str(database)]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        text: str = Path(out_path).read_text(encoding="utf-8", errors="replace")
        return parse_amr_tsv(text)
    except (OSError, subprocess.SubprocessError) as exc:
        warnings.warn(f"amrfinder run failed ({exc}); skipping AMR scan.")
        return []
    finally:
        try:
            Path(out_path).unlink(missing_ok=True)
        except OSError:
            pass


def hits_per_query(hits: list[AmrHit]) -> dict[str, list[AmrHit]]:
    """Group AMR hits by query protein id (input order preserved)."""
    grouped: dict[str, list[AmrHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.query_id, []).append(hit)
    return grouped
