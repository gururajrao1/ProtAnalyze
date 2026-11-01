"""HMMER phmmer CLI wrapper with graceful skip when HMMER is missing."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HmmerHit:
    """One parsed phmmer --tblout row."""

    query_id: str
    target_id: str
    evalue: float
    bitscore: float

    def short(self) -> str:
        """Compact 'target (e=.., bit=..)' string for TSV output."""
        return f"{self.target_id} (e={self.evalue:.2g}, bit={self.bitscore:.1f})"


def hmmer_available() -> bool:
    """True when the `phmmer` binary is on PATH."""
    return shutil.which("phmmer") is not None


def parse_tblout(tblout_text: str) -> list[HmmerHit]:
    """Parse phmmer --tblout format (whitespace-delimited, '#' comments).

    Column layout: target name(0) accession(1) query name(2) accession(3)
    E-value(4) bit score(5) ... (remaining columns ignored).
    """
    hits: list[HmmerHit] = []
    for line in tblout_text.splitlines():
        line = line.strip()
        # Skip comments and blank lines.
        if not line or line.startswith("#"):
            continue
        parts: list[str] = line.split()
        if len(parts) < 6:
            continue  # Malformed row; ignore defensively.
        try:
            hits.append(
                HmmerHit(
                    query_id=parts[2],
                    target_id=parts[0],
                    evalue=float(parts[4]),
                    bitscore=float(parts[5]),
                )
            )
        except ValueError:
            continue  # Non-numeric e-value/score; skip row.
    # Best (lowest e-value) first per file convention.
    hits.sort(key=lambda h: h.evalue)
    return hits


def run_phmmer(
    query_fasta: str | Path,
    db_fasta: str | Path,
    evalue_threshold: float = 10.0,
    cpus: int = 1,
) -> list[HmmerHit]:
    """Run `phmmer` query-vs-database search and parse hits.

    Args:
        query_fasta: FASTA of hypothetical proteins to annotate.
        db_fasta: Reference protein database FASTA.
        evalue_threshold: E-value cutoff passed via `-E`.
        cpus: Worker threads passed via `--cpu`.

    Returns:
        Parsed hits (possibly empty). Returns [] with a warning when the
        phmmer binary is missing or the run fails — never raises for that.
    """
    if not hmmer_available():
        warnings.warn("HMMER 'phmmer' not found on PATH; skipping homology search.")
        return []
    # tblout goes to a temp file; stdout is suppressed for quiet runs.
    with tempfile.NamedTemporaryFile(mode="r", suffix=".tbl", delete=False) as handle:
        tbl_path: str = handle.name
    cmd: list[str] = [
        "phmmer",
        "--tblout",
        tbl_path,
        "-E",
        str(evalue_threshold),
        "--cpu",
        str(cpus),
        str(query_fasta),
        str(db_fasta),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
        text: str = Path(tbl_path).read_text(encoding="utf-8", errors="replace")
        return parse_tblout(text)
    except (OSError, subprocess.SubprocessError) as exc:
        warnings.warn(f"phmmer run failed ({exc}); skipping homology search.")
        return []
    finally:
        # Best-effort temp cleanup; ignore failures.
        try:
            Path(tbl_path).unlink(missing_ok=True)
        except OSError:
            pass


def top_hit_per_query(hits: list[HmmerHit]) -> dict[str, HmmerHit]:
    """Keep the single best (lowest e-value) hit per query id."""
    best: dict[str, HmmerHit] = {}
    for hit in hits:
        # Hits arrive best-first, so first occurrence wins.
        best.setdefault(hit.query_id, hit)
    return best
