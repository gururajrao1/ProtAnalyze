"""Core sequence utilities: cleaning/masking and ProtParam features."""

from __future__ import annotations

from dataclasses import dataclass, field

from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Standard 20 amino acids handled reliably by ProtParam.
STANDARD_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Ambiguous / rare residues requiring cleaning + masking.
AMBIGUOUS_AA: frozenset[str] = frozenset("XBZOU")

# Deterministic substitution map so ProtParam sees only standard residues.
# B (Asx) -> N, Z (Glx) -> Q, X (unknown) -> A, U (selenocysteine) -> C,
# O (pyrrolysine) -> K. Choices follow closest physicochemical neighbour.
SUBSTITUTION_MAP: dict[str, str] = {"B": "N", "Z": "Q", "X": "A", "U": "C", "O": "K"}

# Fixed order for the 20-dim AA frequency vector.
AA_ORDER: list[str] = list("ACDEFGHIKLMNPQRSTVWY")

# Coefficients for Ikai (1980) aliphatic index.
_ALIPHATIC_A: float = 2.9
_ALIPHATIC_B: float = 3.9


@dataclass
class CleanedSequence:
    """A raw sequence plus its cleaned form and validity mask."""

    record_id: str
    raw: str
    cleaned: str
    # mask[i] is True when raw residue i (after stripping gaps) is standard.
    mask: list[bool] = field(default_factory=list)

    @property
    def n_masked(self) -> int:
        """Count of ambiguous residues that were substituted."""
        return sum(1 for m in self.mask if not m)


def clean_sequence(record_id: str, sequence: str) -> CleanedSequence:
    """Uppercase, strip gaps, substitute ambiguous residues, build mask.

    Args:
        record_id: FASTA record identifier (kept for provenance).
        sequence: Raw amino-acid string, possibly with gaps/ambiguity codes.

    Returns:
        CleanedSequence with gap-free cleaned string and per-residue mask.
    """
    cleaned_chars: list[str] = []
    mask: list[bool] = []
    for char in sequence.upper():
        # Skip alignment gaps, stop codons and whitespace silently.
        if char in ("-", ".", "*", " ", "\t", "\n", "\r"):
            continue
        if char in STANDARD_AA:
            cleaned_chars.append(char)
            mask.append(True)
        elif char in AMBIGUOUS_AA:
            # Substitute so downstream tools see valid input; mark masked.
            cleaned_chars.append(SUBSTITUTION_MAP[char])
            mask.append(False)
        else:
            # Any other unexpected symbol is treated as unknown alanine.
            cleaned_chars.append("A")
            mask.append(False)
    return CleanedSequence(
        record_id=record_id, raw=sequence, cleaned="".join(cleaned_chars), mask=mask
    )


def aliphatic_index(aa_percent: dict[str, float]) -> float:
    """Compute Ikai (1980) aliphatic index from mole percents.

    Formula: X(Ala) + a*X(Val) + b*(X(Ile) + X(Leu)) with a=2.9, b=3.9,
    where X() are mole percent values (0-100 scale) from ProtParam.
    """
    return (
        aa_percent.get("A", 0.0)
        + _ALIPHATIC_A * aa_percent.get("V", 0.0)
        + _ALIPHATIC_B * (aa_percent.get("I", 0.0) + aa_percent.get("L", 0.0))
    )


@dataclass
class ProtParamFeatures:
    """Physicochemical feature bundle for one protein."""

    length: int
    molecular_weight: float
    isoelectric_point: float
    aromaticity: float
    instability_index: float
    gravy: float
    aliphatic: float
    aa_freq: dict[str, float] = field(default_factory=dict)

    def to_vector(self) -> list[float]:
        """Fixed 27-dim vector: 7 scaled globals + 20 AA frequencies (AA_ORDER).

        Globals are scaled to roughly [-1, 1] so raw magnitudes (e.g. MW in
        Daltons) cannot saturate the downstream classifier head.
        """
        globals_: list[float] = [
            float(self.length) / 500.0,
            self.molecular_weight / 50000.0,
            self.isoelectric_point / 14.0,
            self.aromaticity,  # Already 0-1.
            self.instability_index / 100.0,
            self.gravy / 2.0,
            self.aliphatic / 200.0,
        ]
        # AA frequencies fall back to 0.0 if a residue is absent.
        freqs: list[float] = [float(self.aa_freq.get(aa, 0.0)) for aa in AA_ORDER]
        return globals_ + freqs


def compute_protparam_features(cleaned: CleanedSequence) -> ProtParamFeatures:
    """Run Bio.SeqUtils.ProtParam over a cleaned sequence.

    Args:
        cleaned: Output of clean_sequence(); uses the substituted string.

    Returns:
        ProtParamFeatures with pI, MW, aliphatic index and AA frequencies.
    """
    seq: str = cleaned.cleaned if cleaned.cleaned else "A"  # Guard empty input.
    analysis = ProteinAnalysis(seq)
    # amino_acids_percent gives mole percent on the 0-100 scale.
    aa_percent: dict[str, float] = analysis.amino_acids_percent
    # Normalise to 0-1 fractions for the feature vector / reporting.
    aa_freq: dict[str, float] = {aa: pct / 100.0 for aa, pct in aa_percent.items()}
    return ProtParamFeatures(
        length=len(seq),
        molecular_weight=float(analysis.molecular_weight()),
        isoelectric_point=float(analysis.isoelectric_point()),
        aromaticity=float(analysis.aromaticity()),
        instability_index=float(analysis.instability_index()),
        gravy=float(analysis.gravy()),
        aliphatic=float(aliphatic_index(aa_percent)),
        aa_freq=aa_freq,
    )
