"""Bacterial GO label build: GAF parse -> true-path propagate -> prune -> Y matrix.

Offline preprocessing for training the GO head on real labels (replaces the
12-term demo dict). Reads a GOA GAF 2.2 file plus go-basic.obo; writes a
terms JSON loadable via GOClassifier.load_weights() / --go-terms.
"""

from __future__ import annotations

import gzip
import json
import warnings
from collections import Counter
from pathlib import Path

# Experimental evidence codes kept by default (CAFA/DeepGO convention).
# IEA (electronic) is always dropped; phylogenetic IBA/IBD/IKR/IRD opt-in.
EXPERIMENTAL_EVIDENCE: frozenset[str] = frozenset(
    {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HEP", "HDA", "HMP", "HGI", "HPI"}
)

# GAF 2.2 0-based column indices (1-based spec: 2,4,5,7,9,13).
_GAF_ACC: int = 1
_GAF_QUAL: int = 3
_GAF_GO: int = 4
_GAF_EV: int = 6
_GAF_ASPECT: int = 8
_GAF_TAXON: int = 12


def _open_text(path: str | Path):
    """Open plain or .gz text for reading (GAF ships gzipped)."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def parse_gaf(
    path: str | Path,
    taxon_prefix: str = "taxon:2",
    evidence: frozenset[str] | None = None,
    aspects: frozenset[str] | None = None,
) -> dict[str, set[str]]:
    """Parse GAF 2.2 into {accession: direct GO ids} with standard filters.

    Args:
        path: goa_uniprot_all.gaf(.gz) or a per-species slice.
        taxon_prefix: Keep rows whose taxon column starts with this
            ("taxon:2" = Bacteria; "" disables taxon filtering).
        evidence: Evidence codes to keep (default EXPERIMENTAL_EVIDENCE).
        aspects: Aspect letters to keep, e.g. {"F","P","C"} (None = all).

    Filters: drops '!' comment lines, NOT-qualifier rows, non-kept
    evidence, non-kept aspects, and non-matching taxa.
    """
    keep_ev: frozenset[str] = evidence if evidence is not None else EXPERIMENTAL_EVIDENCE
    labels: dict[str, set[str]] = {}
    n_rows: int = 0
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("!") or not line.strip():
                continue  # GAF header / blank line.
            cols: list[str] = line.rstrip("\n").split("\t")
            if len(cols) <= _GAF_TAXON:
                continue  # Malformed short row; skip defensively.
            if "NOT" in cols[_GAF_QUAL].split("|"):
                continue  # Negative annotation, not a positive label.
            if cols[_GAF_EV] not in keep_ev:
                continue  # Non-experimental evidence (incl. IEA).
            if aspects is not None and cols[_GAF_ASPECT] not in aspects:
                continue
            if taxon_prefix and not cols[_GAF_TAXON].startswith(taxon_prefix):
                continue  # Non-bacterial annotation.
            acc: str = cols[_GAF_ACC].strip()
            go_id: str = cols[_GAF_GO].strip()
            if acc and go_id:
                labels.setdefault(acc, set()).add(go_id)
                n_rows += 1
    if not labels:
        warnings.warn(f"No labels kept from {path}; check taxon/evidence filters.")
    return labels


def parse_obo_ancestors(path: str | Path) -> dict[str, set[str]]:
    """Parse go-basic.obo into {child GO id: parent GO ids} (is_a + part_of).

    Obsolete terms (is_obsolete: true) are excluded so propagation never
    introduces dead terms.
    """
    parents: dict[str, set[str]] = {}
    cur_id: str | None = None
    cur_obsolete: bool = False
    cur_parents: set[str] = set()

    def _flush() -> None:
        if cur_id and not cur_obsolete:
            parents[cur_id] = set(cur_parents)

    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if line == "[Term]":
                _flush()  # Save previous stanza, start a new one.
                cur_id, cur_obsolete, cur_parents = None, False, set()
            elif line.startswith("id: GO:") and cur_id is None:
                cur_id = line.split()[1]
            elif line.startswith("is_obsolete: true"):
                cur_obsolete = True
            elif line.startswith("is_a: GO:"):
                cur_parents.add(line.split()[1])
            elif line.startswith("relationship: part_of GO:"):
                cur_parents.add(line.split()[2])
    _flush()  # Last stanza has no trailing [Term].
    return parents


def propagate(labels: dict[str, set[str]], parents: dict[str, set[str]]) -> dict[str, set[str]]:
    """True-path propagate: add every ancestor of each direct annotation."""
    memo: dict[str, set[str]] = {}

    def _ancestors(go_id: str) -> set[str]:
        if go_id in memo:
            return memo[go_id]
        seen: set[str] = set()
        stack: list[str] = list(parents.get(go_id, ()))
        while stack:  # Iterative DFS over the DAG (cycle-safe via `seen`).
            node: str = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(parents.get(node, ()))
        memo[go_id] = seen
        return seen

    return {acc: set(terms) | {a for t in terms for a in _ancestors(t)} for acc, terms in labels.items()}


def prune_terms(
    labels: dict[str, set[str]],
    min_count: int = 50,
    max_terms: int = 500,
) -> list[str]:
    """Frequency-prune post-propagation terms to a head-trainable label space.

    Keeps terms with >= min_count proteins, most frequent first, capped at
    max_terms. Returns [] with a warning when nothing passes the cutoff.
    """
    counts: Counter = Counter()
    for terms in labels.values():
        counts.update(terms)
    kept: list[str] = [t for t, c in counts.most_common(max_terms) if c >= min_count]
    if not kept:
        warnings.warn(f"No terms pass min_count={min_count}; lower the cutoff.")
    return kept


def build_label_matrix(
    labels: dict[str, set[str]], go_ids: list[str]
) -> tuple[list[str], list[list[int]]]:
    """Build accessions + binary (n, n_terms) Y matrix for GOClassifier.fit."""
    accessions: list[str] = sorted(labels)
    index: dict[str, int] = {go: j for j, go in enumerate(go_ids)}
    matrix: list[list[int]] = []
    for acc in accessions:
        row: list[int] = [0] * len(go_ids)
        for term in labels[acc]:
            if term in index:
                row[index[term]] = 1
        matrix.append(row)
    return accessions, matrix


def save_terms_json(go_ids: list[str], path: str | Path, counts: dict[str, int] | None = None) -> str:
    """Write a terms JSON ({go_id: {name, namespace, count}}) for --go-terms."""
    payload: dict[str, dict] = {
        go: {"name": "GO term " + go, "namespace": "?", "count": int(counts.get(go, 0)) if counts else 0}
        for go in go_ids
    }
    out: str = str(path)
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
