"""Fetch SS3 training labels from RCSB PDB author annotations.

Primary source (verified reachable 2026-11-01):
  * query: POST https://search.rcsb.org/rcsbsearch/v2/query (X-ray,
    resolution <= 2.0 A, exactly one polymer entity; non-protein chains
    are skipped client-side during parsing)
  * files: https://files.rcsb.org/download/{PDBID}.pdb (legacy PDB format)

Labels come from author-annotated HELIX/SHEET records (H/E; everything
else C) aligned to the SEQRES sequence of the first chain. No guessed
URLs: every PDB id is returned by the search API itself.

The original spec pointed at the Princeton CullPDB/CB513 .npy.gz mirror
(https://www.princeton.edu/~jzthree/datasets/ICML2014/), which answers
HTTP 403 from this host; if you can reach it, pass a local download via
--cullpdb-npy and this script converts seq + DSSP-SS3 labels (dropping
the 57-dim profiles) into the same JSONL format the trainer consumes.

Output: JSONL with one {"id", "seq", "ss3", "source"} object per line,
plus a {"provenance": ...} first line recording query/date/counts.

Usage:
  python scripts/fetch_ss_data.py --out data/ss_labels.jsonl --max-entries 400
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import io
import json
import sys
import urllib.request
from pathlib import Path

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

RES3_TO_RES1: dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "MSE": "M", "ASN": "N", "PRO": "P", "GLN": "Q",
    "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W",
    "TYR": "Y",
}

# DSSP 8-state -> SS3 map for the optional CullPDB .npy converter.
DSSP8_TO_SS3 = {"H": "H", "G": "H", "I": "H", "E": "E", "B": "E",
                "S": "C", "T": "C", "C": "C", " ": "C", "-": "C"}


def search_rcsb(max_entries: int, timeout: int = 60) -> list[str]:
    """Return up to `max_entries` PDB ids matching the quality filter."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text",
                 "parameters": {"operator": "exact_match", "value": "X-RAY DIFFRACTION",
                               "attribute": "exptl.method"}},
                {"type": "terminal", "service": "text",
                 "parameters": {"operator": "less_or_equal", "value": 2.0,
                               "attribute": "rcsb_entry_info.resolution_combined"}},
                {"type": "terminal", "service": "text",
                 "parameters": {"operator": "equals", "value": 1,
                               "attribute": "rcsb_entry_info.polymer_entity_count"}},
                {"type": "terminal", "service": "text",
                 "parameters": {"operator": "less_or_equal", "value": 700,
                               "attribute": "rcsb_entry_info.deposited_polymer_monomer_count"}},
            ],
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": max_entries},
                            "results_content_type": ["experimental"]},
    }
    req = urllib.request.Request(SEARCH_URL, data=json.dumps(query).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return sorted(r["identifier"].upper() for r in payload.get("result_set", []))


def fetch_pdb(pdb_id: str, timeout: int = 60) -> str:
    """Download one legacy-PDB file; raises URLError on failure."""
    url = PDB_URL.format(pdb_id=pdb_id.lower())
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_pdb_ss(pdb_text: str) -> tuple[str, str] | None:
    """Extract (SEQRES sequence, H/E/C labels) for the first chain.

    Returns None when the entry is unusable (no SEQRES, non-contiguous
    numbering, non-standard residues): callers skip it and count why.
    """
    seqres: dict[str, list[str]] = {}  # chain -> [res3, ...]
    order: list[str] = []
    helices: list[tuple[str, int, int]] = []
    sheets: list[tuple[str, int, int]] = []
    atom_seqnums: dict[str, list[int]] = {}
    for line in pdb_text.splitlines():
        rec = line[:6]
        if rec == "SEQRES":
            chain = line[11:12]
            if chain not in seqres:
                seqres[chain] = []
                order.append(chain)
            seqres[chain].extend(line[19:70].split())
        elif rec == "HELIX ":
            chain_a, chain_b = line[19:20], line[31:32]
            icode_a, icode_b = line[25:26], line[37:38]
            if chain_a != chain_b or icode_a.strip() or icode_b.strip():
                continue  # Multi-chain span or insertion code: skip range.
            try:
                helices.append((chain_a, int(line[21:25]), int(line[33:37])))
            except ValueError:
                continue
        elif rec == "SHEET ":
            chain_a, chain_b = line[21:22], line[32:33]
            icode_a, icode_b = line[26:27], line[37:38]
            if chain_a != chain_b or icode_a.strip() or icode_b.strip():
                continue
            try:
                sheets.append((chain_a, int(line[22:26]), int(line[33:37])))
            except ValueError:
                continue
        elif rec == "ATOM  ":
            chain = line[21:22]
            if line[26:27].strip():
                continue  # Insertion-coded residue: ignore for numbering.
            try:
                num = int(line[22:26])
            except ValueError:
                continue
            seen = atom_seqnums.setdefault(chain, [])
            if not seen or seen[-1] != num:
                seen.append(num)
    if not order:
        return None
    chain = order[0]
    res3 = seqres[chain]
    if not (40 <= len(res3) <= 700):
        return None
    try:
        seq = "".join(RES3_TO_RES1[r] for r in res3)
    except KeyError:
        return None  # Non-standard residue present: skip entry.
    observed = atom_seqnums.get(chain, [])
    if not observed:
        return None
    # Numbering must fit SEQRES contiguously: index i <-> seqnum0 + i.
    seqnum0 = observed[0]
    if any(n < seqnum0 or n >= seqnum0 + len(res3) for n in observed):
        return None
    labels = ["C"] * len(res3)
    for ch, a, b in helices:
        if ch != chain:
            continue
        for n in range(a, b + 1):
            i = n - seqnum0
            if 0 <= i < len(labels):
                labels[i] = "H"
    for ch, a, b in sheets:
        if ch != chain:
            continue
        for n in range(a, b + 1):
            i = n - seqnum0
            if 0 <= i < len(labels) and labels[i] != "H":
                labels[i] = "E"
    ss3 = "".join(labels)
    if set(ss3) <= {"C"}:
        return None  # No annotated structure at all: uninformative.
    return seq, ss3


def convert_cullpdb_npy(path: Path) -> list[dict[str, str]]:
    """Convert a Princeton cullpdb/cb513 .npy.gz file to id/seq/ss3 dicts.

    Keeps sequence + DSSP-derived SS3 labels only; drops the 57-dim
    profile/sequence one-hot inputs (the ESM head replaces them).
    """
    import numpy as np

    with gzip.open(path, "rb") as handle:
        arr = np.load(io.BytesIO(handle.read()))
    # Layout (Zhou & Troyanskaya): [..., 0:22 one-hot AA, 22:31 DSSP8, ...].
    aa_order = "ACEDGFIHKMLNQPSRTWVYX"
    dssp_order = "LBE G IHS T".replace(" ", "")  # L,B,E,G,I,H,S,T
    out: list[dict[str, str]] = []
    for k in range(arr.shape[0]):
        mat = arr[k]
        n = int((mat[:, 21] < 0.5).sum()) if mat.shape[1] > 21 else mat.shape[0]
        seq_chars, ss_chars = [], []
        for row in mat[:n]:
            seq_chars.append(aa_order[int(np.argmax(row[0:21]))])
            ss_chars.append(DSSP8_TO_SS3[dssp_order[int(np.argmax(row[22:30]))]])
        out.append({"id": f"{path.stem}-{k}", "seq": "".join(seq_chars),
                    "ss3": "".join(ss_chars), "source": path.name})
    return [r for r in out if 10 <= len(r["seq"]) <= 1022]


def main(argv: list[str] | None = None) -> int:
    """CLI: fetch RCSB labels (and optionally convert local .npy files)."""
    parser = argparse.ArgumentParser(description="Fetch SS3 labels from RCSB PDB.")
    parser.add_argument("--out", default="data/ss_labels.jsonl", help="Output JSONL path.")
    parser.add_argument("--max-entries", type=int, default=400, help="RCSB ids to request.")
    parser.add_argument("--max-kept", type=int, default=250, help="Stop after N usable chains.")
    parser.add_argument("--min-kept", type=int, default=60, help="Fail below N usable chains.")
    parser.add_argument("--cullpdb-npy", nargs="*", default=[], help="Local Princeton .npy.gz files.")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout (s).")
    args = parser.parse_args(argv)

    records: list[dict[str, str]] = []
    provenance: dict = {"date": _dt.date.today().isoformat(), "rcsb_ids": 0,
                        "rcsb_kept": 0, "rcsb_skipped": 0, "npy_files": []}
    try:
        pdb_ids = search_rcsb(args.max_entries, timeout=args.timeout)
    except Exception as exc:  # noqa: BLE001 - network failure is fatal here
        print(f"error: RCSB search failed ({exc})", file=sys.stderr)
        return 1
    provenance["rcsb_ids"] = len(pdb_ids)
    for pdb_id in pdb_ids:
        if len(records) >= args.max_kept:
            break
        try:
            parsed = parse_pdb_ss(fetch_pdb(pdb_id, timeout=args.timeout))
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            print(f"warn: {pdb_id} fetch failed ({exc})", file=sys.stderr)
            provenance["rcsb_skipped"] += 1
            continue
        if parsed is None:
            provenance["rcsb_skipped"] += 1
            continue
        seq, ss3 = parsed
        records.append({"id": pdb_id, "seq": seq, "ss3": ss3, "source": "rcsb-pdb"})
        provenance["rcsb_kept"] += 1
    for npy in args.cullpdb_npy:
        converted = convert_cullpdb_npy(Path(npy))
        records.extend(converted)
        provenance["npy_files"].append({"file": npy, "kept": len(converted)})
    if len(records) < args.min_kept:
        print(f"error: only {len(records)} usable chains (< {args.min_kept})", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"provenance": provenance}) + "\n")
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"Kept {len(records)} chains -> {out} (skipped {provenance['rcsb_skipped']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
