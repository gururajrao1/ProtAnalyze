"""Train the GO head on real labels: UniProt REST -> OBO propagate -> fit.

One-shot laptop-scale training (no login, no GB downloads):
  1. Page UniProt REST for reviewed bacterial proteins with GO annotations.
  2. Download go-basic.obo once and true-path propagate each protein's terms.
  3. Keep proteins hitting the 12 demo GO terms, embed (hashed fallback),
     split 80/10/10, train GOClassifier.fit, save weights, CAFA-evaluate.

Outputs (all under data/, gitignored): raw REST TSV, OBO cache,
head_demo12.{pt,json} weights bundle, train_report.json.

Honest limitations (also stamped into train_report.json): REST rows carry
no per-term evidence codes, so IEA filtering needs the full GAF path
instead (see scripts/build_go_labels.py); the split is random, not
homology-aware; demo-12 term space keeps UI compatibility over coverage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_annotation.embeddings import embed_sequences  # noqa: E402
from protein_annotation.go_labels import parse_obo_ancestors, propagate  # noqa: E402
from protein_annotation.model import GO_IDS, GOClassifier, cafa_eval_report  # noqa: E402
from protein_annotation.sequence import clean_sequence, compute_protparam_features  # noqa: E402

REST_BASE = "https://rest.uniprot.org/uniprotkb/search"
OBO_URL = "https://purl.obolibrary.org/obo/go/go-basic.obo"
GO_ID_RE = re.compile(r"GO:\d{7}")


def fetch_url(url: str, timeout: int = 120, tries: int = 4) -> tuple[bytes, str]:
    """GET with backoff; returns (body, Link header or '')."""
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "protein-annotation-train/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), resp.headers.get("Link", "")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            wait = 5 * (2**attempt)
            print(f"  retry {attempt + 1}/{tries} after {wait}s ({exc})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"GET failed after {tries} tries: {url} ({last})")


def next_link(link_header: str) -> str | None:
    """Extract the rel=next URL from a REST pagination Link header."""
    for part in link_header.split(","):
        if 'rel="next"' in part:
            url = part.split(";")[0].strip().strip("<>")
            return url or None
    return None


def fetch_proteins(out_tsv: Path, max_proteins: int, page_size: int) -> int:
    """Page reviewed bacterial GO-annotated proteins into one TSV. Returns rows."""
    # NOTE: no go_id:* clause (REST 400s on it); rows without GO terms are
    # dropped client-side in load_labels instead.
    query = "(reviewed:true) AND (taxonomy_id:2)"
    params = {"query": query, "format": "tsv",
              "fields": "accession,sequence,go_id", "size": str(page_size)}
    url: str | None = REST_BASE + "?" + urllib.parse.urlencode(params)
    rows: list[str] = []
    header: str | None = None
    while url and len(rows) < max_proteins:
        body, link = fetch_url(url)
        lines = body.decode("utf-8", errors="replace").splitlines()
        if not lines:
            break
        if header is None:
            header = lines[0]
            print(f"  REST header: {header}", flush=True)
        rows.extend(lines[1:])
        print(f"  fetched {len(rows)}/{max_proteins} rows", flush=True)
        url = next_link(link)
    rows = rows[:max_proteins]
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.write_text((header or "accession\tsequence\tgo_id") + "\n" + "\n".join(rows) + "\n",
                       encoding="utf-8")
    return len(rows)


def load_labels(raw_tsv: Path, parents: dict[str, set[str]]) -> tuple[list[str], list[str], dict[str, set[str]]]:
    """Parse cached TSV -> (accessions, sequences, propagated GO sets)."""
    accs: list[str] = []
    seqs: list[str] = []
    direct: dict[str, set[str]] = {}
    with raw_tsv.open(encoding="utf-8") as handle:
        header = handle.readline()
        # REST TSV labels are "Entry/Sequence/Gene Ontology IDs" (not the
        # field names); match case-insensitively, fall back to position.
        cols = [c.strip().lower() for c in header.rstrip("\n").split("\t")]

        def _col(*names: str, default: int) -> int:
            for name in names:
                if name in cols:
                    return cols.index(name)
            return default

        ia = _col("entry", "accession", default=0)
        iseq = _col("sequence", default=1)
        igo = _col("gene ontology ids", "go_id", "go", default=2)
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(ia, iseq, igo):
                continue
            acc, seq = parts[ia].strip(), parts[iseq].strip().replace(" ", "")
            terms = set(GO_ID_RE.findall(parts[igo]))
            if acc and seq and terms:
                accs.append(acc)
                seqs.append(seq)
                direct[acc] = terms
    propagated = propagate(direct, parents)
    return accs, seqs, propagated


def main() -> int:
    """Fetch, label, train, save, and evaluate the demo-12 GO head."""
    parser = argparse.ArgumentParser(description="Train GO head on UniProt REST labels.")
    parser.add_argument("--max-proteins", type=int, default=3000)
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-len", type=int, default=30, help="Drop shorter (fragment) sequences.")
    parser.add_argument("--max-len", type=int, default=5000, help="Drop longer (junk) sequences.")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--skip-fetch", action="store_true", help="Reuse cached TSV+OBO.")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    raw_tsv = out_dir / "uniprot_bacterial_go.tsv"
    obo_path = out_dir / "go-basic.obo"

    if args.skip_fetch:
        print("Reusing cached downloads (--skip-fetch).", flush=True)
    else:
        print(f"[1/5] Fetching up to {args.max_proteins} proteins from UniProt REST...", flush=True)
        n = fetch_proteins(raw_tsv, args.max_proteins, args.page_size)
        print(f"Saved {n} rows -> {raw_tsv}", flush=True)
        if not obo_path.exists():
            print("[2/5] Downloading go-basic.obo (~35MB)...", flush=True)
            body, _ = fetch_url(OBO_URL, timeout=300)
            obo_path.parent.mkdir(parents=True, exist_ok=True)
            obo_path.write_bytes(body)
            print(f"Saved -> {obo_path} ({len(body) / 1e6:.1f}MB)", flush=True)
        else:
            print("[2/5] OBO cache hit.", flush=True)

    print("[3/5] Parsing + propagating labels...", flush=True)
    parents = parse_obo_ancestors(obo_path)
    accs, seqs, propagated = load_labels(raw_tsv, parents)
    # Length filter + keep proteins hitting the demo-12 space.
    want = set(GO_IDS)
    kept = [(a, s) for a, s in zip(accs, seqs)
            if args.min_len <= len(s) <= args.max_len and (propagated.get(a, set()) & want)]
    print(f"Kept {len(kept)}/{len(accs)} proteins (length + demo-12 coverage).", flush=True)
    if len(kept) < 50:
        print("ERROR: too few labelled proteins; widen the query.", flush=True)
        return 1
    per_term = {g: sum(1 for a, _ in kept if g in propagated[a]) for g in GO_IDS}
    print("Per-term positives: " + ", ".join(f"{g}={c}" for g, c in per_term.items()), flush=True)

    print("[4/5] Embedding (hashed fallback, CPU)...", flush=True)
    cleaned = [clean_sequence(a, s) for a, s in kept]
    feats = [compute_protparam_features(c) for c in cleaned]
    emb = embed_sequences([c.cleaned for c in cleaned], force_fallback=True)
    X = np.concatenate([emb.embeddings.astype(np.float32),
                        np.array([f.to_vector() for f in feats], dtype=np.float32)], axis=1)
    Y = np.zeros((len(kept), len(GO_IDS)), dtype=int)
    for i, (a, _) in enumerate(kept):
        for term in propagated[a] & want:
            Y[i, GO_IDS.index(term)] = 1

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(kept))
    n = len(kept)
    tr, va = int(0.8 * n), int(0.9 * n)
    splits = {"train": idx[:tr], "val": idx[tr:va], "test": idx[va:]}
    print(f"Split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}",
          flush=True)
    clf = GOClassifier(input_dim=X.shape[1])
    clf.fit(X[splits["train"]], Y[splits["train"]], X[splits["val"]], Y[splits["val"]],
            epochs=args.epochs, batch_size=args.batch_size)
    weights = clf.save_weights(str(out_dir / "head_demo12"))
    print(f"[5/5] Saved weights -> {weights}", flush=True)

    report = cafa_eval_report(Y[splits["test"]], clf.predict_proba(X[splits["test"]]), go_ids=GO_IDS)
    report["train_args"] = vars(args)
    report["n_fetched"] = len(accs)
    report["n_kept"] = len(kept)
    report["per_term_positives"] = per_term
    report["weights"] = weights
    report["limitations"] = [
        "REST rows carry no per-term evidence codes; IEA filtering needs the GAF path.",
        "Random split, not homology-aware; scores may be optimistic.",
        "Demo-12 term space; sparse terms (e.g. nucleus in bacteria) are near-always-negative.",
    ]
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Test Fmax={report['fmax']:.3f} Smin={report['smin']:.3f} "
          f"AUPR={report['aupr_micro']:.3f} (n={report['n_proteins']})", flush=True)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
