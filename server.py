"""Unified protein annotation web app: FastAPI + static domain-style UI.

Backend reuses the core pipeline modules (no streamlit anywhere):
  sequence.py  - cleaning/masking + ProtParam features
  embeddings.py- ESM-2 / hashed fallback
  hmmer.py     - phmmer wrapper (graceful skip)
  model.py     - GO multilabel head
  pipeline.py  - TSV orchestration

Run:
  pip install -r requirements.txt
  uvicorn server:app --reload            # then open http://127.0.0.1:8000/
"""

from __future__ import annotations

import csv
import io
import sys
import tempfile
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from Bio import SeqIO
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from protein_annotation.embeddings import DEFAULT_ESM_MODEL, embed_sequences
from protein_annotation.hmmer import HmmerHit, hmmer_available, run_phmmer, top_hit_per_query
from protein_annotation.model import GO_IDS, GO_TERMS, GOClassifier, cafa_eval_report
from protein_annotation.pipeline import (
    TSV_COLUMNS,
    AnnotationRecord,
    annotate_records,
    describe_function,
    load_eval_labels,
)
from protein_annotation.sequence import (
    clean_sequence,
    compute_protparam_features,
    disorder_scores,
)

# Kyte-Doolittle hydrophobicity scale for per-residue tracks.
KD_SCALE: dict[str, float] = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

# Residue property classes for the sequence-viewer colouring.
AA_CLASS: dict[str, str] = {}
for _aa in "AILMFWYV":
    AA_CLASS[_aa] = "hydrophobic"
for _aa in "RK":
    AA_CLASS[_aa] = "positive"
for _aa in "DE":
    AA_CLASS[_aa] = "negative"
for _aa in "STNQ":
    AA_CLASS[_aa] = "polar"
for _aa in "CHPG":
    AA_CLASS[_aa] = "special"

def is_vfdb_hit(target_id: str) -> bool:
    """True when a HMMER target id looks like a VFDB virulence factor (VFG...)."""
    tid = target_id.strip().split()[0].split("|")[0] if target_id.strip() else ""
    return tid.upper().startswith("VFG")


def hmmer_db_kind(hmmer_db_text: str | None) -> str:
    """Classify the pasted reference DB: 'vfdb' when headers carry VFG ids."""
    if not hmmer_db_text or not hmmer_db_text.strip():
        return "none"
    return "vfdb" if ">VFG" in hmmer_db_text.upper() else "custom"


app = FastAPI(title="Protein Annotation Workbench", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


def smooth(values: list[float], window: int = 5) -> list[float]:
    """Centered moving average (edge-shrunk) for track smoothing."""
    if window <= 1 or not values:
        return list(values)
    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(float(sum(values[lo:hi]) / (hi - lo)))
    return out


def residue_tracks(sequence: str) -> dict[str, list[float]]:
    """Per-residue hydrophobicity (KD) + charge tracks for the viewer."""
    hydro = [KD_SCALE.get(aa, 0.0) for aa in sequence]
    charge = [1.0 if aa in ("K", "R") else (-1.0 if aa in ("D", "E") else 0.0) for aa in sequence]
    return {"hydro": hydro, "hydro_smooth": smooth(hydro), "charge": charge}


def annotate_fasta_text(
    fasta_text: str,
    top_k: int = 3,
    esm_model: str = DEFAULT_ESM_MODEL,
    force_fallback: bool = True,
    hmmer_db_text: str | None = None,
    eval_labels_text: str | None = None,
) -> tuple[list[AnnotationRecord], list[dict], dict | None]:
    """Annotate FASTA text, returning TSV records + rich detail + CAFA eval."""
    parsed = list(SeqIO.parse(io.StringIO(fasta_text), "fasta"))
    if not parsed:
        raise ValueError("No FASTA records found in input")
    cleaned = [clean_sequence(r.id, str(r.seq)) for r in parsed]
    features = [compute_protparam_features(c) for c in cleaned]
    embedding = embed_sequences(
        [c.cleaned for c in cleaned],
        [c.mask for c in cleaned],
        model_name=esm_model,
        force_fallback=force_fallback,
    )
    # Optional in-memory HMMER reference DB (temp files, graceful skip).
    hmmer_best: dict[str, HmmerHit] = {}
    if hmmer_db_text and hmmer_db_text.strip():
        if hmmer_available():
            with tempfile.TemporaryDirectory() as tmp:
                q = Path(tmp) / "q.fasta"
                db = Path(tmp) / "db.fasta"
                q.write_text(fasta_text, encoding="utf-8")
                db.write_text(hmmer_db_text, encoding="utf-8")
                hmmer_best = top_hit_per_query(run_phmmer(q, db))
        else:
            warnings.warn("phmmer binary not found; skipping HMMER step")
    classifier = GOClassifier(input_dim=embedding.dim + len(features[0].to_vector()))
    records = annotate_records(cleaned, features, embedding, hmmer_best, classifier, top_k)

    # Optional CAFA eval report over pasted labels TSV (protein_id<TAB>GO:1,GO:2).
    eval_report: dict | None = None
    if eval_labels_text and eval_labels_text.strip():
        prot_vectors = np.array([f.to_vector() for f in features], dtype=np.float32)
        combined = np.concatenate([embedding.embeddings.astype(np.float32), prot_vectors], axis=1)
        head = classifier
        if head.input_dim != combined.shape[1]:
            head = GOClassifier(
                input_dim=combined.shape[1], go_ids=head.go_ids, thresholds=head.thresholds
            )
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.tsv"
            labels_path.write_text(eval_labels_text, encoding="utf-8")
            Y_true = load_eval_labels(labels_path, [c.record_id for c in cleaned], head.go_ids)
        eval_report = cafa_eval_report(Y_true, head.predict_proba(combined), go_ids=head.go_ids)

    # Rich detail payload for the domain UI.
    detail: list[dict] = []
    for i, seq in enumerate(cleaned):
        feat = features[i]
        rec = records[i]
        tracks = residue_tracks(seq.cleaned)
        go_terms = [
            {
                "id": gid,
                "name": GO_TERMS.get(gid, {}).get("name", "?"),
                "namespace": GO_TERMS.get(gid, {}).get("namespace", "?"),
                "score": round(float(score), 4),
            }
            for gid, score in zip(rec.top_go_terms, rec.confidence_scores)
        ]
        hit = hmmer_best.get(seq.record_id)
        detail.append(
            {
                "protein_id": seq.record_id,
                "length": feat.length,
                "molecular_weight": round(feat.molecular_weight, 2),
                "isoelectric_point": round(feat.isoelectric_point, 2),
                "aromaticity": round(feat.aromaticity, 4),
                "instability_index": round(feat.instability_index, 2),
                "gravy": round(feat.gravy, 4),
                "aliphatic": round(feat.aliphatic, 2),
                "n_masked": seq.n_masked,
                "sequence": seq.cleaned,
                "aa_class": [AA_CLASS.get(aa, "special") for aa in seq.cleaned],
                "tracks": tracks,
                "disorder": [round(s, 3) for s in disorder_scores(seq.cleaned)],
                "disordered_fraction": rec.disordered_fraction,
                "embedding_source": rec.embedding_source,
                "aa_freq": {k: round(float(v), 4) for k, v in sorted(feat.aa_freq.items())},
                "go_terms": go_terms,
                "hmmer": (
                    {"target": hit.target_id, "evalue": hit.evalue, "bitscore": hit.bitscore}
                    if hit
                    else None
                ),
                "hmmer_hit": rec.hmmer_hit,
                "vfdb": bool(hit and is_vfdb_hit(hit.target_id)),
                "predicted_function": rec.predicted_function,
            }
        )
    return records, detail, eval_report


def records_to_tsv(records: list[AnnotationRecord]) -> str:
    """Serialise annotation records to a TSV string (same contract as CLI)."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t")
    writer.writerow(TSV_COLUMNS)
    for record in records:
        writer.writerow(record.to_row())
    return buf.getvalue()


@app.get("/health")
def health() -> dict:
    """Liveness probe with capability flags."""
    return {
        "status": "ok",
        "service": "protein-annotation",
        "esm_default": DEFAULT_ESM_MODEL,
        "go_terms": len(GO_IDS),
        "hmmer_available": hmmer_available(),
    }


@app.post("/annotate")
async def annotate(
    file: UploadFile | None = File(default=None),
    fasta: str | None = Form(default=None),
    hmmer_db: str | None = Form(default=None),
    eval_labels: str | None = Form(default=None),
    top_k: int = Form(default=3),
    no_esm: bool = Form(default=True),
    esm_model: str = Form(default=DEFAULT_ESM_MODEL),
) -> JSONResponse:
    """Annotate uploaded FASTA (file or pasted text) -> JSON detail + TSV."""
    fasta_text = ""
    if file is not None and file.filename:
        raw = await file.read()
        fasta_text = raw.decode("utf-8", errors="replace")
    elif fasta:
        fasta_text = fasta
    if not fasta_text.strip():
        return JSONResponse({"error": "Provide a FASTA file or pasted FASTA text"}, status_code=400)
    top_k = max(1, min(int(top_k), len(GO_IDS)))
    started = time.perf_counter()
    try:
        records, detail, eval_report = annotate_fasta_text(
            fasta_text,
            top_k=top_k,
            esm_model=esm_model,
            force_fallback=bool(no_esm),
            hmmer_db_text=hmmer_db,
            eval_labels_text=eval_labels,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    # Run provenance for the inspector + status bar (Benchling-style audit info).
    meta = {
        "n_proteins": len(records),
        "embeddings": "hashed fallback (offline)" if no_esm else "ESM-2",
        "esm_model": esm_model,
        "hmmer_available": hmmer_available(),
        "hmmer_db_provided": bool(hmmer_db and hmmer_db.strip()),
        "hmmer_db_kind": hmmer_db_kind(hmmer_db),
        "hmmer_hits": sum(1 for r in records if r.hmmer_hit != "-"),
        "vfdb_hits": sum(1 for d in detail if d["vfdb"]),
        "go_terms": len(GO_IDS),
        "go_weights": "demo random-init (untrained)",
        "top_k": top_k,
        "eval": eval_report,
        "elapsed_s": time.perf_counter() - started,
    }
    return JSONResponse(
        {"records": detail, "tsv": records_to_tsv(records), "columns": TSV_COLUMNS, "meta": meta}
    )


@app.get("/")
def index() -> FileResponse:
    """Serve the domain-style protein workbench UI."""
    return FileResponse(str(ROOT / "static" / "index.html"))
