# Protein Annotation Pipeline

End-to-end CLI pipeline that annotates **hypothetical proteins** from FASTA:

1. **Parse + clean** — Biopython `SeqIO`; ambiguous residues (`X/B/Z/U/O`)
   are substituted (`X→A, B→N, Z→Q, U→C, O→K`) and tracked with a
   per-residue validity mask.
2. **Physicochemical features** — `Bio.SeqUtils.ProtParam` (length, MW, pI,
   aromaticity, instability, GRAVY), Ikai aliphatic index, 20-dim AA
   frequencies → 27-dim vector.
3. **ESM-2 embeddings** — mean-pooled per-residue representations
   (`esm2_t6_8M_UR50D` default, `esm2_t33_650M_UR50D` via `--esm-model`),
   mask-aware. Offline or missing `torch`/`esm` → deterministic hashed
   k-mer fallback (same width), with a warning.
4. **HMMER `phmmer`** — optional subprocess homology search (`--hmmer-db`);
   parses `--tblout` e-value/bit score; skipped with a warning when the
   binary is absent.
5. **GO multilabel head** — PyTorch `Linear→BatchNorm→GELU→Dropout(0.3)→Linear`
   over 12 demo GO terms (BP/MF/CC); per-term F1 threshold tuning on a
   validation split (`tune_thresholds`); uniform-score dummy when torch
   is unavailable.
6. **TSV report** — `protein_annotations.tsv` with columns
   `Protein_ID, Sequence_Length, Molecular_Weight, Top_GO_Terms,
   Confidence_Scores, HMMER_Top_Hit, Predicted_Function`.

No Streamlit: the core CLI is UI-independent, and the web workbench is
FastAPI + static HTML/JS (vanilla + Plotly CDN).

## Install

```bash
pip install -r requirements.txt
```

HMMER (`phmmer`) is optional and must come from your system package manager.

## Web workbench (unified app)

```bash
uvicorn server:app --reload
# open http://127.0.0.1:8000/
```

Benchling/Geneious-style workbench: run setup + searchable protein
navigator (left), tabbed viewer with GO evidence table, ruler-based sequence
viewer with region selection, stacked hydrophobicity/charge tracks and AA
composition (center), homology + run-provenance inspector (right), export menu
(TSV/JSON/FASTA), and a status bar. No Streamlit.

API:

- `GET /health` — liveness + capability flags
- `POST /annotate` — multipart `file` (FASTA) or `fasta` text field,
  plus `top_k`, `no_esm`, `esm_model`, `hmmer_db` (reference FASTA text);
  returns `{records: [...], tsv: "...", columns: [...], meta: {...}}`

## CLI usage

```bash
# Offline-safe demo (hashed embeddings, no HMMER)
python -m protein_annotation hypothetical_proteins.fasta -o protein_annotations.tsv --no-esm

# Full run with ESM-2 8M + homology search
python -m protein_annotation hypothetical_proteins.fasta --hmmer-db reference.fasta

# Large ESM-2 model
python -m protein_annotation hypothetical_proteins.fasta --esm-model esm2_t33_650M_UR50D
```

Run from the repo root with `PYTHONPATH=src`, or `pip install -e .` if you
add packaging later. Tests set `sys.path` automatically.

## Layout

```text
src/protein_annotation/
  sequence.py    cleaning/masking + ProtParam features
  embeddings.py  ESM-2 + hashed fallback
  hmmer.py       phmmer wrapper + tblout parser
  model.py       GO multilabel head + threshold tuning
  pipeline.py    orchestration + TSV writer
  cli.py         argparse entry point
hypothetical_proteins.fasta   6 synthetic bacterial sequences (one with X/B/Z/U/O)
tests/test_pipeline.py        smoke tests
```

## Notes

- GO head ships with **randomly initialised demo weights**; call
  `GOClassifier.fit(X_train, Y_train, X_val, Y_val)` to train on labelled
  data, which also tunes per-term thresholds.
- `fair-esm` downloads weights from the internet on first use; without
  network access the pipeline warns and continues with hashed embeddings.
