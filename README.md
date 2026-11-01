# Protein Annotation Pipeline

End-to-end CLI pipeline that annotates **hypothetical proteins** from FASTA.
It unifies the original ProtAnalyze notebook (preserved below) with a
production pipeline and web workbench.

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
viewer with region selection, stacked hydrophobicity/charge/disorder
tracks and AA composition (center), homology + CAFA eval +
run-provenance inspector (right), export menu (TSV/JSON/FASTA), and a
status bar. No Streamlit.

Per-protein badges surface the embedding source (`hashed`/`ESM-2`),
disordered fraction (flagged at ≥50%), VFDB virulence hits, blended
scores, and AMR hit counts; the inspector shows the HMMER DB kind,
GO-weights provenance, an optional CAFA report (Fmax/Smin/AUPR +
per-term F1), AMRFinderPlus hits, and blend/AMR provenance.

API:

- `GET /health` — liveness + capability flags (incl. `amrfinder_available`)
- `POST /annotate` — multipart `file` (FASTA) or `fasta` text field,
  plus `top_k`, `no_esm`, `esm_model`, `hmmer_db` (reference FASTA text),
  `eval_labels` (labels TSV text: `protein_id<TAB>GO:1,GO:2`),
  `go_map` (target→GO TSV text), `blend_alpha`, `obo` (OBO text),
  `run_amrfinder`;
  returns `{records: [...], tsv: "...", columns: [...], meta: {...}}`
  where each record carries `disorder`, `disordered_fraction`,
  `embedding_source`, `vfdb`, `blended`, `amr`, and `meta` carries
  `hmmer_db_kind`, `vfdb_hits`, `go_weights`, `eval` (null without
  labels), `blend`, and `amr`

## CLI usage

```bash
# Offline-safe demo (hashed embeddings, no HMMER)
python -m protein_annotation hypothetical_proteins.fasta -o protein_annotations.tsv --no-esm

# Full run with ESM-2 8M + homology search
python -m protein_annotation hypothetical_proteins.fasta --hmmer-db reference.fasta

# Large ESM-2 model
python -m protein_annotation hypothetical_proteins.fasta --esm-model esm2_t33_650M_UR50D

# Extended TSV (+Disordered_Fraction, +Embedding_Source), no disorder signal
python -m protein_annotation hypothetical_proteins.fasta --no-esm --extended-tsv
python -m protein_annotation hypothetical_proteins.fasta --no-esm --no-disorder

# Trained GO head + CAFA eval sidecar
python -m protein_annotation hypothetical_proteins.fasta --no-esm --go-weights data/head \
    --eval-labels labels.tsv --eval-report eval.json

# Homology blend (bitscore-weighted GO transfer + DAG smoothing)
python -m protein_annotation hypothetical_proteins.fasta --no-esm --hmmer-db reference.fasta \
    --go-map target_go.tsv --blend-alpha 0.5 --obo go-basic.obo

# AMRFinderPlus scan (Linux/macOS; warns + skips elsewhere)
python -m protein_annotation hypothetical_proteins.fasta --no-esm --run-amrfinder --amr-report amr.tsv
```

## Virulence tier (VFDB)

The existing `--hmmer-db` path doubles as a VFDB virulence tier with no
pipeline change — pass a VFDB FASTA as the reference database:

```bash
bash scripts/fetch_vfdb.sh            # or: powershell -File scripts\fetch_vfdb.ps1
python -m protein_annotation hypothetical_proteins.fasta --no-esm --hmmer-db data/vfdb/VFDB_setA_pro.fas
```

VFDB data is CC BY-NC 4.0 (research/non-profit use only). Hits land in the
unchanged `HMMER_Top_Hit` column; without `phmmer` the step warns and skips.

## Real GO labels + trained head

`scripts/build_go_labels.py` filters a GOA GAF to bacterial experimental
annotations, true-path propagates via `go-basic.obo`, frequency-prunes to
100–500 terms, and writes a terms JSON plus an optional labels NPZ:

```bash
python scripts/build_go_labels.py goa_uniprot_all.gaf.gz go-basic.obo \
    --out-terms data/go_bacterial_terms.json --out-labels data/go_labels.npz
```

Train with `GOClassifier.fit`, persist with `save_weights`, and reuse with
`--go-weights data/head` (or `--go-terms` for the label space alone). The
CAFA report (`--eval-report`) scores Fmax/Smin/micro-AUPR plus per-term F1
against a labels TSV (`protein_id<TAB>GO:1,GO:2`).

Run from the repo root with `PYTHONPATH=src`, or `pip install -e .` if you
add packaging later. Tests set `sys.path` automatically.

## Layout

```text
src/protein_annotation/
  sequence.py    cleaning/masking + ProtParam features + disorder
  embeddings.py  ESM-2 + hashed fallback
  hmmer.py       phmmer wrapper + tblout parser
  amr.py         AMRFinderPlus wrapper + tabular parser
  go_labels.py   GAF/OBO label build (true-path + prune)
  model.py       GO head + tuning + CAFA eval + homology blend
  pipeline.py    orchestration + TSV/AMR writers
  cli.py         argparse entry point
server.py + static/          FastAPI workbench (Benchling/Geneious-style UI)
scripts/                     fetch_vfdb, build_go_labels helpers
hypothetical_proteins.fasta  6 synthetic bacterial sequences (one with X/B/Z/U/O)
tests/                       pipeline + shortlist + batch-2 suites
KT_NOTES.md                  knowledge-transfer handoff
analyze.ipynb + hemo.*       legacy ProtAnalyze notebook (below)
```

## Legacy ProtAnalyze notebook

The original ProtAnalyze tool is preserved in this repo: put a protein
sequence in a FASTA file (e.g. `hemo.fasta`), run `analyze.ipynb`, and
read `hemo_analysis.txt` + `amino_acid_composition.png` (MW, pI,
aromaticity, instability, GRAVY, AA composition). Its reusable logic —
ProtParam analysis, the AA-composition chart, and the batch SeqIO loop —
is ported into `src/protein_annotation/` and the workbench above; the
notebook remains for teaching and quick single-protein checks.

## Notes

- GO head ships with **randomly initialised demo weights**; train real
  ones with `python scripts/train_uniprot_head.py` (≈3k reviewed bacterial
  proteins, Fmax 0.74 on held-out test), then reuse via
  `--go-weights data/head_demo12` (CLI) — the workbench auto-loads
  `data/head_demo12.json` on the hashed path when present.
- `fair-esm` downloads weights from the internet on first use; without
  network access the pipeline warns and continues with hashed embeddings.
