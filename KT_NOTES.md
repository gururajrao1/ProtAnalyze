# Knowledge Transfer Notes — Protein Annotation Workbench

Short handoff for the signal-expansion round: VFDB virulence tier, real GO
labels + trained head, CAFA eval harness, offline disorder signal, and the
web-UI wiring for all four. No Streamlit anywhere; core CLI stays
UI-independent, web app is FastAPI + static HTML/JS.

## What was researched (with sources)

- **VFDB virulence tier** — VFDB SetA core dataset (`VFDB_setA_pro.fas`,
  ~1.3 MB, CC BY-NC 4.0) reused through the existing `phmmer` path as the
  `--hmmer-db` value. Sources: [VFDB](http://www.mgc.ac.cn/VFs/),
  [HMMER](http://hmmer.org/).
- **Bacterial GO labels + trained head** — filter GOA GAF to bacterial
  experimental annotations, true-path propagate via `go-basic.obo`, prune to
  100–500 terms, train `GOClassifier.fit`. Sources:
  [GO ontology downloads](http://geneontology.org/docs/download-ontology/),
  [EBI GOA](https://www.ebi.ac.uk/GOA/),
  [AmiGO 2](https://amigo.geneontology.org/amigo) (linked from the UI).
- **CAFA eval harness** — per-term F1 plus protein-centric Fmax, IC-weighted
  Smin, micro-AUPR, CAFA-evaluator-compatible semantics. Source:
  [CAFA challenge](https://biofunctionprediction.org/cafa/).
- **Disorder signal** — TOP-IDP-style propensity + window-9 smoothing as a
  pure-CPU IUPred3-style fallback (no new deps). Source:
  [IUPred3](https://iupred3.elte.hu/).
- **Embeddings baseline** — frozen ESM-2 8M default + deterministic hashed
  k-mer fallback (offline). Source: [fair-esm](https://github.com/facebookresearch/esm).

## What got implemented and why

| Signal | Implementation | Why |
|---|---|---|
| VFDB tier | `scripts/fetch_vfdb.{sh,ps1}` + zero pipeline change; `server.py` classifies DB kind (`vfdb`/`custom`/`none`) and flags `VFG*` hits | Direct virulence signal for BROAD pathogens at zero code cost |
| GO labels + head | `src/protein_annotation/go_labels.py` (`parse_gaf`, `parse_obo_ancestors`, `propagate`, `prune_terms`, `build_label_matrix`), `scripts/build_go_labels.py`, `GOClassifier.save/load_weights`, CLI `--go-weights/--go-terms` | Replaces random-init demo weights with real bacterial labels |
| CAFA eval | `model.cafa_eval_report` + CLI `--eval-labels/--eval-report` sidecar; `/annotate` accepts `eval_labels` and returns `meta.eval` | Grounds demo metrics in the community standard, zero runtime cost |
| Disorder | `sequence.disorder_scores/disorder_fraction`; `AnnotationRecord.disordered_fraction/.embedding_source`; `--extended-tsv` (9-col) / `--no-disorder`; disorder track + fraction + source in `/annotate` JSON | Explains hypothetical-protein failures on orphans, orthogonal to ESM confidence |
| Web UI | Badges (embedding source, disorder %, VFDB), disorder rows in Overview, 3rd disorder track with 0.5 threshold, VFDB badge in Homology panel, new CAFA panel + provenance rows, eval-labels input | Surfaces every new signal Benchling/Geneious-style; existing views untouched |

Default 7-column TSV is unchanged (backward compatible); new signals go to
JSON/UI, `--extended-tsv`, or the eval sidecar.

## How to test each feature (exact commands, from repo root)

```powershell
# 0. Prereqs + full suite (20 tests: 8 pre-existing + 12 new)
pip install -r requirements.txt
python -m pytest tests/ -q

# 1. VFDB tier (parser-level; needs phmmer + download for end-to-end)
bash scripts/fetch_vfdb.sh   # or: powershell -File scripts\fetch_vfdb.ps1
$env:PYTHONPATH = "src"
python -m protein_annotation hypothetical_proteins.fasta --no-esm --hmmer-db data/vfdb/VFDB_setA_pro.fas

# 2. GO labels + trained head
python scripts/build_go_labels.py goa_uniprot_all.gaf.gz go-basic.obo --out-terms data/go_bacterial_terms.json --out-labels data/go_labels.npz
python -m protein_annotation hypothetical_proteins.fasta --no-esm --go-weights data/head -o protein_annotations.tsv

# 3. CAFA eval sidecar (needs labels TSV: protein_id<TAB>GO:1,GO:2)
python -m protein_annotation hypothetical_proteins.fasta --no-esm --eval-labels labels.tsv --eval-report eval.json

# 4. Disorder + extended TSV
python -m protein_annotation hypothetical_proteins.fasta --no-esm --extended-tsv -o protein_annotations_ext.tsv
python -m protein_annotation hypothetical_proteins.fasta --no-esm --no-disorder -o protein_annotations.tsv

# 5. Web UI + API smoke
node --check static/app.js
uvicorn server:app --reload   # open http://127.0.0.1:8000/
python -c "from fastapi.testclient import TestClient; from server import app; c = TestClient(app); r = c.post('/annotate', data={'fasta': open('hypothetical_proteins.fasta').read(), 'no_esm': 'true'}); print(r.status_code, sorted(r.json()['meta'].keys()))"
```

## What was deferred and why

- **AMRFinderPlus wrapper** — high AMR value, public domain; needs a new
  subprocess module + graceful skip + Linux-only/WSL handling. Do after VFDB
  proves the multi-DB pattern.
- **DiamondScore homology blend + GO-DAG smoothing** — needs the TOP2
  trained head + calibrated scores first, else it blends noise.
- **DeepSig signal peptides** — needs wrapper + `--gram` flag + TF-era
  dep/container validation. SignalP6/DeepLoc2 stay out (license-gated
  weights break offline-first).
- **PSORTb / CARD-RGI** — PSORTb is Linux-only Perl with unverified
  download; CARD has a commercial-fee license. Optional backends only.
- **Embedding/structure scale-ups** (ProtT5-XL, ProstT5 3Di, Foldseek,
  ESMFold/AFDB, DeepGO-SE, LoRA) — need GPU/GB-downloads/online; keep
  frozen ESM-2 + hashed default on laptop CPU.
- **TALE reimplementation / full CAFA ensembles / operon context** —
  unmaintained stacks or wrong label space; operon needs genome/GFF mode
  (single-protein input has no coordinates).

## Batch 2 — AMR scan + homology blend (autonomous follow-up)

Shipped two more shortlist items without changing the default TSV contract:

| Signal | Implementation | Why |
|---|---|---|
| AMRFinderPlus AMR scan | `src/protein_annotation/amr.py` (`amr_available`, header-driven `parse_amr_tsv`, `run_amrfinder` with graceful skip, `hits_per_query`); CLI `--run-amrfinder/--amrfinder-db/--amr-report`; `/annotate` `run_amrfinder` field; per-record `amr` hits + Homology-panel table + provenance | Best offline AMR fit license-wise (NCBI public domain); protein-input mode matches our FASTA |
| Homology blend + DAG smoothing | `model.parse_go_map/homolog_scores_from_hits/blend_scores/smooth_over_ancestors`; CLI `--go-map/--blend-alpha/--obo`; `/annotate` `go_map/blend_alpha/obo` fields; `record.blended` flag + badges + provenance | DiamondScore-portable idea: bitscore-weighted GO transfer blended with head sigmoids, then true-path ancestor smoothing |

Staph-specific fetch script was evaluated and **skipped**: the curated
`macgenomegue/Staphylococcus_VFs` DB is nucleotide, not protein, so it cannot
feed the `phmmer` protein path; VFDB protein already covers S. aureus.

```powershell
# Batch-2 tests (11 new; 31 total with batch 1)
python -m pytest tests/test_batch2.py -q

# Blend + smoothing offline (homolog arm abstains without phmmer hits)
python -m protein_annotation hypothetical_proteins.fasta --no-esm --go-map map.tsv --blend-alpha 0.5 --obo go-basic.obo

# AMR scan (warns + header-only sidecar when the Linux-only binary is absent)
python -m protein_annotation hypothetical_proteins.fasta --no-esm --run-amrfinder --amr-report amr.tsv
```

Caveat carried forward: the blend is a mechanism, not yet a quality gain —
it needs the trained head (batch-1 builder) plus calibrated scores before the
homolog arm helps; with demo weights use `--blend-alpha 1.0` (default).

## Training run — UniProt bacterial head v1 (autonomous)

`scripts/train_uniprot_head.py` paged 3,000 reviewed bacterial proteins
(UniProt REST, `reviewed:true + taxonomy_id:2`), propagated GO terms via
`go-basic.obo`, kept 2,788 hitting the demo-12 space, trained 80/10/10
(hashed embeddings, 30 epochs, CPU): **test Fmax 0.743, Smin 1.844,
micro-AUPR 0.733 (n=279)**. Weights at `data/head_demo12.{pt,json}`
(gitignored) + `data/train_report.json`. Two terms had zero positives in
this sample (`GO:0004871` signal transducer, `GO:0016021` membrane) —
expected to stay weak until the GAF-scale build. CLI: `--go-weights
data/head_demo12`; the workbench auto-loads the bundle on the hashed
path only (ESM vectors share the width, not the semantics) and the
About panel + provenance report which head scored each run.

## Known limitations

- **Demo weights**: the GO head ships randomly initialised; scores are
  ranking signals until `GOClassifier.fit` trains on real labels.
- **VFDB end-to-end untested**: fetch scripts not executed here and
  `phmmer` is absent on this host — only the tblout-parser fixture ran.
- **Disorder is heuristic**: a smoothed propensity scale, not IUPred3/Dishegg
  proper; good for flagging, not for publication-grade disorder maps.
- **CPU-only**: ESM-2 650M and the deferred structure tiers are impractical
  on laptop CPU; use `--no-esm` for offline/fast runs.
- **Charts need the Plotly CDN**; offline, the JSON export still carries
  all per-residue values.
