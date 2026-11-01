#!/usr/bin/env bash
# Download the VFDB core virulence-factor protein set for use as a
# `--hmmer-db` reference database (zero pipeline code change: the existing
# run_phmmer(input_fasta, hmmer_db) + top_hit_per_query path handles it).
#
# License note: VFDB data is CC BY-NC 4.0 (non-profit/research use only).
# Source: https://www.mgc.ac.cn/VFs/download.htm
#
# Usage: bash scripts/fetch_vfdb.sh [out_dir]   (default: data/vfdb)
set -euo pipefail

OUT_DIR="${1:-data/vfdb}"
URL="https://www.mgc.ac.cn/VFs/Down/VFDB_setA_pro.fas.gz"

mkdir -p "$OUT_DIR"
echo "Downloading VFDB core set (CC BY-NC 4.0, research-only) ..."
curl -fL --retry 3 -o "$OUT_DIR/VFDB_setA_pro.fas.gz" "$URL"
gunzip -f "$OUT_DIR/VFDB_setA_pro.fas.gz"
echo "Wrote $OUT_DIR/VFDB_setA_pro.fas"
echo "Use: python -m protein_annotation input.fasta --hmmer-db $OUT_DIR/VFDB_setA_pro.fas"
