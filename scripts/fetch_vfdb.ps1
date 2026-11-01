# Download the VFDB core virulence-factor protein set for use as a
# `--hmmer-db` reference database (Windows equivalent of fetch_vfdb.sh).
#
# License note: VFDB data is CC BY-NC 4.0 (non-profit/research use only).
# Source: https://www.mgc.ac.cn/VFs/download.htm
#
# Usage: powershell -File scripts\fetch_vfdb.ps1 [-OutDir data\vfdb]
param([string]$OutDir = "data\vfdb")
$ErrorActionPreference = "Stop"

$Url = "https://www.mgc.ac.cn/VFs/Down/VFDB_setA_pro.fas.gz"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Gz = Join-Path $OutDir "VFDB_setA_pro.fas.gz"
Write-Host "Downloading VFDB core set (CC BY-NC 4.0, research-only) ..."
Invoke-WebRequest -Uri $Url -OutFile $Gz
# Gunzip via .NET (no 7-Zip dependency).
$Out = Join-Path $OutDir "VFDB_setA_pro.fas"
$InStream = [System.IO.File]::OpenRead($Gz)
$Gzip = New-Object System.IO.Compression.GzipStream($InStream, [System.IO.Compression.CompressionMode]::Decompress)
$OutStream = [System.IO.File]::Create($Out)
$Gzip.CopyTo($OutStream)
$OutStream.Close(); $Gzip.Close(); $InStream.Close()
Remove-Item $Gz
Write-Host "Wrote $Out"
Write-Host "Use: python -m protein_annotation input.fasta --hmmer-db $Out"
