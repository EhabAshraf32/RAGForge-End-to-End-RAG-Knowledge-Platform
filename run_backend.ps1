# Start the FastAPI backend with the project environment on E: (no `conda activate` needed).
#   .\run_backend.ps1            -> http://127.0.0.1:8000/api/v1
#   .\run_backend.ps1 -Port 8010
param([int]$Port = 8000)

$EnvRoot = "E:\From-Transformers-To-Proffisional\envs\mini-rag"
$env:PYTHONNOUSERSITE = "1"                                              # ignore packages from C:\...\Roaming
$env:HF_HOME = "E:\From-Transformers-To-Proffisional\.hf_cache\hf_home"  # bge-m3 cache stays on E:
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Set-Location (Join-Path $PSScriptRoot "src")
# --workers 1: embedded Qdrant and the in-memory rate limiter are per-process
& "$EnvRoot\python.exe" -m uvicorn main:app --host 127.0.0.1 --port $Port --workers 1
