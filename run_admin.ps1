# Start the TA / admin Streamlit dashboard (keep this URL private).
#   .\run_admin.ps1              -> http://localhost:8502
param([int]$Port = 8502)

$EnvRoot = "E:\From-Transformers-To-Proffisional\envs\mini-rag"
$env:PYTHONNOUSERSITE = "1"

Set-Location $PSScriptRoot
& "$EnvRoot\python.exe" -m streamlit run streamlit_app/admin_app.py --server.port $Port --browser.gatherUsageStats false
