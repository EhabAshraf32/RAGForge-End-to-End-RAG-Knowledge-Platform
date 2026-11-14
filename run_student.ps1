# Start the student chat Streamlit app.
#   .\run_student.ps1            -> http://localhost:8501
param([int]$Port = 8501)

$EnvRoot = "E:\From-Transformers-To-Proffisional\envs\mini-rag"
$env:PYTHONNOUSERSITE = "1"

Set-Location $PSScriptRoot
& "$EnvRoot\python.exe" -m streamlit run streamlit_app/student_app.py --server.port $Port --browser.gatherUsageStats false
