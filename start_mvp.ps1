$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    throw "The virtual environment does not exist. Run .\install.ps1 first."
}

$env:ORBITA_MVP_DB = Join-Path $PSScriptRoot "orbita_mvp.db"
$env:ORBITA_MVP_WORKSPACE = Join-Path $PSScriptRoot "orbita_workspace"

Write-Host "Orbita Research MVP: http://127.0.0.1:8010/" -ForegroundColor Cyan
Write-Host "Interactive API docs: http://127.0.0.1:8010/docs" -ForegroundColor Cyan
Write-Host "Leave this window open. Press CTRL+C to stop." -ForegroundColor Yellow

& .\.venv\Scripts\python.exe -m uvicorn orbita_mvp.api:app --host 127.0.0.1 --port 8010
