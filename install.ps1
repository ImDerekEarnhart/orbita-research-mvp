$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python 3.11 or newer is required and must be available as 'python'."
}

python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e .

Write-Host ""
Write-Host "Orbita Research MVP installed." -ForegroundColor Green
Write-Host "Start it with: powershell -ExecutionPolicy Bypass -File .\start_mvp.ps1"
