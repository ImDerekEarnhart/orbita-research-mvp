$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    Write-Host "Orbita is not installed yet. Running install.ps1..." -ForegroundColor Yellow
    powershell -ExecutionPolicy Bypass -File .\install.ps1
}

$serverScript = Join-Path $PSScriptRoot "start_mvp.ps1"
Start-Process -FilePath "powershell.exe" -WorkingDirectory $PSScriptRoot -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$serverScript`""
)

Start-Sleep -Seconds 3
Start-Process "http://127.0.0.1:8010/"
