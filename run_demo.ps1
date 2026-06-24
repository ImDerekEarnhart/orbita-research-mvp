$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& .\.venv\Scripts\orbita-mvp.exe --db .\demo.db --workspace .\demo_workspace demo .\examples\marker_response.csv --name "Open discovery demo"
