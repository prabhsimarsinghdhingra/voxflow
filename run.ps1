# VoxFlow setup. Usage: .\run.ps1 [-Test] [-Bench] [-Serve]   (no switch = Test, then Serve)
param([switch]$Test, [switch]$Bench, [switch]$Serve)
$ErrorActionPreference = "Stop"
if (-not ($Test -or $Bench -or $Serve)) { $Test = $true; $Serve = $true }
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
if (Test-Path .env) {
  Get-Content .env | Where-Object { $_ -match '^\s*[^#\s][^=]*=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2; Set-Item "env:$($k.Trim())" $v.Trim() }
}
if (-not $env:VLLM_URL) { $env:VLLM_URL = "http://localhost:8001/v1" }
if ($Test) { Write-Host "`n== Unit tests ==" -ForegroundColor Cyan; python -m pytest -q; if ($LASTEXITCODE) { throw "Tests failed" } }
if ($Bench -or $Serve) {
  try { Invoke-WebRequest "$($env:VLLM_URL)/models" -UseBasicParsing -TimeoutSec 3 | Out-Null }
  catch { throw "LLM server not reachable at $($env:VLLM_URL). Start vLLM or Ollama first (see README step 3)." }
}
if ($Bench) { Write-Host "`n== Benchmark (real models) ==" -ForegroundColor Cyan; python -m voxflow.benchmark }
if ($Serve) { Write-Host "`nOpen http://localhost:8000/  (mic client)  |  /dashboard/  |  /docs" -ForegroundColor Green
  uvicorn voxflow.server:app --port 8000 }
