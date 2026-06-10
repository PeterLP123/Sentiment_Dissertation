$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$sentimentBench = Join-Path $repoRoot ".venv\Scripts\sentiment-bench.exe"

if (-not (Test-Path $sentimentBench)) {
    Write-Error "Virtual environment not found. Run .\scripts\setup_windows_py312.ps1 from the repo first."
}

Push-Location $repoRoot
try {
    & $sentimentBench tui
}
finally {
    Pop-Location
}
