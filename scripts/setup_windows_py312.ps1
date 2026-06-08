$ErrorActionPreference = "Stop"

$python = py -3.12 -c "import sys; print(sys.executable)" 2>$null
if (-not $python) {
    Write-Error "Python 3.12 is required. Install it from https://www.python.org/downloads/ or with: winget install Python.Python.3.12"
}

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

Write-Host ""
Write-Host "Setup complete. Activate with:"
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then run:"
Write-Host "sentiment-bench tui"
