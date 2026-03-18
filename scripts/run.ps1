Write-Host "Starting Japanese Teacher..." -ForegroundColor Cyan

$projectRoot = Split-Path -Parent $PSScriptRoot
$activateScript = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
$appPath = Join-Path $projectRoot "app.py"

$ollamaRunning = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaRunning) {
    Write-Host "Starting Ollama..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

if (-not (Test-Path $activateScript)) {
    Write-Error "Virtual environment not found at $activateScript. Run .\scripts\setup_env.ps1 first."
    exit 1
}

Set-Location $projectRoot
. $activateScript
python $appPath
