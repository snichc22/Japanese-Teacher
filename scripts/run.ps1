Write-Host "Starting Japanese Teacher..." -ForegroundColor Cyan

# Check if Ollama is running
$ollamaRunning = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaRunning) {
    Write-Host "Starting Ollama..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

.\.venv\Scripts\Activate.ps1
python app.py