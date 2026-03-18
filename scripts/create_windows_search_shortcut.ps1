param(
    [string]$ShortcutName = "Japanese Teacher",
    [string]$IconPath,
    [int]$IconIndex = 0
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$runScript = Join-Path $projectRoot "scripts\run.ps1"
$programsDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$shortcutPath = Join-Path $programsDir ("{0}.lnk" -f $ShortcutName)
$defaultIconLocation = "$env:SystemRoot\System32\shell32.dll,220"
$projectIconPng = Join-Path $projectRoot "icon.png"
$projectIconIco = Join-Path $projectRoot "icon.ico"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pythonCommand = if (Test-Path $venvPython) { $venvPython } else { "python" }

if (-not (Test-Path $runScript)) {
    Write-Error "run.ps1 was not found at $runScript"
    exit 1
}

function Convert-ImageToIco {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ImagePath,
        [Parameter(Mandatory = $true)]
        [string]$IcoPath
    )

    $pythonScriptPath = [System.IO.Path]::GetTempFileName() + ".py"
    $pythonScript = @"
from PIL import Image
import sys

src = sys.argv[1]
dst = sys.argv[2]

img = Image.open(src).convert("RGBA")
w, h = img.size
size = max(w, h)
canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
canvas.paste(img, ((size - w) // 2, (size - h) // 2))

sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (24, 24), (16, 16)]
canvas.save(dst, format="ICO", sizes=sizes)
"@

    try {
        Set-Content -Path $pythonScriptPath -Value $pythonScript -Encoding UTF8
        & $pythonCommand $pythonScriptPath $ImagePath $IcoPath
        if ($LASTEXITCODE -ne 0) {
            throw "Python icon conversion failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Remove-Item $pythonScriptPath -ErrorAction SilentlyContinue
    }
}

function Test-IcoFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$IcoPath
    )

    if (-not (Test-Path $IcoPath)) {
        return $false
    }

    try {
        Add-Type -AssemblyName System.Drawing
        $icon = New-Object System.Drawing.Icon($IcoPath)
        $null = $icon.Handle
        return $icon.Width -gt 0 -and $icon.Height -gt 0
    }
    catch {
        return $false
    }
    finally {
        if ($icon) { $icon.Dispose() }
    }
}

if (-not $IconPath -and (Test-Path $projectIconPng)) {
    $IconPath = $projectIconPng
}

$iconLocation = $defaultIconLocation
if ($IconPath) {
    $resolvedIconPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($IconPath)
    if (-not (Test-Path $resolvedIconPath)) {
        Write-Error "Icon path was not found: $resolvedIconPath"
        exit 1
    }

    $extension = [System.IO.Path]::GetExtension($resolvedIconPath).ToLowerInvariant()
    if ($extension -eq ".ico") {
        $iconLocation = "$resolvedIconPath,0"
    }
    elseif ($extension -eq ".dll" -or $extension -eq ".exe") {
        $iconLocation = "$resolvedIconPath,$IconIndex"
    }
    elseif ($extension -eq ".png" -or $extension -eq ".jpg" -or $extension -eq ".jpeg" -or $extension -eq ".bmp") {
        try {
            Convert-ImageToIco -ImagePath $resolvedIconPath -IcoPath $projectIconIco
        }
        catch {
            Write-Error "Failed to convert image to icon. Ensure Pillow is installed for '$pythonCommand'. Error: $($_.Exception.Message)"
            exit 1
        }

        if (-not (Test-IcoFile -IcoPath $projectIconIco)) {
            Write-Error "Failed to generate a valid icon file at $projectIconIco"
            exit 1
        }

        $iconLocation = "$projectIconIco,0"
    }
    else {
        Write-Error "Unsupported icon format '$extension'. Use .png, .jpg, .jpeg, .bmp, .ico, .dll, or .exe."
        exit 1
    }
}

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$runScript`""
$shortcut.WorkingDirectory = $projectRoot
$shortcut.IconLocation = $iconLocation
$shortcut.Description = "Launch Japanese Teacher"
$shortcut.Save()

Write-Host "Shortcut created: $shortcutPath" -ForegroundColor Green
Write-Host "Icon source: $iconLocation" -ForegroundColor DarkCyan
Write-Host "Open Windows Search and type '$ShortcutName' to launch the app." -ForegroundColor Cyan

