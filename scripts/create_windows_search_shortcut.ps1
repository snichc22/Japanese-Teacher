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

    Add-Type -AssemblyName System.Drawing
    if (-not ("NativeIcon" -as [type])) {
        Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class NativeIcon {
    [DllImport("user32.dll", SetLastError = true)]
    public static extern bool DestroyIcon(IntPtr hIcon);
}
"@
    }

    $bitmap = New-Object System.Drawing.Bitmap($ImagePath)
    $hIcon = [IntPtr]::Zero
    $icon = $null
    $stream = $null

    try {
        $hIcon = $bitmap.GetHicon()
        $icon = [System.Drawing.Icon]::FromHandle($hIcon)
        $stream = New-Object System.IO.FileStream($IcoPath, [System.IO.FileMode]::Create)
        $icon.Save($stream)
    }
    finally {
        if ($stream) { $stream.Dispose() }
        if ($icon) { $icon.Dispose() }
        if ($hIcon -ne [IntPtr]::Zero) { [NativeIcon]::DestroyIcon($hIcon) | Out-Null }
        $bitmap.Dispose()
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
        $iconLocation = $resolvedIconPath
    }
    elseif ($extension -eq ".dll" -or $extension -eq ".exe") {
        $iconLocation = "$resolvedIconPath,$IconIndex"
    }
    elseif ($extension -eq ".png" -or $extension -eq ".jpg" -or $extension -eq ".jpeg" -or $extension -eq ".bmp") {
        Convert-ImageToIco -ImagePath $resolvedIconPath -IcoPath $projectIconIco
        $iconLocation = $projectIconIco
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

