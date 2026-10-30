# SURF (Spontaneous Undulation in Retina Finder) - Windows installer
# Copies the app to %LOCALAPPDATA%\WaveAnalysis, creates a private Python environment there
# (outside OneDrive), installs the packages and adds Desktop + Start-menu shortcuts.
# Run via install_windows.bat (double-click).

$ErrorActionPreference = 'Stop'
$src  = Split-Path -Parent $PSScriptRoot          # the unzipped SURF folder
$root = Join-Path $env:LOCALAPPDATA 'WaveAnalysis'
$app  = Join-Path $root 'app'
$venv = Join-Path $root 'venv'

Write-Host ''
Write-Host '=== SURF - Spontaneous Undulation in Retina Finder - installer ===' -ForegroundColor Cyan
Write-Host "Install folder: $root"
Write-Host ''

function Test-Python($cmd) {
    $exe = $cmd[0]
    $rest = @()
    if ($cmd.Count -gt 1) { $rest = $cmd[1..($cmd.Count - 1)] }
    try {
        $v = & $exe @rest -c "import sys; print('%d.%d' % sys.version_info[:2]); sys.exit(0 if sys.version_info >= (3, 9) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { return $v }
    } catch {}
    return $null
}

# 1. find a Python >= 3.9  (python.org launcher first, then PATH, then Anaconda/Miniconda)
$candidates = @()
if (Get-Command py -ErrorAction SilentlyContinue)     { $candidates += ,@('py', '-3') }
if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += ,@('python') }
foreach ($p in @("$env:USERPROFILE\anaconda3\python.exe", "$env:USERPROFILE\miniconda3\python.exe",
                 "$env:ProgramData\anaconda3\python.exe", "$env:ProgramData\miniconda3\python.exe",
                 "$env:LOCALAPPDATA\anaconda3\python.exe", "$env:LOCALAPPDATA\miniconda3\python.exe")) {
    if (Test-Path $p) { $candidates += ,@($p) }
}
$py = $null
foreach ($c in $candidates) {
    $ver = Test-Python $c
    if ($ver) { $py = $c; Write-Host "Using Python $ver  ($($c -join ' '))"; break }
}
if (-not $py) {
    Write-Host 'No Python 3.9+ found.' -ForegroundColor Red
    Write-Host 'Install it from https://www.python.org/downloads/ (tick "Add python.exe to PATH"),'
    Write-Host 'then double-click install_windows.bat again.'
    exit 1
}
$pyExe = $py[0]
$pyArgs = @()
if ($py.Count -gt 1) { $pyArgs = $py[1..($py.Count - 1)] }

# 2. copy the app
New-Item -ItemType Directory -Force -Path $app | Out-Null
foreach ($f in @('wave_gui.py', 'wave_core.py', 'wave_plots.py', 'wave_stats.py', 'compare_tab.py', 'surf_calibrate.py', 'presets.py', 'info_tab.py', 'player.py', 'mpl_nav.py', 'figexport.py', 'cells_core.py', 'cells_tab.py', 'surf.ico', 'surf.png', 'requirements.txt', 'README.md')) {
    Copy-Item -Force (Join-Path $src $f) $app
}
Copy-Item -Recurse -Force (Join-Path $src 'logo') $app
Copy-Item -Recurse -Force (Join-Path $src 'defaults') $app
Copy-Item -Force (Join-Path $PSScriptRoot 'uninstall_windows.ps1') $root
Copy-Item -Force (Join-Path $PSScriptRoot 'install_cells_addon.ps1') $root

# 3. private environment + packages
$vpy = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $vpy)) {
    Write-Host 'Creating the Python environment...'
    & $pyExe @pyArgs -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
Write-Host 'Installing packages (first time only: a few minutes, several hundred MB)...'
& $vpy -m pip install --disable-pip-version-check --upgrade pip | Out-Null
& $vpy -m pip install --disable-pip-version-check -r (Join-Path $app 'requirements.txt')
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host 'Package installation failed.' -ForegroundColor Red
    Write-Host 'If the error mentions SSL and you use Anaconda, install Python from python.org and run'
    Write-Host 'the installer again (delete the folder above first).'
    exit 1
}
& $vpy -c "import PySide6, pyqtgraph, cv2, sklearn, scipy, roifile, openpyxl, statsmodels; print('All packages OK')"
if ($LASTEXITCODE -ne 0) { throw 'Package check failed.' }

# 4. shortcuts (old "Wave Analysis" shortcuts are replaced by "SURF")
$ws = New-Object -ComObject WScript.Shell
$targets = @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))
foreach ($dir in $targets) {
    foreach ($old in @('Wave Analysis.lnk', 'Wave Analysis (console).lnk')) {
        $op = Join-Path $dir $old
        if (Test-Path $op) { Remove-Item -Force $op }
    }
    $lnk = $ws.CreateShortcut((Join-Path $dir 'SURF.lnk'))
    $lnk.TargetPath       = Join-Path $venv 'Scripts\pythonw.exe'
    $lnk.Arguments        = '"' + (Join-Path $app 'wave_gui.py') + '"'
    $lnk.WorkingDirectory = $app
    $lnk.IconLocation     = (Join-Path $app 'surf.ico') + ',0'
    $lnk.Description      = 'SURF - Spontaneous Undulation in Retina Finder'
    $lnk.Save()
}
# a console variant that shows error messages, for troubleshooting
$lnk = $ws.CreateShortcut((Join-Path $targets[1] 'SURF (console).lnk'))
$lnk.TargetPath       = $vpy
$lnk.Arguments        = '"' + (Join-Path $app 'wave_gui.py') + '"'
$lnk.WorkingDirectory = $app
$lnk.IconLocation     = (Join-Path $app 'surf.ico') + ',0'
$lnk.Save()

# refresh the Windows icon cache so the new icon shows immediately
try { Start-Process -FilePath "$env:SystemRoot\System32\ie4uinit.exe" -ArgumentList '-show' -WindowStyle Hidden -Wait } catch {}

Write-Host ''
Write-Host 'Done. Start "SURF" from the Desktop or the Start menu.' -ForegroundColor Green
Write-Host 'To update later: unzip the new version and run install_windows.bat again.'
