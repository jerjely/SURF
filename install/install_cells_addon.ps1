# SURF - optional cell-detection add-on (Suite2p + Cellpose + PyTorch CPU)
# Runs while SURF is CLOSED (Windows locks the files of loaded packages).
# Started by SURF (Cells tab > Install) or by double-clicking install_cells_addon.bat.
param([int]$WaitForPid = 0, [string]$RestartWith = "")

$ErrorActionPreference = 'Stop'
$root = Join-Path $env:LOCALAPPDATA 'WaveAnalysis'
$Host.UI.RawUI.WindowTitle = 'SURF - installing Suite2p + Cellpose'
try { Start-Transcript -Path (Join-Path $root 'addon_install.log') -Force | Out-Null } catch {}
$vpy  = Join-Path $root 'venv\Scripts\python.exe'
Write-Host ''
Write-Host '=== SURF - installing the cell-detection add-on (Suite2p + Cellpose) ===' -ForegroundColor Cyan
if (-not (Test-Path $vpy)) {
    Write-Host "SURF's Python environment was not found ($vpy). Run install_windows.bat first." -ForegroundColor Red
    Read-Host 'Press Enter to close'; exit 1
}
if ($WaitForPid -gt 0) {
    Write-Host 'Waiting for SURF to close...'
    try { Wait-Process -Id $WaitForPid -Timeout 120 -ErrorAction SilentlyContinue } catch {}
}
# any other SURF window still open would lock files
$busy = Get-Process python, pythonw -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$root*" }
if ($busy) {
    Write-Host 'SURF is still running. Close every SURF window, then press Enter.' -ForegroundColor Yellow
    Read-Host
}
function Pip([string[]]$a) {
    & $vpy -m pip install --disable-pip-version-check @a
    if ($LASTEXITCODE -ne 0) { throw "pip failed: $($a -join ' ')" }
}
try {
    Write-Host "`n[1/3] PyTorch and the other libraries (the largest download, a few minutes)..."
    Pip @('torch', 'torchvision', 'numba', 'natsort', 'tifffile', 'tqdm', 'fastremap', 'imagecodecs',
          'roifile', 'fill-voids', 'segment_anything', 'importlib-metadata')
    Write-Host "`n[2/3] Suite2p and Cellpose (without their own OpenCV copy, SURF's is used)..."
    Pip @('--no-deps', 'suite2p>=1.0,<2', 'cellpose>=4,<5')
    Write-Host "`n[3/3] Checking OpenCV..."
    & $vpy -m pip show opencv-python-headless *> $null
    if ($LASTEXITCODE -eq 0) {
        & $vpy -m pip uninstall -y opencv-python-headless
        $cv = (& $vpy -m pip show opencv-python | Select-String '^Version:').ToString().Split(':')[1].Trim()
        Pip @('--force-reinstall', '--no-deps', "opencv-python==$cv")
    }
    & $vpy -c "import cv2, torch, suite2p, cellpose; print('OK - OpenCV', cv2.__version__, '| PyTorch', torch.__version__)"
    if ($LASTEXITCODE -ne 0) { throw 'import check failed' }
    # leftovers of interrupted installs (folders called ~...) are harmless; remove them quietly
    Get-ChildItem (Join-Path $root 'venv\Lib\site-packages') -Directory -Filter '~*' -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "`nLog: $(Join-Path $root 'addon_install.log')"
    Write-Host "`nDone. Cellpose downloads its model the first time you use it (internet needed)." -ForegroundColor Green
} catch {
    Write-Host "`n$_" -ForegroundColor Red
    Write-Host 'The add-on could not be installed. SURF itself still works.' -ForegroundColor Red
}
if ($RestartWith) {
    Write-Host 'Starting SURF again...'
    Start-Process -FilePath $RestartWith
}
try { Stop-Transcript | Out-Null } catch {}
Read-Host 'Press Enter to close'
