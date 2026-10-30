# Removes %LOCALAPPDATA%\WaveAnalysis and the shortcuts. Your data and results are not touched.
$root = Join-Path $env:LOCALAPPDATA 'WaveAnalysis'
foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
    foreach ($n in @('SURF.lnk', 'SURF (console).lnk', 'Wave Analysis.lnk', 'Wave Analysis (console).lnk')) {
        $p = Join-Path $dir $n
        if (Test-Path $p) { Remove-Item -Force $p }
    }
}
if (Test-Path $root) { Remove-Item -Recurse -Force $root }
Write-Host 'SURF removed.'
