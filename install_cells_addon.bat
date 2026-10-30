@echo off
rem SURF - optional cell-detection add-on (Suite2p + Cellpose). Close SURF first.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install\install_cells_addon.ps1"
