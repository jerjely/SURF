@echo off
rem Removes the app, its Python environment and the shortcuts. Data and results are not touched.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install\uninstall_windows.ps1"
pause
