@echo off
rem Double-click to install SURF (no admin rights needed).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install\install_windows.ps1"
echo.
pause
