@echo off
rem Builds a standalone "SURF" folder (with SURF.exe) that runs on PCs without Python.
rem Run install_windows.bat first. Output: dist\SURF  - zip that folder and share it.
set VPY=%LOCALAPPDATA%\WaveAnalysis\venv\Scripts\python.exe
if not exist "%VPY%" (echo Run install_windows.bat first. & pause & exit /b 1)
"%VPY%" -m pip install --disable-pip-version-check pyinstaller
cd /d "%~dp0.."
"%VPY%" -m PyInstaller --noconfirm --distpath dist --workpath build install\surf.spec
if errorlevel 1 (echo Build failed. & pause & exit /b 1)
echo.
echo Built: %CD%\dist\SURF\SURF.exe
pause
