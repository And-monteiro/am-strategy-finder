@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto end
if not exist ".venv\Scripts\python.exe" (
    echo Strategy Finder's local Python environment is missing.
    echo See README.md for installation instructions.
    goto end
)
".venv\Scripts\python.exe" -m strategy_finder.windows_launcher

:end
echo(
pause
