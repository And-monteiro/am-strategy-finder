@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Strategy Finder's local Python environment is missing.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -u -m strategy_finder.long_learning_benchmark status "runs\long-learning-benchmark" --watch
echo(
pause
