@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Watch Learning Benchmark.ps1" %*
echo(
pause
