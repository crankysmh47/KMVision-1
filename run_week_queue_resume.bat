@echo off
REM Resume Run 2+ only (Run 1 already complete).
cd /d "%~dp0"
python scripts\week_queue.py --from-run 2 %*
exit /b %ERRORLEVEL%
