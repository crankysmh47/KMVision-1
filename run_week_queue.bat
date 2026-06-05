@echo off
REM KMVision-1 unified week queue (single log, auto-resume).
REM   run_week_queue.bat           - full pipeline, skip completed stages
REM   run_week_queue.bat --from-run 2   - resume Run 2 onward

cd /d "%~dp0"
python scripts\week_queue.py %*
exit /b %ERRORLEVEL%
