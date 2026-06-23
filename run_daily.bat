@echo off
REM DUGOUT BRAIN daily update — double-click to run, or point Task Scheduler here.
cd /d "%~dp0"
python run_daily.py >> daily.log 2>&1
echo Done. Check daily.log for details.
