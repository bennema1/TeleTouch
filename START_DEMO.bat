@echo off
title TeleTouch Demo
cd /d C:\python_project\TeleTouch
call C:\venv_teletouch\Scripts\activate.bat
echo ========================================
echo Starting TeleTouch Demo
echo ========================================
echo.
echo Make sure START_AGENT.bat is running first!
echo.
python demo\main.py
pause
