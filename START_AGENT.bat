@echo off
title Surgical Assistant Agent
cd /d C:\python_project\TeleTouch
call C:\venv_teletouch\Scripts\activate.bat
echo ========================================
echo Starting Surgical Assistant Agent
echo ========================================
echo.
echo Keep this window open!
echo.
python integrations\surgical_assistant.py connect --room surgery-demo
pause
