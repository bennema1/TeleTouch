@echo off
REM TELE-TOUCH Demo Launcher for Windows
REM Run this script to start the demo

echo ==========================================
echo    TELE-TOUCH Surgical Prediction Demo
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

REM Check/install dependencies
echo Checking dependencies...
pip show opencv-python >nul 2>&1
if errorlevel 1 (
    echo Installing opencv-python...
    pip install opencv-python
)

pip show numpy >nul 2>&1
if errorlevel 1 (
    echo Installing numpy...
    pip install numpy
)

echo.
echo Starting demo...
echo.
echo Controls:
echo   SPACE  - Pause/Resume
echo   R      - Restart  
echo   Q/ESC  - Quit
echo   S      - Screenshot
echo   1/2/3  - Toggle cursors
echo   T      - Toggle trails
echo   I      - Toggle info panel
echo.

cd demo
python main.py %*

pause