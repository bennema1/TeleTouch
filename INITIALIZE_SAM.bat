@echo off
echo ==========================================
echo   SAM Initialization Tool
echo ==========================================
echo.
echo This will help you initialize SAM by clicking on instruments.
echo.
set /p VIDEO_PATH="Enter path to video file: "

if "%VIDEO_PATH%"=="" (
    echo Error: Video path required
    pause
    exit /b 1
)

if not exist "%VIDEO_PATH%" (
    echo Error: Video file not found: %VIDEO_PATH%
    pause
    exit /b 1
)

echo.
echo Loading SAM model (this may take a minute on first run)...
echo.

python demo\sam_initializer.py --video "%VIDEO_PATH%" --frame 0

pause
