@echo off
echo ==========================================
echo   Run Demo with Color Detection
echo ==========================================
echo.

REM Check if detector_params.json exists
if not exist "detector_params.json" (
    echo WARNING: detector_params.json not found!
    echo You should run TUNE_DETECTOR.bat first to create parameters.
    echo.
    echo Do you want to continue with default parameters? (Y/N)
    set /p CONTINUE="> "
    if /i not "%CONTINUE%"=="Y" (
        echo Exiting. Run TUNE_DETECTOR.bat first.
        pause
        exit /b
    )
)

REM Get video path
set /p VIDEO_PATH="Enter path to video file: "

if "%VIDEO_PATH%"=="" (
    echo Error: Video path required
    pause
    exit /b 1
)

REM Check if video exists
if not exist "%VIDEO_PATH%" (
    echo Error: Video file not found: %VIDEO_PATH%
    pause
    exit /b 1
)

echo.
echo Starting demo with color detection...
echo Video: %VIDEO_PATH%
echo Model: checkpoints\surgical_lstm_v1_20260117_220117\best_model.pth
echo.

python demo\main.py --video "%VIDEO_PATH%" --color-detect --detector-params detector_params.json --model checkpoints\surgical_lstm_v1_20260117_220117\best_model.pth

pause
