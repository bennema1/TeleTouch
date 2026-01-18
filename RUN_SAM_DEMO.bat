@echo off
echo ==========================================
echo   Run Demo with SAM Detection
echo ==========================================
echo.

REM Get video path
set /p VIDEO_PATH="Enter path to video file: "

if "%VIDEO_PATH%"=="" (
    echo Error: Video path required
    pause
    exit /b 1
)

REM Check for tips file
set TIPS_FILE=%VIDEO_PATH:.mp4=.sam_tips.txt
if exist "%TIPS_FILE%" (
    echo Found tips file: %TIPS_FILE%
    set USE_TIPS=--sam-tips "%TIPS_FILE%"
) else (
    echo Tips file not found. SAM will try auto-initialization.
    set USE_TIPS=
)

echo.
echo Starting demo with SAM detection...
echo Video: %VIDEO_PATH%
echo Model: checkpoints\surgical_lstm_v1_20260117_220117\best_model.pth
echo.

python demo\main.py --video "%VIDEO_PATH%" --sam-detect %USE_TIPS% --model checkpoints\surgical_lstm_v1_20260117_220117\best_model.pth

pause
