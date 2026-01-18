@echo off
echo ==========================================
echo   Run Demo with SAM Video Tracking
echo ==========================================
echo.

REM Check if tips file exists
set VIDEO_PATH=C:\Users\T2016\Downloads\X01_Pea_on_a_Peg_01.mp4
set TIPS_FILE=C:\Users\T2016\Downloads\X01_Pea_on_a_Peg_01.sam_tips.txt

if not exist "%TIPS_FILE%" (
    echo Tips file not found: %TIPS_FILE%
    echo.
    echo You need to initialize SAM first:
    echo   1. Run: python demo\sam_initializer.py --video "%VIDEO_PATH%"
    echo   2. Click points along instrument, press 'S', then 'Q'
    echo   3. Then run this script again
    echo.
    pause
    exit /b 1
)

echo Found tips file: %TIPS_FILE%
echo Video: %VIDEO_PATH%
echo.
echo Starting demo with SAM video tracking...
echo This will track instruments across all frames in the video.
echo.

python demo\main.py --video "%VIDEO_PATH%" --sam-detect --sam-tips "%TIPS_FILE%" --model checkpoints\surgical_lstm_v1_20260117_220117\best_model.pth

pause
