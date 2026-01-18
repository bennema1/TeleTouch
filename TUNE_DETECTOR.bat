@echo off
echo ==========================================
echo   Color Detector Parameter Tuner
echo ==========================================
echo.
echo This will help you tune the color detection parameters.
echo.
echo Usage: TUNE_DETECTOR.bat [video_path] [frame_number]
echo   Example: TUNE_DETECTOR.bat "C:\path\to\video.mp4" 100
echo.
echo If no arguments provided, you'll be prompted.
echo.
pause

if "%1"=="" (
    set /p VIDEO_PATH="Enter path to video file: "
    set /p FRAME_NUM="Enter frame number to use (default 100): "
    if "%FRAME_NUM%"=="" set FRAME_NUM=100
) else (
    set VIDEO_PATH=%1
    if "%2"=="" (
        set FRAME_NUM=100
    ) else (
        set FRAME_NUM=%2
    )
)

echo.
echo Running tuner on frame %FRAME_NUM% of %VIDEO_PATH%
echo.
python demo\tune_detector.py --video "%VIDEO_PATH%" --frame %FRAME_NUM%

pause
