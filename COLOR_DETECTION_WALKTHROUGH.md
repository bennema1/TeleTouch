# Step-by-Step Guide: Using Color Detection

## Overview
This guide walks you through using color-based detection to track surgical instruments in your video when annotations aren't available.

---

## Step 1: Get Your Video Ready

You need a surgical video file. Common formats: `.mp4`, `.avi`, `.mov`

**Example locations:**
- `C:\Users\T2016\Downloads\surgical_video.mp4`
- Or any path to your video file

---

## Step 2: Tune Detection Parameters (30 minutes)

### Option A: Using the Batch File (Easiest)

1. **Double-click `TUNE_DETECTOR.bat`**
2. When prompted, enter:
   - Path to your video file
   - Frame number (start with 100, or pick a frame where instruments are clearly visible)

### Option B: Using Command Line

Open PowerShell or Command Prompt in the project folder, then run:

```powershell
python demo\tune_detector.py --video "C:\path\to\your\video.mp4" --frame 100
```

**Replace `C:\path\to\your\video.mp4` with your actual video path**

### What You'll See

A window with **3 panels**:
- **Left**: Original video frame
- **Middle**: Color mask (what the detector sees)
- **Right**: Detected instruments with circles

### How to Tune

1. **Look at the middle panel (mask)**:
   - Instruments should appear **white**
   - Background should be **black**
   - If instruments are black → adjust Hue range

2. **Adjust sliders**:
   - **Hue Low/High**: Main color of instruments (usually 80-130 for green/blue tools)
   - **Sat Low/High**: Color saturation (lower max to reject colorful tissue)
   - **Val Low/High**: Brightness (raise min to reject dark shadows)
   - **Min Area**: Filter small noise (start with 500)
   - **Min Elongation**: Filter round objects (instruments are long, start with 3.0)

3. **Check the right panel**:
   - Red circles = detected instrument tips
   - Green boxes = detected instrument bodies
   - You should see 1-2 instruments detected

4. **When it looks good**:
   - Press **'S'** to save parameters
   - This creates `detector_params.json` in the current folder
   - Press **'Q'** to quit

### Tips for Tuning

- **Start with default values** - they work for many videos
- **Adjust one slider at a time** - easier to see what changes
- **Try different frames** - run tuner on multiple frames to ensure consistency
- **If too much noise**: Lower Sat High or raise Val Low
- **If instruments not detected**: Adjust Hue range

---

## Step 3: Run the Demo with Color Detection

### Option A: Using Command Line

```powershell
python demo\main.py --video "C:\path\to\your\video.mp4" --color-detect --detector-params detector_params.json --model checkpoints\surgical_lstm_v1_20260117_220117\best_model.pth
```

**What each flag does:**
- `--video`: Your video file path
- `--color-detect`: Enable color-based detection
- `--detector-params`: Path to your tuned parameters (use `detector_params.json` if you saved it)
- `--model`: Your trained LSTM model

### Option B: Create a Batch File

I'll create `RUN_COLOR_DETECTION.bat` for you - just double-click it!

---

## Step 4: What You Should See

The demo window will show:
- **White cursor**: Detected instrument position (from color detection)
- **Red cursor**: Lagged position (500ms behind)
- **Green cursor**: AI predicted position (using your LSTM model)

**Controls:**
- **SPACE**: Pause/Resume
- **R**: Restart
- **Q or ESC**: Quit
- **1/2/3**: Toggle cursors
- **I**: Toggle info panel

---

## Troubleshooting

### "No module named 'instrument_detector'"
**Fix**: Make sure you're running from the project root folder, not from inside the `demo` folder.

### Instruments not detected
1. Re-run the tuner on a different frame
2. Try adjusting parameters manually
3. Check if instruments are clearly visible in your video

### Tips jumping around
- This is normal - the detector includes smoothing
- If too jittery, you can increase smoothing in `demo/instrument_detector.py` (lower `alpha` value)

### Multiple instruments detected
- Use `--instrument-index 0` for first instrument
- Use `--instrument-index 1` for second instrument
- etc.

---

## Quick Test (Without Video)

If you want to test the system first without a video:

```powershell
python demo\main.py --model checkpoints\surgical_lstm_v1_20260117_220117\best_model.pth
```

This uses synthetic data, so you can see the system working before tuning on real video.

---

## Next Steps

Once color detection is working:
1. **Improve accuracy**: Tune parameters on multiple frames
2. **Try different videos**: Parameters might need adjustment per video
3. **Consider alternatives**: If color detection isn't accurate enough, consider SAM or fixing ROSMA annotations

---

## Need Help?

Check `demo/COLOR_DETECTION_GUIDE.md` for more detailed technical information.
