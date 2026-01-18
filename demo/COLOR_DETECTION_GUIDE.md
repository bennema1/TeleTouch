# Color-Based Instrument Detection Guide

This guide explains how to use the color-based surgical instrument detection system when annotations are not available.

## Quick Start

### Step 1: Tune Parameters (30 minutes)

First, tune the detection parameters using the interactive tuner:

```bash
python demo/tune_detector.py --video path/to/video.mp4 --frame 100
```

Or with an image:

```bash
python demo/tune_detector.py --image path/to/frame.jpg
```

**How to use the tuner:**
1. Adjust sliders until instruments appear white in the mask panel
2. The right panel shows detected instruments with circles
3. Press 'S' to save parameters to `detector_params.json`
4. Press 'Q' to quit

**What to tune:**
- **Hue Low/High**: Adjust to capture instrument color (usually green/blue for surgical tools)
- **Sat Low/High**: Lower max to reject colorful tissue
- **Val Low/High**: Raise min to reject dark shadows
- **Min Area**: Filter out small noise (start with 500)
- **Min Elongation**: Filter out round objects (instruments are elongated, start with 3.0)

### Step 2: Run Demo with Color Detection

Once you have tuned parameters, run the demo:

```bash
python demo/main.py --video path/to/video.mp4 --color-detect --detector-params detector_params.json --model checkpoints/surgical_lstm_v1_20260117_220117/best_model.pth
```

**Arguments:**
- `--video`: Path to surgical video
- `--color-detect`: Enable color-based detection
- `--detector-params`: Path to parameters JSON (optional, uses defaults if not provided)
- `--instrument-index`: Which instrument to track (0=first, 1=second, default=0)
- `--model`: Path to trained LSTM model

## How It Works

### Detection Pipeline

1. **HSV Color Filtering**: Converts frame to HSV color space and filters pixels matching instrument color
2. **Mask Cleaning**: Removes noise and fills holes using morphological operations
3. **Contour Detection**: Finds connected regions (instruments)
4. **Filtering**: Removes small or round detections (not instruments)
5. **Tip Extraction**: Finds the tip as the point furthest from the centroid
6. **Temporal Smoothing**: Smooths positions across frames to reduce jitter

### Parameter File Format

The `detector_params.json` file looks like:

```json
{
  "hue": [80, 130],
  "sat": [0, 80],
  "val": [120, 255],
  "min_area": 500,
  "min_elongation": 3.0
}
```

## Troubleshooting

### Instruments Not Detected

1. **Check color range**: Instruments might be a different color than expected
   - Try adjusting Hue range (0-179)
   - Check if instruments are visible in the mask panel

2. **Too much noise**: Lower saturation max or raise value min
   - Reduces colorful tissue and shadows

3. **Instruments too small**: Lower min_area threshold
   - But be careful - too low will detect noise

### Tips Jumping Around

1. **Increase smoothing**: The detector uses temporal smoothing by default
   - Adjust `alpha` in `instrument_detector.py` (default 0.7)
   - Higher = more responsive, lower = smoother

2. **Check detection stability**: Run tuner on multiple frames
   - If detection is inconsistent, parameters need adjustment

### Multiple Instruments

If you have multiple instruments:

1. **Track specific instrument**: Use `--instrument-index` to select which one
   - 0 = first detected instrument
   - 1 = second detected instrument
   - etc.

2. **Filter by position**: Modify `instrument_detector.py` to filter by screen region
   - Add position-based filtering in `_extract_tips()`

## Performance Tips

- **For real-time**: Use smaller video resolution or skip frames
- **For accuracy**: Tune parameters carefully on representative frames
- **For stability**: Increase smoothing factor (lower alpha)

## Next Steps

If color detection isn't accurate enough:

1. **Try SAM (Segment Anything Model)**: More robust but slower
2. **Train YOLOv8**: Best accuracy but requires training data
3. **Fix ROSMA annotations**: Perfect accuracy if you can get them working

See the main guide for details on these alternatives.
