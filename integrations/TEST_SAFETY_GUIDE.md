# Testing Safety Check with Demo Screenshots

## Quick Test (Action 4)

Test the safety check with different error values to verify it works correctly.

### Test 1: Test with Specific Values (10px and 50px)

```powershell
C:\venv_teletouch\Scripts\Activate.ps1
cd C:\python_project\TeleTouch
python integrations/test_safety_from_demo.py --test-values
```

**Expected Results:**
- 10 pixels → `SAFE` ✓
- 50 pixels → `UNSAFE` ✓

---

### Test 2: Test with Actual Screenshot

1. **Run Person C's demo** (if available)
2. **Take a screenshot** of the demo window
   - Save it as `demo_screenshot.png` in the project root
   - Or use Windows Snipping Tool / Print Screen
3. **Calculate the error** between white and green dots
   - Look at the demo's info panel (it shows error in pixels)
   - Or manually measure the distance
4. **Run the test:**
   ```powershell
   python integrations/test_safety_from_demo.py --error 15.5 --screenshot demo_screenshot.png
   ```

---

### Test 3: Test with LiveKit Integration

To test the full warning system (sends to LiveKit):

```powershell
# Terminal 1: Start the agent
python integrations/surgical_assistant.py connect --room surgery-demo

# Terminal 2: Test safety with LiveKit
python integrations/test_safety_from_demo.py --error 25.0 --livekit
```

**What happens:**
- Safety check runs
- If UNSAFE, warning is sent to LiveKit
- Agent announces the warning via voice

---

## How to Get Error Value from Demo

### Option 1: Use Demo's Display
The demo shows error in the info panel:
- Look for "Prediction Error: X.X px"
- Use that value

### Option 2: Calculate from Positions
If you know the dot positions:

```python
from integrations.test_safety_with_screenshot import calculate_error_from_positions

white_pos = (100, 100)  # Actual position (x, y)
green_pos = (110, 110)  # Predicted position (x, y)

error = calculate_error_from_positions(white_pos, green_pos)
print(f"Error: {error:.2f} pixels")
```

### Option 3: Use Demo's calculate_error Method
If you have access to the demo code:

```python
from demo.overlay_renderer import OverlayRenderer

renderer = OverlayRenderer(width=800, height=600)
error = renderer.calculate_error(white_pos, green_pos)
```

---

## Test Cases

### Safe Errors (≤ 15 pixels)
```powershell
python integrations/test_safety_from_demo.py --error 5.0
python integrations/test_safety_from_demo.py --error 10.0
python integrations/test_safety_from_demo.py --error 15.0
```
**Expected:** All should return `SAFE`

### Unsafe Errors (> 15 pixels)
```powershell
python integrations/test_safety_from_demo.py --error 20.0
python integrations/test_safety_from_demo.py --error 30.0
python integrations/test_safety_from_demo.py --error 50.0
```
**Expected:** All should return `UNSAFE`

---

## Integration Example

Here's how Person C would integrate this into their demo:

```python
import asyncio
from integrations.safety_monitor import check_safety_with_warning_async
from livekit.rtc import Room

async def check_prediction_safety(renderer, white_pos, green_pos, room: Room):
    """Check if current prediction error is safe."""
    # Calculate error (demo already does this)
    error = renderer.calculate_error(white_pos, green_pos)
    
    # Take screenshot (optional, for Overshoot API)
    screenshot_path = "current_frame.png"
    cv2.imwrite(screenshot_path, current_frame)
    
    # Check safety
    result = await check_safety_with_warning_async(
        image_path=screenshot_path,
        error_pixels=error,
        room=room,
        send_to_livekit=True
    )
    
    # Display warning if UNSAFE
    if result["safety"] == "UNSAFE":
        renderer.draw_warning(
            frame,
            result["message"]  # "UNSAFE: Prediction error X pixels exceeds safety threshold"
        )
        return True  # Unsafe
    else:
        return False  # Safe
```

---

## Troubleshooting

### "Screenshot not found"
- Take a screenshot manually and save as `demo_screenshot.png`
- Or use `--screenshot path/to/your/image.png`

### "OVERSHOOT_API_KEY not set"
- This is OK for testing - it uses fallback rule-based check
- To use Overshoot API, add key to `config.py`

### "ModuleNotFoundError"
- Make sure virtual environment is activated:
  ```powershell
  C:\venv_teletouch\Scripts\Activate.ps1
  ```

---

## Summary

✅ **Tested Values:**
- 10 pixels → SAFE ✓
- 50 pixels → UNSAFE ✓

✅ **Fallback Check Working:**
- Uses rule: ≤ 15px = SAFE, > 15px = UNSAFE

✅ **Ready for Integration:**
- Can work with or without screenshots
- Can send warnings to LiveKit
- Returns results for demo display

**Next:** Get Overshoot API key to use vision AI instead of fallback rules.
