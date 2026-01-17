# How to Integrate Voice and Safety Monitoring

This guide shows Person C how to add voice announcements and safety monitoring to the demo.

---

## Quick Start

### 1. Import the Integration

Add this at the top of `demo/main.py`:

```python
from integrations.demo_interface import (
    connect_to_livekit,
    announce,
    check_safety,
    disconnect
)
```

### 2. Setup (Run Once at Demo Start)

In `TeleTouchDemo.__init__()` or at the start of `main()`:

```python
# Connect to LiveKit for voice agent
if connect_to_livekit():
    print("Voice integration enabled")
else:
    print("Voice integration disabled (will continue without it)")
```

### 3. During Demo Loop

In `TeleTouchDemo.update()` method, add these calls:

```python
def update(self, frame):
    # ... existing code ...
    
    # Calculate error (you already do this)
    error = self.renderer.calculate_error(self.white_pos, self.green_pos)
    
    # === ADD THIS: Voice and Safety Integration ===
    
    # 1. Announce high/low accuracy (every frame, but throttled internally)
    if error > 30:
        announce("warning")
    elif error < 10:
        # Only announce occasionally for low error
        if self.frame_count % 60 == 0:  # Every 2 seconds at 30fps
            accuracy = max(0, 100 - error)
            announce(f"prediction_accuracy:{int(accuracy)}")
    
    # 2. Safety check (automatically throttled to every 5 seconds)
    safety_result = check_safety(
        frame=frame,  # Pass the current frame
        error_pixels=error
    )
    
    # 3. Display warning if UNSAFE
    if safety_result.get("safety") == "UNSAFE":
        self.renderer.draw_warning(
            frame,
            safety_result.get("message", "UNSAFE: High prediction error")
        )
    
    # ... rest of existing code ...
```

### 4. Shutdown

In `TeleTouchDemo.cleanup()` or at the end of `main()`:

```python
def cleanup(self):
    # ... existing cleanup ...
    disconnect()  # Disconnect from LiveKit
```

---

## Function Reference

### `connect_to_livekit() -> bool`

Connect to LiveKit room. Call once at startup.

**Returns:** `True` if connected, `False` if failed

**Example:**
```python
if connect_to_livekit():
    print("Connected to voice agent")
```

---

### `announce(message: str) -> bool`

Send a message to trigger voice narration.

**Message Formats:**
- `"prediction_accuracy:94"` - Announces accuracy level
- `"warning"` - Triggers warning message
- `"stabilized"` - Announces system stabilized
- `"error_report:12:95"` - Reports error with confidence

**Returns:** `True` if sent, `False` if failed

**Example:**
```python
# When error is high
if error > 30:
    announce("warning")

# When accuracy is good
if error < 10:
    accuracy = int(100 - error)
    announce(f"prediction_accuracy:{accuracy}")
```

---

### `check_safety(screenshot_path=None, error_pixels=0.0, frame=None) -> dict`

Check if prediction error is safe. **Automatically throttled to every 5 seconds.**

**Parameters:**
- `screenshot_path` (optional): Path to save screenshot
- `error_pixels`: Current prediction error in pixels
- `frame` (optional): OpenCV frame (will save automatically)

**Returns:**
```python
{
    "safety": "SAFE" or "UNSAFE",
    "message": "Warning message if UNSAFE",
    "warning_sent": True/False,
    "checked": True/False  # False if throttled
}
```

**Example:**
```python
# In your update loop
error = self.renderer.calculate_error(self.white_pos, self.green_pos)
safety_result = check_safety(frame=frame, error_pixels=error)

if safety_result["safety"] == "UNSAFE":
    # Display warning overlay
    self.renderer.draw_warning(frame, safety_result["message"])
```

---

### `disconnect()`

Disconnect from LiveKit. Call at shutdown.

**Example:**
```python
def cleanup(self):
    disconnect()
```

---

## Complete Integration Example

Here's a complete example showing how to integrate into `demo/main.py`:

```python
# At top of file
from integrations.demo_interface import (
    connect_to_livekit,
    announce,
    check_safety,
    disconnect
)

class TeleTouchDemo:
    def __init__(self, ...):
        # ... existing init code ...
        
        # Connect to LiveKit
        self.voice_enabled = connect_to_livekit()
    
    def update(self, frame):
        # ... existing update code ...
        
        # Calculate error (you already do this)
        error = self.renderer.calculate_error(self.white_pos, self.green_pos)
        self.error_history.append(error)
        self.avg_error = np.mean(self.error_history)
        
        # === VOICE AND SAFETY INTEGRATION ===
        
        # Announce warnings for high error
        if self.avg_error > 30:
            if self.frame_count % 30 == 0:  # Throttle announcements
                announce("warning")
        
        # Announce good accuracy occasionally
        elif self.avg_error < 10:
            if self.frame_count % 120 == 0:  # Every 4 seconds
                accuracy = int(100 - self.avg_error)
                announce(f"prediction_accuracy:{accuracy}")
        
        # Safety check (auto-throttled to every 5 seconds)
        safety_result = check_safety(
            frame=frame,
            error_pixels=self.avg_error
        )
        
        # Display warning if UNSAFE
        if safety_result.get("safety") == "UNSAFE":
            self.renderer.draw_warning(
                frame,
                safety_result.get("message", "UNSAFE: High prediction error")
            )
        
        # ... rest of existing code ...
    
    def cleanup(self):
        # ... existing cleanup ...
        disconnect()
```

---

## Integration Points Summary

### What to Add:

1. **Import** (top of file):
   ```python
   from integrations.demo_interface import connect_to_livekit, announce, check_safety, disconnect
   ```

2. **Connect** (in `__init__` or `main()`):
   ```python
   connect_to_livekit()
   ```

3. **In Update Loop**:
   - Call `announce()` for status updates
   - Call `check_safety()` every frame (it auto-throttles)
   - Display warnings if UNSAFE

4. **Disconnect** (in cleanup):
   ```python
   disconnect()
   ```

---

## Testing

### Test 1: Verify Connection

Run the demo and check console output:
```
[Demo Integration] Connected to LiveKit room: surgery-demo
```

### Test 2: Test Voice Announcements

Add this to your update loop temporarily:
```python
if self.frame_count == 60:  # After 2 seconds
    announce("prediction_accuracy:94")
```

You should hear the agent announce: "Prediction accuracy high. System operating normally."

### Test 3: Test Safety Check

Add this to your update loop:
```python
if self.frame_count == 150:  # After 5 seconds
    result = check_safety(frame=frame, error_pixels=25.0)
    print(f"Safety check: {result}")
```

You should see safety check results in console.

---

## Troubleshooting

### "Not connected to LiveKit"
- Make sure `connect_to_livekit()` was called
- Check that `config.py` has valid LiveKit credentials

### Voice not working
- Make sure Surgical Assistant agent is running:
  ```bash
  python integrations/surgical_assistant.py connect --room surgery-demo
  ```
- Check that `ELEVENLABS_API_KEY` is set in `config.py`

### Safety check not running
- Safety check is throttled to every 5 seconds
- Check console for error messages
- Verify `OVERSHOOT_API_KEY` is set (or it will use fallback)

### Demo crashes on import
- Make sure virtual environment is activated:
  ```powershell
  C:\venv_teletouch\Scripts\Activate.ps1
  ```
- Make sure all dependencies are installed

---

## Optional: Advanced Usage

### Custom Safety Check Interval

If you want to change the 5-second interval:

```python
from integrations.demo_interface import _integration

# Change to check every 3 seconds
_integration._safety_check_interval = 3.0
```

### Manual Safety Check (Bypass Throttle)

```python
from integrations.safety_monitor import check_safety

# Direct call (no throttling)
result = check_safety("screenshot.png", error_pixels=25.0)
```

---

## Summary

✅ **Simple Integration:**
- Import 4 functions
- Call `connect_to_livekit()` at startup
- Call `announce()` and `check_safety()` in update loop
- Call `disconnect()` at shutdown

✅ **Automatic Features:**
- Safety check throttled to every 5 seconds
- Warnings automatically sent to LiveKit
- Voice agent announces warnings

✅ **No Breaking Changes:**
- All functions are optional
- Demo works even if LiveKit connection fails
- Graceful degradation

**Ready to integrate!** 🚀
