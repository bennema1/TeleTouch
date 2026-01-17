# Warning Display Logic Guide

## Overview

When Overshoot returns "UNSAFE", the system triggers warnings in **two ways**:

1. **LiveKit Room** → Surgical Assistant agent announces warning via voice
2. **Demo Display** → Person C's demo shows warning overlay

---

## How It Works

### Flow Diagram

```
Demo detects unsafe error
    ↓
check_safety_with_warning() called
    ↓
Overshoot API analyzes screenshot
    ↓
Returns "UNSAFE"
    ↓
┌─────────────────┬─────────────────┐
│                 │                 │
│  Send to        │  Return to      │
│  LiveKit        │  Demo           │
│                 │                 │
│  Message:       │  Result dict:  │
│  "safety_       │  {              │
│   warning:      │    "safety":    │
│   UNSAFE:       │     "UNSAFE",   │
│   error:25.0"   │    "message":   │
│                 │     "...",      │
│                 │    "warning_   │
│                 │     sent": true │
│                 │  }              │
│                 │                 │
└─────────────────┴─────────────────┘
         ↓                    ↓
    Agent receives      Demo displays
    and announces       warning overlay
    via TTS
```

---

## For Person C (Demo Integration)

### Option 1: Use the Helper Function (Recommended)

```python
from integrations.safety_monitor import check_safety_with_warning_async
from livekit.rtc import Room

# In your demo code:
async def on_prediction_error(image_path, error_pixels, room: Room):
    result = await check_safety_with_warning_async(
        image_path=image_path,
        error_pixels=error_pixels,
        room=room,  # Your connected LiveKit room
        send_to_livekit=True
    )
    
    # Display warning overlay if UNSAFE
    if result["safety"] == "UNSAFE":
        show_warning_overlay(result["message"])
        # result["message"] contains: "UNSAFE: Prediction error X pixels exceeds safety threshold"
    else:
        hide_warning_overlay()
```

### Option 2: Manual Integration

```python
from integrations.safety_monitor import check_safety

# Check safety
safety = check_safety(screenshot_path, error_pixels)

# Send warning to LiveKit if UNSAFE
if safety == "UNSAFE":
    warning_msg = f"safety_warning:UNSAFE:error:{error_pixels}"
    await room.local_participant.publish_data(warning_msg.encode("utf-8"))
    
    # Display overlay
    show_warning_overlay(f"UNSAFE: {error_pixels} pixels")
```

---

## For Surgical Assistant (Already Implemented)

The agent automatically:
1. Listens for `safety_warning:UNSAFE:error:X` messages
2. Looks up the "warning" phrase from `narration_script.json`
3. Uses ElevenLabs TTS to announce: *"Warning: high prediction variance. Manual override suggested."*
4. Plays audio in the LiveKit room

---

## Message Format

**Safety Warning Message:**
```
safety_warning:UNSAFE:error:25.0
```

**Format:** `safety_warning:{STATUS}:error:{pixels}`

- `STATUS`: "SAFE" or "UNSAFE"
- `pixels`: The error value in pixels

---

## Testing

### Test 1: Safety Check Only
```bash
python integrations/safety_monitor.py
```

### Test 2: Full Warning Flow
1. Start the agent:
   ```bash
   python integrations/surgical_assistant.py connect --room surgery-demo
   ```

2. Send a test warning:
   ```bash
   python integrations/test_safety_warning.py
   ```

**Expected Result:**
- Agent receives warning message
- Agent announces: *"Warning: high prediction variance. Manual override suggested."*
- Demo receives result dict to display overlay

---

## Files Created

- ✅ `integrations/safety_monitor.py` - Safety checking with warning support
- ✅ `integrations/demo_safety_integration.py` - Example integration for Person C
- ✅ `integrations/test_safety_warning.py` - Test script for full flow
- ✅ `integrations/surgical_assistant.py` - Updated to handle safety warnings

---

## Next Steps

1. **Get Overshoot API key** from NexHacks booth
2. **Add to config.py**: `OVERSHOOT_API_KEY = "your-key"`
3. **Person C integrates** `check_safety_with_warning_async()` into demo
4. **Test end-to-end** with real screenshots
