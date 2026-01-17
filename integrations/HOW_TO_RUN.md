# How to Run Your TeleTouch Integration

## ✅ What You've Built

1. **LiveKit Connection** - Connects to real-time communication room
2. **ElevenLabs Voice Synthesis** - Text-to-speech for the surgical assistant
3. **Surgical Assistant Agent** - AI agent that joins room and responds to messages
4. **Narration Script** - Pre-written phrases for different scenarios
5. **Safety Monitor** - Checks if prediction errors are safe/unsafe
6. **Warning System** - Sends warnings to LiveKit and returns to demo

---

## 🚀 Quick Start Guide

### Step 1: Activate Virtual Environment

**Always start with this:**
```powershell
C:\venv_teletouch\Scripts\Activate.ps1
cd C:\python_project\TeleTouch
```

---

### Step 2: Test Individual Components

#### Test 1: LiveKit Connection
```powershell
python integrations/test_livekit.py
```
**Expected:** `[OK] Connected to room: surgery-demo`

#### Test 2: Voice Synthesis
```powershell
python integrations/test_voice.py
```
**Expected:** Audio file opens and plays "Surgical assistance system online"

#### Test 3: Safety Monitor
```powershell
python integrations/safety_monitor.py
```
**Expected:** Shows SAFE/UNSAFE results for test errors

---

### Step 3: Run the Full System

#### Terminal 1: Start the Surgical Assistant Agent

```powershell
C:\venv_teletouch\Scripts\Activate.ps1
cd C:\python_project\TeleTouch
python integrations/surgical_assistant.py connect --room surgery-demo
```

**What you'll see:**
```
Starting LiveKit agent...
[OK] Connected to room: surgery-demo
[OK] Surgical Assistant is online
Waiting for events...
```

**The agent will:**
- Connect to the "surgery-demo" room
- Announce: "Surgical assistance system online. Latency compensation active."
- Listen for messages and respond with voice

#### Terminal 2: Send Test Messages

**Option A: Test individual messages**
```powershell
C:\venv_teletouch\Scripts\Activate.ps1
cd C:\python_project\TeleTouch
python integrations/test_message.py
```

**Option B: Run comprehensive demo**
```powershell
C:\venv_teletouch\Scripts\Activate.ps1
cd C:\python_project\TeleTouch
python integrations/demo_all_features.py
```

**What happens:**
- Sends `prediction_accuracy:94` → Agent says: "Prediction accuracy high. System operating normally."
- Sends `prediction_accuracy:55` → Agent says: "Caution: prediction error increasing. Recommend verification."
- Tests safety warning → Agent says: "Warning: high prediction variance. Manual override suggested."
- Sends error report → Agent announces error details

---

## 📋 Message Formats

Your demo can send these messages to the agent:

### 1. Prediction Accuracy
```
prediction_accuracy:94
```
- ≥ 80: "Prediction accuracy high. System operating normally."
- 60-79: "Prediction accuracy moderate. Monitoring performance."
- < 60: "Caution: prediction error increasing. Recommend verification."

### 2. Safety Warning
```
safety_warning:UNSAFE:error:25.0
```
- Agent announces: "Warning: high prediction variance. Manual override suggested."

### 3. Error Report
```
error_report:12:95
```
- Agent announces: "Current prediction error: 12 pixels. Confidence: 95 percent."

### 4. System Stabilized
```
stabilized
```
- Agent announces: "System stabilized. Prediction accuracy restored."

---

## 🔧 Integration for Person C (Demo)

### Simple Integration Example

```python
from integrations.safety_monitor import check_safety_with_warning_async
from livekit.rtc import Room

async def on_prediction_error(image_path, error_pixels, room: Room):
    # Check safety and send warning if needed
    result = await check_safety_with_warning_async(
        image_path=image_path,
        error_pixels=error_pixels,
        room=room,
        send_to_livekit=True
    )
    
    # Display warning overlay if UNSAFE
    if result["safety"] == "UNSAFE":
        show_warning_overlay(result["message"])
    else:
        hide_warning_overlay()
    
    # Send accuracy message
    accuracy = calculate_accuracy(error_pixels)
    await room.local_participant.publish_data(
        f"prediction_accuracy:{accuracy}".encode("utf-8")
    )
```

---

## 📁 Files You've Created

### Core Files
- `integrations/surgical_assistant.py` - Main voice agent
- `integrations/safety_monitor.py` - Safety checking with warnings
- `integrations/narration_script.json` - Voice phrases

### Test Files
- `integrations/test_livekit.py` - Test LiveKit connection
- `integrations/test_voice.py` - Test ElevenLabs TTS
- `integrations/test_message.py` - Test sending messages
- `integrations/test_safety_warning.py` - Test safety warnings
- `integrations/demo_all_features.py` - Full system demo

### Integration Files
- `integrations/demo_safety_integration.py` - Example for Person C
- `integrations/WARNING_DISPLAY_GUIDE.md` - Warning system docs

### Config
- `config.py` - All API keys and credentials

---

## 🎯 What Works Right Now

✅ **LiveKit Connection** - Tested and working  
✅ **ElevenLabs Voice** - Tested and working  
✅ **Surgical Assistant** - Ready to run  
✅ **Message Handling** - All formats supported  
✅ **Safety Monitor** - Fallback working (needs Overshoot API key)  
✅ **Warning System** - Both LiveKit and demo paths ready  

---

## 🔑 Next Steps

1. **Get Overshoot API Key** from NexHacks booth
   - Add to `config.py`: `OVERSHOOT_API_KEY = "your-key"`

2. **Person C Integration**
   - Use `check_safety_with_warning_async()` in demo
   - Send messages to room using formats above

3. **Test End-to-End**
   - Run agent in one terminal
   - Run demo in another
   - Verify voice announcements work

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'livekit'"
**Solution:** Activate the virtual environment first:
```powershell
C:\venv_teletouch\Scripts\Activate.ps1
```

### Agent doesn't respond to messages
**Check:**
- Agent is running and connected
- Messages are sent to the same room ("surgery-demo")
- Message format is correct

### Voice doesn't play
**Check:**
- `ELEVENLABS_API_KEY` is set in `config.py`
- API key has "Text to Speech" permissions
- You have credits on ElevenLabs account

### Safety monitor always uses fallback
**Solution:** Add `OVERSHOOT_API_KEY` to `config.py` (or use fallback for testing)

---

## 📞 Quick Reference

**Start Agent:**
```powershell
C:\venv_teletouch\Scripts\Activate.ps1
python integrations/surgical_assistant.py connect --room surgery-demo
```

**Send Test Message:**
```powershell
C:\venv_teletouch\Scripts\Activate.ps1
python integrations/test_message.py
```

**Run Full Demo:**
```powershell
C:\venv_teletouch\Scripts\Activate.ps1
python integrations/demo_all_features.py
```

---

**You've built a complete voice-enabled surgical assistant system! 🎉**
