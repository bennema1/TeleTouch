# How to Run the Demo with Voice Integration

## Quick Start

### Step 1: Start the Surgical Assistant Agent (Terminal 1)

```powershell
C:\venv_teletouch\Scripts\Activate.ps1
cd C:\python_project\TeleTouch
python integrations/surgical_assistant.py connect --room surgery-demo
```

**You should see:**
```
Starting LiveKit agent...
[OK] Connected to room: surgery-demo
[OK] Surgical Assistant is online
Waiting for events...
```

**Keep this terminal open!**

---

### Step 2: Run the Demo (Terminal 2)

```powershell
C:\venv_teletouch\Scripts\Activate.ps1
cd C:\python_project\TeleTouch
python demo/main.py
```

**The demo will:**
- Show the surgical prediction visualization
- Connect to LiveKit automatically
- Send voice announcements when error is high/low
- Run safety checks every 5 seconds
- Display warnings when UNSAFE

---

## What You'll See

### In the Demo Window:
- **White cursor**: Actual instrument position
- **Red cursor**: Lagged position (500ms behind)
- **Green cursor**: AI predicted position
- **Info panel**: Shows prediction error, confidence, etc.
- **Warning banner**: Appears when error is UNSAFE

### In Terminal 1 (Agent):
- Voice announcements when messages are received
- Messages like:
  - "Prediction accuracy high. System operating normally."
  - "Warning: high prediction variance. Manual override suggested."

### In Terminal 2 (Demo):
- Connection status
- Safety check results
- Frame count and FPS

---

## Controls

- **SPACE** - Pause/Resume
- **R** - Restart
- **Q or ESC** - Quit
- **S** - Screenshot
- **1/2/3** - Toggle cursors (white/red/green)
- **T** - Toggle trails
- **I** - Toggle info panel

---

## What Happens Automatically

### Voice Announcements:
- **High error (>30px)**: Announces warning every 2 seconds
- **Low error (<10px)**: Announces accuracy every 4 seconds

### Safety Checks:
- **Every 5 seconds**: Checks if error is SAFE/UNSAFE
- **If UNSAFE**: 
  - Sends warning to LiveKit
  - Agent announces warning
  - Demo displays warning banner

---

## Testing Different Scenarios

### Test 1: Low Error (Should hear accuracy announcements)
- Watch the demo run
- When error is low (<10px), you should hear:
  - "Prediction accuracy high. System operating normally."

### Test 2: High Error (Should hear warnings)
- The demo naturally has varying error
- When error is high (>30px), you should hear:
  - "Warning: high prediction variance. Manual override suggested."

### Test 3: Safety Check (Should see UNSAFE warnings)
- Every 5 seconds, safety check runs
- If error > 15px, you'll see:
  - Warning banner in demo
  - Agent announces safety warning

---

## Troubleshooting

### "Voice integration: DISABLED"
- Check that `config.py` has valid LiveKit credentials
- Make sure you're in the virtual environment
- Check internet connection

### No voice announcements
- Make sure the agent is running in Terminal 1
- Check that agent shows "Surgical Assistant is online"
- Verify `ELEVENLABS_API_KEY` is set in `config.py`

### Demo crashes on import
- Make sure virtual environment is activated
- Check that all dependencies are installed:
  ```powershell
  pip install livekit livekit-agents livekit-plugins-elevenlabs
  ```

### Safety check not working
- Safety check is throttled to every 5 seconds (this is normal)
- Check console for error messages
- Verify `OVERSHOOT_API_KEY` is set (or it uses fallback)

---

## Expected Behavior

✅ **Demo starts** → Connects to LiveKit automatically  
✅ **Low error** → Agent announces accuracy  
✅ **High error** → Agent announces warning  
✅ **Safety check** → Runs every 5 seconds  
✅ **UNSAFE detected** → Warning banner + agent announcement  
✅ **Demo quits** → Disconnects from LiveKit cleanly  

---

## Summary

**You now have:**
- ✅ Voice-enabled surgical assistant
- ✅ Real-time safety monitoring
- ✅ Automatic warnings
- ✅ Visual feedback in demo
- ✅ Audio feedback from agent

**Everything is integrated and ready to demo!** 🚀
