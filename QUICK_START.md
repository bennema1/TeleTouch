# Quick Start Guide - Run Your Demo with Voice! 🚀

## Step-by-Step Instructions

### Step 1: Start the Surgical Assistant Agent

**Option A: Double-click the batch file**
- Double-click `START_AGENT.bat`
- A window will open showing the agent connecting
- **Keep this window open!**

**Option B: Run manually**
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

---

### Step 2: Start the Demo

**Option A: Double-click the batch file**
- Double-click `START_DEMO.bat`
- The demo window will open

**Option B: Run manually**
```powershell
C:\venv_teletouch\Scripts\Activate.ps1
cd C:\python_project\TeleTouch
python demo/main.py
```

**You should see:**
- Demo window opens showing the visualization
- Console shows: "Voice integration: ENABLED"
- White, red, and green cursors moving

---

## What You'll Experience

### Visual (Demo Window):
- ✅ Surgical prediction visualization
- ✅ White cursor = Actual position
- ✅ Red cursor = Lagged position (500ms behind)
- ✅ Green cursor = AI predicted position
- ✅ Warning banner when error is UNSAFE
- ✅ Info panel with error and confidence

### Audio (From Agent Window):
- ✅ "Prediction accuracy high. System operating normally." (when error is low)
- ✅ "Warning: high prediction variance. Manual override suggested." (when error is high)
- ✅ Safety warnings when UNSAFE detected

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

1. **On Startup:**
   - Demo connects to LiveKit
   - Agent announces: "Surgical assistance system online. Latency compensation active."

2. **During Demo:**
   - **Low error (<10px)**: Agent announces accuracy every 4 seconds
   - **High error (>30px)**: Agent announces warning every 2 seconds
   - **Safety check**: Runs every 5 seconds
   - **UNSAFE detected**: Warning banner + agent announcement

3. **On Quit:**
   - Demo disconnects from LiveKit cleanly

---

## Troubleshooting

### "Voice integration: DISABLED"
- Check that `config.py` has valid LiveKit credentials
- Make sure agent is running first
- Check internet connection

### No voice announcements
- Make sure agent window is open and shows "Surgical Assistant is online"
- Check that `ELEVENLABS_API_KEY` is set in `config.py`
- Wait a few seconds for first announcement

### Demo window doesn't open
- Make sure OpenCV is installed: `pip install opencv-python`
- Check that you're in the virtual environment
- Try running from command line to see error messages

### "ModuleNotFoundError"
- Activate virtual environment: `C:\venv_teletouch\Scripts\Activate.ps1`
- Install missing packages: `pip install opencv-python numpy`

---

## Expected Output

### Agent Window:
```
Starting LiveKit agent...
[OK] Connected to room: surgery-demo
[OK] Surgical Assistant is online
Waiting for events...
Received data: prediction_accuracy:94
Agent speaking: 'Prediction accuracy high. System operating normally.'
Received data: warning
Agent speaking: 'Warning: high prediction variance. Manual override suggested.'
```

### Demo Console:
```
TeleTouchDemo initialized:
  Resolution: 1280x720
  FPS: 30
  Latency: 500ms
  Predictor: Quadratic Extrapolation
  Voice integration: ENABLED

TELE-TOUCH Demo Starting
...
Frame 100 | FPS: 30.0 | Avg Error: 12.5px
```

---

## Summary

✅ **Start Agent** → `START_AGENT.bat` or run manually  
✅ **Start Demo** → `START_DEMO.bat` or run manually  
✅ **Watch & Listen** → See visualization + hear voice announcements  
✅ **Quit** → Press Q or ESC in demo window  

**Everything is ready! Just start both and enjoy!** 🎉
