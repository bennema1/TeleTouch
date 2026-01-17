# How to Run Manually (If Batch Files Don't Work)

## Step 1: Open PowerShell Terminal 1

1. Press `Windows Key + X`
2. Select "Windows PowerShell" or "Terminal"
3. Navigate to project:
   ```powershell
   cd C:\python_project\TeleTouch
   ```

## Step 2: Start the Agent (Terminal 1)

Run these commands:
```powershell
C:\venv_teletouch\Scripts\Activate.ps1
python integrations\surgical_assistant.py connect --room surgery-demo
```

**Keep this window open!** You should see:
```
[OK] Connected to room: surgery-demo
[OK] Surgical Assistant is online
Waiting for events...
```

---

## Step 3: Open PowerShell Terminal 2

1. Press `Windows Key + X` again
2. Select "Windows PowerShell" or "Terminal" (opens new window)
3. Navigate to project:
   ```powershell
   cd C:\python_project\TeleTouch
   ```

## Step 4: Start the Demo (Terminal 2)

Run these commands:
```powershell
C:\venv_teletouch\Scripts\Activate.ps1
python demo\main.py
```

**The demo window will open!**

---

## What You'll See

### Terminal 1 (Agent):
- Shows connection status
- Shows voice announcements as they happen
- Example: "Agent speaking: 'Prediction accuracy high...'"

### Terminal 2 (Demo):
- Shows "Voice integration: ENABLED"
- Shows frame count and FPS
- Shows safety check results

### Demo Window:
- Visual demo with cursors
- Warning banner when error is high
- Info panel with stats

---

## Quick Copy-Paste Commands

### Terminal 1 (Agent):
```powershell
cd C:\python_project\TeleTouch
C:\venv_teletouch\Scripts\Activate.ps1
python integrations\surgical_assistant.py connect --room surgery-demo
```

### Terminal 2 (Demo):
```powershell
cd C:\python_project\TeleTouch
C:\venv_teletouch\Scripts\Activate.ps1
python demo\main.py
```

---

## Troubleshooting

### "Activate.ps1 cannot be loaded"
Run this first in PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "ModuleNotFoundError"
Make sure virtual environment is activated (you should see `(venv_teletouch)` in prompt)

### Batch files still don't work
Use the manual PowerShell method above - it always works!

---

**This method will definitely work!** Just open two PowerShell windows and run the commands.
