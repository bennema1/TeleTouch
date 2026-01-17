# How the Demo Works & Why Confidence is Volatile

## What the Demo is Doing

### The Three Cursors

1. **White Cursor (ACTUAL)**
   - Represents the **real instrument position**
   - Comes from synthetic data source (simulates surgical instrument movement)
   - Moves smoothly in patterns (circles, figure-8s, etc.)

2. **Red Cursor (LAGGED 500ms)**
   - Shows what a **lagged robot would see**
   - Simulates network delay: 500ms behind the white cursor
   - This is the problem we're solving!

3. **Green Cursor (PREDICTED)**
   - Shows the **AI's prediction** of where the white cursor will be in 500ms
   - Uses the last 10 positions to predict 15 frames ahead (500ms at 30fps)
   - Currently uses "Quadratic Extrapolation" (simple velocity + acceleration)

### The Prediction Process

```
Every Frame (30 times per second):
1. Get white cursor position (actual)
2. Add to lag buffer (simulates 500ms delay)
3. Get red cursor from lag buffer (what robot sees)
4. Add white position to position history
5. Use last 10 positions to predict green cursor (where white will be in 500ms)
6. Calculate error = distance between white and green
7. Update confidence = 100 - average_error
```

---

## Why Confidence is Volatile

### The Problem

**Confidence = 100 - average_error**

The confidence jumps around because:

1. **Error Changes Every Frame**
   - White cursor moves
   - Green prediction tries to follow
   - When movement changes direction/speed, prediction lags behind
   - Error spikes → confidence drops

2. **Simple Predictor**
   - Currently uses "Quadratic Extrapolation"
   - Just calculates velocity and acceleration from last 2-3 positions
   - **Not a real AI model** - it's a placeholder!
   - Can't handle sudden direction changes well

3. **Averaging Window**
   - Uses last 30 frames (1 second) for average
   - When error spikes, it takes time to average out
   - Causes confidence to "bounce" up and down

### Example Scenario

```
Frame 100: White moving right → Green predicts right → Error: 5px → Confidence: 95%
Frame 101: White suddenly turns left → Green still predicting right → Error: 25px → Confidence: 75%
Frame 102: Green catches up → Error: 10px → Confidence: 90%
Frame 103: White accelerates → Green lags → Error: 20px → Confidence: 80%
```

**Result:** Confidence bounces between 75-95% as the predictor struggles to keep up!

---

## How It Connects to Integrations

### The Integration Flow

```
┌─────────────────────────────────────────────────────────┐
│                    DEMO (Every Frame)                    │
│                                                          │
│  1. Calculate error between white & green dots         │
│  2. Update confidence = 100 - avg_error                 │
│  3. Display in info panel                               │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Error data
                   ▼
┌─────────────────────────────────────────────────────────┐
│              INTEGRATION LAYER (demo_interface)         │
│                                                          │
│  • announce("prediction_accuracy:94")  ← Low error     │
│  • announce("warning")                  ← High error    │
│  • check_safety(frame, error)          ← Every 5 sec   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Messages via LiveKit
                   ▼
┌─────────────────────────────────────────────────────────┐
│         SURGICAL ASSISTANT AGENT (surgical_assistant)   │
│                                                          │
│  • Receives messages from demo                          │
│  • Looks up phrases from narration_script.json          │
│  • Uses ElevenLabs TTS to generate speech               │
│  • Announces via voice in LiveKit room                 │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Audio
                   ▼
              🔊 You hear voice!
```

### Specific Integration Points

#### 1. **Voice Announcements** (Lines 250-260 in `demo/main.py`)

```python
# High error (>30px) → Send warning every 2 seconds
if self.avg_error > 30:
    if self.frame_count % 60 == 0:  # Every 2 seconds
        announce("warning")
        # → Agent says: "Warning: high prediction variance..."

# Low error (<10px) → Send accuracy every 4 seconds  
elif self.avg_error < 10:
    if self.frame_count % 120 == 0:  # Every 4 seconds
        accuracy = int(100 - self.avg_error)
        announce(f"prediction_accuracy:{accuracy}")
        # → Agent says: "Prediction accuracy high, 94 percent"
```

#### 2. **Safety Monitoring** (Lines 262-271)

```python
# Every 5 seconds (auto-throttled)
safety_result = check_safety(
    frame=frame,              # Current demo frame
    error_pixels=self.avg_error  # Current error
)

# If UNSAFE → Display warning banner
if safety_result["safety"] == "UNSAFE":
    self.renderer.draw_warning(frame, safety_result["message"])
    # Also sends warning to LiveKit automatically
    # → Agent announces safety warning
```

#### 3. **Message Flow**

```
Demo calculates error
    ↓
If error > 30px → announce("warning")
    ↓
Message sent to LiveKit room "surgery-demo"
    ↓
Surgical Assistant receives message
    ↓
Looks up "warning" in narration_script.json
    ↓
Gets phrase: "Warning: high prediction variance..."
    ↓
ElevenLabs TTS generates speech
    ↓
Audio plays in LiveKit room
    ↓
You hear the announcement!
```

---

## Why Confidence Volatility is Expected

### Current System (Dummy Predictor)

- ✅ **Simple and fast** - works immediately
- ✅ **Good for demos** - shows the concept
- ❌ **Not accurate** - just velocity extrapolation
- ❌ **Volatile** - struggles with direction changes

### Future System (Real AI Model)

When Person B adds the real LSTM predictor:
- ✅ **Much more accurate** - trained on real data
- ✅ **Handles complex patterns** - understands movement better
- ✅ **More stable confidence** - better predictions = lower error
- ✅ **Less volatility** - smoother confidence curve

---

## The Complete System

```
┌─────────────────────────────────────────────────────────────┐
│                    SYNTHETIC DATA SOURCE                     │
│              (Generates white cursor movement)               │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      LAG BUFFER                              │
│         (Simulates 500ms network delay)                     │
│              → Red cursor (lagged)                           │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    PREDICTOR                                 │
│    (Predicts where white will be in 500ms)                  │
│              → Green cursor (predicted)                       │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  ERROR CALCULATION                           │
│         error = distance(white, green)                       │
│         confidence = 100 - avg_error                        │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              INTEGRATION LAYER                               │
│  • announce() → Voice messages                              │
│  • check_safety() → Safety monitoring                       │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              SURGICAL ASSISTANT AGENT                       │
│  • Receives messages                                        │
│  • Announces via ElevenLabs TTS                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

### Why Confidence is Volatile:
1. **Simple predictor** - just velocity extrapolation, not real AI
2. **Direction changes** - predictor lags when movement changes
3. **Averaging window** - takes time to smooth out spikes
4. **This is expected!** - Real AI model will be much better

### How Integration Works:
1. **Demo calculates error** every frame
2. **Sends messages** to LiveKit when error is high/low
3. **Runs safety checks** every 5 seconds
4. **Agent receives** messages and announces via voice
5. **You hear** the voice feedback in real-time!

### The Flow:
```
Error → Integration → LiveKit → Agent → Voice → You!
```

**Everything is connected and working!** The volatility will improve when the real AI model is added. 🚀
