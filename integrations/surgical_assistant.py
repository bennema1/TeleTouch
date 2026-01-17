"""
Surgical Assistant Voice Agent for TeleTouch.
Joins the LiveKit room, announces itself, and responds to data messages with voice (ElevenLabs TTS).
Run: python integrations/surgical_assistant.py dev
Or to connect directly to surgery-demo: python integrations/surgical_assistant.py connect --room surgery-demo
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root for config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# Load narration script
NARRATION_SCRIPT_PATH = Path(__file__).parent / "narration_script.json"
try:
    with open(NARRATION_SCRIPT_PATH, "r") as f:
        NARRATION = json.load(f)
except FileNotFoundError:
    print(f"WARNING: {NARRATION_SCRIPT_PATH} not found. Using fallback phrases.")
    NARRATION = {
        "intro": "Surgical assistance system online.",
        "high_accuracy": "Prediction accuracy high.",
        "medium_accuracy": "Prediction accuracy moderate.",
        "low_accuracy": "Caution: prediction error increasing.",
        "warning": "Warning: high prediction variance.",
        "stabilized": "System stabilized.",
        "error_report": "Current prediction error: {error} pixels."
    }

# ElevenLabs plugin uses ELEVEN_API_KEY
if getattr(config, "ELEVENLABS_API_KEY", None):
    os.environ["ELEVEN_API_KEY"] = config.ELEVENLABS_API_KEY

from livekit import rtc
from livekit.agents import WorkerOptions
from livekit.agents.cli import run_app
from livekit.agents.job import JobContext


def _parse_message(text: str) -> str | None:
    """Parse messages and return appropriate narration phrase."""
    text = text.strip()
    if not text:
        return None
    
    # Handle prediction_accuracy messages
    if "prediction_accuracy:" in text:
        try:
            val = int(text.split(":")[1].strip())
            if val >= 80:
                return NARRATION.get("high_accuracy", f"Prediction accuracy high, {val} percent")
            elif val >= 60:
                return NARRATION.get("medium_accuracy", f"Prediction accuracy moderate, {val} percent")
            else:
                return NARRATION.get("low_accuracy", f"Prediction accuracy low, {val} percent")
        except (IndexError, ValueError):
            return "Could not read prediction accuracy."
    
    # Handle error_report messages (format: "error_report:error:confidence")
    if "error_report:" in text:
        try:
            parts = text.split(":")
            if len(parts) >= 3:
                error = parts[1].strip()
                confidence = parts[2].strip()
                return NARRATION.get("error_report", "").format(error=error, confidence=confidence)
        except Exception:
            pass
    
    # Handle safety warning messages (format: "safety_warning:UNSAFE:error:25.0")
    if "safety_warning:" in text:
        try:
            parts = text.split(":")
            if len(parts) >= 4 and parts[1] == "UNSAFE":
                error = parts[3].strip()
                return NARRATION.get("warning", f"Warning: Unsafe prediction error detected. Error: {error} pixels. Manual override suggested.")
        except Exception:
            pass
        return NARRATION.get("warning", "Warning: high prediction variance. Manual override suggested.")
    
    # Handle warning messages
    if "warning:" in text.lower():
        return NARRATION.get("warning", "Warning: high prediction variance. Manual override suggested.")
    
    # Handle stabilized messages
    if "stabilized" in text.lower():
        return NARRATION.get("stabilized", "System stabilized. Prediction accuracy restored.")
    
    return None


async def entrypoint(ctx: JobContext) -> None:
    """Runs when the agent is assigned to a room."""
    await ctx.connect()
    room = ctx.room
    room_name = room.name or ctx.job.room.name

    print("[OK] Connected to room: " + room_name)
    print("[OK] Surgical Assistant is online")
    print("Waiting for events...")

    tts = None
    audio_source = None
    try:
        from livekit.plugins.elevenlabs import TTS

        api_key = os.environ.get("ELEVEN_API_KEY") or getattr(config, "ELEVENLABS_API_KEY", None)
        if api_key:
            tts = TTS(api_key=api_key)
            # AudioSource and track for TTS playback
            audio_source = rtc.AudioSource(tts.sample_rate, 1)
            track = rtc.LocalAudioTrack.create_audio_track("surgical-voice", audio_source)
            await ctx.agent.publish_track(track)

            # Announce using narration script
            intro_text = NARRATION.get("intro", "Surgical assistance system online. Latency compensation active.")
            stream = tts.synthesize(intro_text)
            async for ev in stream:
                await audio_source.capture_frame(ev.frame)
            await audio_source.wait_for_playout()
        else:
            print("(No ELEVENLABS_API_KEY: voice disabled, only logging)")
    except Exception as e:
        print(f"(Voice setup failed: {e}. Responses will be logged only.)")
        tts = None
        audio_source = None

    def _speak(text: str) -> None:
        async def _do():
            if tts and audio_source:
                try:
                    stream = tts.synthesize(text)
                    async for ev in stream:
                        await audio_source.capture_frame(ev.frame)
                    await audio_source.wait_for_playout()
                except Exception as e:
                    print(f"TTS error: {e}")
            else:
                print(f"[Would say] {text}")

        asyncio.create_task(_do())

    @room.on("data_received")
    def on_data(data_packet: rtc.DataPacket) -> None:
        try:
            msg = data_packet.data.decode("utf-8")
        except Exception:
            return
        response = _parse_message(msg)
        if response:
            print(f"Message: {msg} -> {response}")
            _speak(response)

    # Keep the job running until the room/process ends
    await asyncio.Future()


if __name__ == "__main__":
    opts = WorkerOptions(
        entrypoint_fnc=entrypoint,
        ws_url=config.LIVEKIT_URL,
        api_key=config.LIVEKIT_API_KEY,
        api_secret=config.LIVEKIT_API_SECRET,
    )
    run_app(opts)
