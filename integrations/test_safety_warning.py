"""
Test script to demonstrate safety warning flow:
1. Check safety (simulated)
2. Send warning to LiveKit room
3. Agent receives and announces warning
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livekit.api import AccessToken, VideoGrants
from livekit.rtc import Room
from integrations.safety_monitor import check_safety_with_warning_async

from config import LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET


async def main():
    """Test sending a safety warning to the LiveKit room."""
    room_name = "surgery-demo"
    identity = "safety-tester"

    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    room = Room()
    await room.connect(LIVEKIT_URL, token)
    print(f"Connected to {room_name}")
    
    # Simulate checking safety with an unsafe error
    print("\nSimulating safety check with unsafe error (25 pixels)...")
    result = await check_safety_with_warning_async(
        image_path="dummy_screenshot.png",  # Would be actual screenshot in real demo
        error_pixels=25.0,
        room=room,
        send_to_livekit=True
    )
    
    print(f"\nSafety result: {result['safety']}")
    if result["safety"] == "UNSAFE":
        print(f"Warning message: {result['message']}")
        if result["warning_sent"]:
            print("[OK] Warning sent to LiveKit room")
            print("The Surgical Assistant agent should announce the warning")
        else:
            print("[WARNING] Warning was not sent to LiveKit")
    
    print("\nWaiting 3 seconds for agent to process...")
    await asyncio.sleep(3)
    
    await room.disconnect()
    print("Disconnected")


if __name__ == "__main__":
    print("Testing safety warning flow...")
    print("Make sure the Surgical Assistant is running in another terminal:")
    print("  python integrations/surgical_assistant.py connect --room surgery-demo")
    print()
    asyncio.run(main())
