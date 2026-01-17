"""
Comprehensive demo of all implemented features.
This shows everything working together.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livekit.api import AccessToken, VideoGrants
from livekit.rtc import Room, DataPacketKind
from integrations.safety_monitor import check_safety_with_warning_async

from config import LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET


async def demo_all_features():
    """Demonstrate all implemented features."""
    print("=" * 60)
    print("TeleTouch Integration Demo")
    print("=" * 60)
    print()
    
    # Step 1: Connect to LiveKit
    print("STEP 1: Connecting to LiveKit room 'surgery-demo'...")
    room_name = "surgery-demo"
    identity = "demo-client"
    
    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants(room_join=True, room=room_name, can_publish_data=True))
        .to_jwt()
    )
    
    room = Room()
    await room.connect(LIVEKIT_URL, token)
    print("[OK] Connected to LiveKit room")
    print()
    
    # Step 2: Test prediction accuracy messages
    print("STEP 2: Testing prediction accuracy messages...")
    print("Sending: prediction_accuracy:94")
    await room.local_participant.publish_data(
        "prediction_accuracy:94".encode("utf-8"),
        reliable=True,
        kind=DataPacketKind.KIND_RELIABLE
    )
    print("[OK] Message sent - Agent should announce: 'Prediction accuracy high. System operating normally.'")
    await asyncio.sleep(3)
    print()
    
    print("Sending: prediction_accuracy:55")
    await room.local_participant.publish_data(
        "prediction_accuracy:55".encode("utf-8"),
        reliable=True,
        kind=DataPacketKind.KIND_RELIABLE
    )
    print("[OK] Message sent - Agent should announce: 'Caution: prediction error increasing. Recommend verification.'")
    await asyncio.sleep(3)
    print()
    
    # Step 3: Test safety warning
    print("STEP 3: Testing safety warning system...")
    print("Simulating unsafe prediction error (25 pixels)...")
    result = await check_safety_with_warning_async(
        image_path="dummy_screenshot.png",
        error_pixels=25.0,
        room=room,
        send_to_livekit=True
    )
    
    print(f"Safety result: {result['safety']}")
    if result["safety"] == "UNSAFE":
        print(f"Warning message: {result['message']}")
        if result["warning_sent"]:
            print("[OK] Warning sent to LiveKit - Agent should announce safety warning")
        print("[OK] Demo can display warning overlay using result['message']")
    await asyncio.sleep(3)
    print()
    
    # Step 4: Test error report
    print("STEP 4: Testing error report message...")
    print("Sending: error_report:12:95")
    await room.local_participant.publish_data(
        "error_report:12:95".encode("utf-8"),
        reliable=True,
        kind=DataPacketKind.KIND_RELIABLE
    )
    print("[OK] Message sent - Agent should announce error report")
    await asyncio.sleep(3)
    print()
    
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print()
    print("Summary of what was tested:")
    print("  ✓ LiveKit connection")
    print("  ✓ Prediction accuracy messages (high and low)")
    print("  ✓ Safety warning system (UNSAFE detection)")
    print("  ✓ Error reporting")
    print()
    print("The Surgical Assistant agent should have announced all messages via voice.")
    print("Make sure the agent is running in another terminal:")
    print("  python integrations/surgical_assistant.py connect --room surgery-demo")
    
    await room.disconnect()


if __name__ == "__main__":
    print("Starting comprehensive demo...")
    print("Make sure the Surgical Assistant is running first!")
    print()
    asyncio.run(demo_all_features())
