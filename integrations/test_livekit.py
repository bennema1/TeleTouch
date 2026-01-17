"""
Minimal LiveKit connection test.
Connects to room 'surgery-demo', prints confirmation, and disconnects.
"""
import asyncio
import sys
from pathlib import Path

# Add project root so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livekit.api import AccessToken, VideoGrants
from livekit.rtc import Room

from config import LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET


async def main():
    room_name = "surgery-demo"
    identity = "test-agent"

    print("Attempting to connect to LiveKit...")

    # 1. Create a room token (permission to join)
    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    # 2. Connect to the room
    room = Room()

    try:
        await room.connect(LIVEKIT_URL, token)
        print("[OK] Connected to room: surgery-demo")
        print("[OK] Connection test successful")

        # Brief moment to ensure connection is stable
        await asyncio.sleep(1)

    except Exception as e:
        print(f"Connection failed: {e}")
        print("\nTroubleshooting: Are credentials correct in config.py? Is the URL wss://...?")
        return

    finally:
        print("Disconnecting...")
        await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
