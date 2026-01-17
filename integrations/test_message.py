"""
Test script: connects to the surgery-demo room and sends a data message
that the Surgical Assistant agent should respond to with voice.
Run the agent first (surgical_assistant.py dev or connect --room surgery-demo),
then run: python integrations/test_message.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livekit.api import AccessToken, VideoGrants
from livekit.rtc import Room

from config import LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET


async def main():
    room_name = "surgery-demo"
    identity = "test-sender"

    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    room = Room()
    await room.connect(LIVEKIT_URL, token)
    print(f"Connected to {room_name}. Sending: prediction_accuracy:94")

    # Agent should respond: "Prediction accuracy high, 94 percent"
    await room.local_participant.publish_data(b"prediction_accuracy:94")
    print("Message sent. (Agent should speak if it is in the room.)")

    await asyncio.sleep(3)
    await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
