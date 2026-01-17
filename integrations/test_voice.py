"""
Test ElevenLabs voice synthesis.
Generates audio for "Surgical assistance system online" and plays it.
Run: python integrations/test_voice.py
"""
import os
import sys
import tempfile
import platform
import subprocess
from pathlib import Path

# Add project root for config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

try:
    from elevenlabs import ElevenLabs
except ImportError:
    print("ERROR: elevenlabs package not installed.")
    print("Run: pip install elevenlabs")
    print("\nNote: If installation fails due to Windows long path issues,")
    print("you may need to enable long path support in Windows.")
    sys.exit(1)

# Check for API key
api_key = getattr(config, "ELEVENLABS_API_KEY", None) or ""
if not api_key or api_key == "" or api_key.startswith("your-key"):
    print("ERROR: ELEVENLABS_API_KEY not set in config.py")
    print("Please add your ElevenLabs API key to config.py")
    sys.exit(1)

# Create client
client = ElevenLabs(api_key=api_key)

print("Generating audio for: 'Surgical assistance system online'...")

try:
    # Generate audio using the client
    # The convert method returns an Iterator[bytes]
    audio_stream = client.text_to_speech.convert(
        voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice (default)
        text="Surgical assistance system online"
    )
    
    # Collect all audio chunks into a single bytes object
    audio_bytes = b""
    for chunk in audio_stream:
        if chunk:
            audio_bytes += chunk
    
    print("Audio generated successfully!")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    print(f"Audio saved to: {tmp_path}")
    print("Playing audio with system default player...")
    
    # Open with system default player
    if platform.system() == "Windows":
        os.startfile(tmp_path)  # type: ignore
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", tmp_path])
    else:  # Linux
        subprocess.run(["xdg-open", tmp_path])
    
    print("\n[OK] Voice synthesis test successful!")
    print("You should hear: 'Surgical assistance system online'")
    print(f"(Audio file: {tmp_path})")
    
except Exception as e:
    print(f"ERROR: Voice synthesis failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check that ELEVENLABS_API_KEY is correct in config.py")
    print("2. Verify your API key has 'Text to Speech' permissions enabled")
    print("3. Check your internet connection")
    print("4. Verify you have credits/quota available on ElevenLabs")
    sys.exit(1)
