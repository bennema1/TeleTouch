#!/usr/bin/env python3
"""
INTEGRATED TELE-TOUCH DEMO
Combines ML backend with frontend demo and voice/safety integrations

Features:
- ML-based latency compensation (our models)
- Visual demo interface (friends' work)
- Voice announcements (LiveKit integration)
- Safety monitoring (safety systems)
- Screenshot capabilities
- Real-time performance metrics
"""

import sys
import os
import time
import cv2
import numpy as np
import torch
from pathlib import Path

# Import our ML components
from models.surgical_lstm import create_cholecystectomy_lstm
from models.ensemble_model import TrajectoryPredictor

# Import friends' demo components
sys.path.append('TeleTouch_github')
from demo.main import TeleTouchDemo
from demo.predictor import SurgicalPredictor
from demo.overlay_renderer import OverlayRenderer
from demo.position_history import PositionHistory

# Import friends' integration components
from integrations.demo_interface import (
    connect_to_livekit,
    announce,
    check_safety,
    disconnect
)
from integrations.safety_monitor import SafetyMonitor
from integrations.surgical_assistant import SurgicalAssistant


class IntegratedTeleTouchDemo:
    """
    Fully integrated TeleTouch demo combining:
    - ML latency compensation models
    - Visual demo interface
    - Voice integration
    - Safety monitoring
    - Surgical assistant features
    """

    def __init__(self, use_voice=True, use_safety=True):
        print("🚀 Starting Integrated TeleTouch Demo")
        print("=" * 60)

        # Initialize ML components
        self._init_ml_models()

        # Initialize demo components
        self._init_demo_components()

        # Initialize integrations
        self.use_voice = use_voice
        self.use_safety = use_safety
        self._init_integrations()

        print("✅ All components initialized successfully!")

    def _init_ml_models(self):
        """Initialize our ML models for latency compensation."""
        print("🤖 Initializing ML models...")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Try to load trained model, fallback to synthetic
        try:
            self.predictor = TrajectoryPredictor('checkpoints/best_model.pth')
            print("✅ Loaded trained cholecystectomy model")
        except:
            # Create basic model for demo
            model = create_cholecystectomy_lstm('v1')
            model.eval()
            self.predictor = TrajectoryPredictor()
            self.predictor.model = model
            print("⚠️ Using untrained model (for demo purposes)")

        # Initialize friends' predictor wrapper
        self.surgical_predictor = SurgicalPredictor()

    def _init_demo_components(self):
        """Initialize demo visualization components."""
        print("🎮 Initializing demo interface...")

        # Initialize position history
        self.position_history = PositionHistory(max_history=200)

        # Initialize overlay renderer
        self.overlay_renderer = OverlayRenderer()

        # Screen dimensions
        self.screen_width = 1200
        self.screen_height = 800

    def _init_integrations(self):
        """Initialize voice and safety integrations."""
        print("🔗 Initializing integrations...")

        # Voice integration
        if self.use_voice:
            if connect_to_livekit():
                print("✅ Voice integration connected")
                self.voice_enabled = True
                # Initial announcement
                announce("TeleTouch system online. Surgical latency compensation active.")
            else:
                print("⚠️ Voice integration failed")
                self.voice_enabled = False
        else:
            self.voice_enabled = False

        # Safety monitoring
        if self.use_safety:
            self.safety_monitor = SafetyMonitor()
            self.surgical_assistant = SurgicalAssistant()
            print("✅ Safety monitoring enabled")
        else:
            self.safety_monitor = None
            self.surgical_assistant = None

    def run_demo(self):
        """Run the integrated demo."""
        print("\n🎯 INTEGRATED TELE-TOUCH DEMO CONTROLS:")
        print("SPACE: Pause/Resume | R: Restart | Q/ESC: Quit")
        print("S: Screenshot | 1/2/3: Toggle cursors | T: Toggle trails")
        print("I: Toggle info | V: Voice command | H: Help")
        print("=" * 60)

        # Create main demo instance
        demo = TeleTouchDemo(
            predictor=self.surgical_predictor,
            overlay_renderer=self.overlay_renderer,
            position_history=self.position_history,
            safety_monitor=self.safety_monitor,
            surgical_assistant=self.surgical_assistant,
            voice_enabled=self.voice_enabled
        )

        # Add our ML predictor integration
        demo.ml_predictor = self.predictor

        # Add integration callbacks
        demo.on_prediction = self._on_prediction
        demo.on_safety_check = self._on_safety_check
        demo.on_voice_command = self._on_voice_command

        try:
            # Run the demo
            demo.run()

        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user")
        except Exception as e:
            print(f"❌ Demo error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup integrations
            if self.voice_enabled:
                announce("TeleTouch system shutting down.")
                disconnect()

    def _on_prediction(self, position, prediction, uncertainty):
        """Callback when a prediction is made."""
        # Update our ML predictor
        if self.predictor:
            pred, conf = self.predictor.predict_single_step(position)
            return pred, conf
        return prediction, 0.5

    def _on_safety_check(self, screenshot_path):
        """Callback for safety monitoring."""
        if self.safety_monitor and screenshot_path:
            is_safe, message = self.safety_monitor.check_screenshot(screenshot_path)
            if not is_safe:
                print(f"🚨 SAFETY ALERT: {message}")
                if self.voice_enabled:
                    announce(f"Safety warning: {message}")
            return is_safe, message
        return True, "No safety check performed"

    def _on_voice_command(self, command):
        """Callback for voice commands."""
        print(f"🎤 Voice command: {command}")

        # Process voice commands
        if "pause" in command.lower():
            return "pause"
        elif "resume" in command.lower() or "start" in command.lower():
            return "resume"
        elif "screenshot" in command.lower():
            return "screenshot"
        elif "help" in command.lower():
            if self.voice_enabled:
                announce("Available voice commands: pause, resume, screenshot, help")
            return "help"

        return None


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Integrated TeleTouch Demo")
    parser.add_argument('--no-voice', action='store_true',
                       help='Disable voice integration')
    parser.add_argument('--no-safety', action='store_true',
                       help='Disable safety monitoring')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model (optional)')

    args = parser.parse_args()

    print("🎯 TELE-TOUCH INTEGRATED DEMO")
    print("Combining ML models with frontend demo and integrations")
    print("=" * 60)

    try:
        demo = IntegratedTeleTouchDemo(
            use_voice=not args.no_voice,
            use_safety=not args.no_safety
        )

        if args.model_path:
            demo.predictor.load_model(args.model_path)
            print(f"✅ Loaded custom model: {args.model_path}")

        demo.run_demo()

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()