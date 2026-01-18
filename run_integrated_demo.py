#!/usr/bin/env python3
"""
LAUNCHER FOR INTEGRATED TELE-TOUCH DEMO
Combines ML backend + Frontend demo + Voice/Safety integrations

This script properly sets up the environment and launches the integrated demo.
"""

import sys
import os
from pathlib import Path

def setup_environment():
    """Set up the Python path and environment for integrated demo."""

    # Get current directory
    current_dir = Path(__file__).resolve().parent

    # Add current directory to Python path
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    # Add TeleTouch_github to Python path for friends' components
    github_dir = current_dir / "TeleTouch_github"
    if github_dir.exists() and str(github_dir) not in sys.path:
        sys.path.insert(0, str(github_dir))

    # Create necessary directories
    screenshot_dir = current_dir / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)

    checkpoint_dir = current_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print("🔧 Environment setup complete")
    print(f"📁 Current directory: {current_dir}")
    print(f"📁 GitHub components: {github_dir}")

def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = [
        'torch', 'cv2', 'numpy', 'pygame'
    ]

    optional_modules = [
        'livekit', 'asyncio'
    ]

    print("🔍 Checking dependencies...")

    missing_required = []
    missing_optional = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing_required.append(module)
            print(f"❌ {module}")

    for module in optional_modules:
        try:
            __import__(module)
            print(f"✅ {module} (optional)")
        except ImportError:
            missing_optional.append(module)
            print(f"⚠️ {module} (optional - integrations disabled)")

    if missing_required:
        print("
❌ MISSING REQUIRED DEPENDENCIES:"        for module in missing_required:
            print(f"   pip install {module}")
        print("\nRun: pip install -r requirements_integrated.txt"
        return False

    if missing_optional:
        print(f"\n⚠️ Missing optional dependencies: {', '.join(missing_optional)}")
        print("Voice and safety integrations will be disabled.")

    return True

def launch_integrated_demo():
    """Launch the integrated TeleTouch demo."""

    print("🚀 LAUNCHING INTEGRATED TELE-TOUCH DEMO")
    print("=" * 60)
    print("Combining:")
    print("  🤖 ML Models (Latency Compensation)")
    print("  🎮 Demo Interface (Visualization)")
    print("  🔊 Voice Integration (LiveKit)")
    print("  🛡️ Safety Monitoring (Overshoot)")
    print("=" * 60)

    try:
        # Import our integrated demo
        from integrated_demo import IntegratedTeleTouchDemo

        # Launch the demo
        demo = IntegratedTeleTouchDemo()
        demo.run_demo()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all files are in the correct directories")
        print("2. Run: pip install -r requirements_integrated.txt")
        print("3. Check that TeleTouch_github folder exists")

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def launch_fallback_demo():
    """Launch a simplified demo if integration fails."""
    print("🔄 Launching simplified demo...")

    try:
        # Import and run our simple demo
        from simple_demo import LatencyDemo

        demo = LatencyDemo()
        demo.run()

    except Exception as e:
        print(f"❌ Even simplified demo failed: {e}")

        # Last resort - try friends' standalone demo
        print("🔄 Trying friends' standalone demo...")
        try:
            # Change to their directory and run their demo
            github_demo = Path("TeleTouch_github/demo/main.py")
            if github_demo.exists():
                os.chdir("TeleTouch_github/demo")
                os.system("python main.py")
            else:
                print("❌ No demo files found")
        except Exception as e2:
            print(f"❌ All demo options failed: {e2}")

def main():
    """Main launcher function."""
    print("🎯 TELE-TOUCH INTEGRATED LAUNCHER")
    print("Bringing together ML models + Demo interface + Voice/Safety integrations")
    print("=" * 80)

    # Setup environment
    setup_environment()

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot launch demo due to missing dependencies.")
        return 1

    # Try integrated demo first
    try:
        launch_integrated_demo()
        return 0
    except Exception as e:
        print(f"\n⚠️ Integrated demo failed: {e}")
        print("Falling back to simplified demo...")

        # Try fallback demo
        launch_fallback_demo()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)