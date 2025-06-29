"""
Launch script for the Multi-Agent Dialogue Simulator Streamlit UI
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit application"""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    app_file = script_dir / "streamlit_app.py"
    
    if not app_file.exists():
        print(f"❌ Error: {app_file} not found!")
        return 1
    
    print("🚀 Starting Multi-Agent Dialogue Simulator...")
    print("📱 The web interface will open in your browser")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Server stopped.")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
