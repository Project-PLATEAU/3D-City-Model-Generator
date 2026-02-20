#!/usr/bin/env python3
"""
Backend startup script with automatic dependency installation
"""

import sys
import subprocess
import os
from pathlib import Path

def install_requirements():
    """Install Python dependencies if needed"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Check if dependencies are installed
        import flask
        import flask_cors
        print("✅ Dependencies already installed")
        return True
    except ImportError:
        print("📦 Installing Python dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False

def start_server():
    """Start the Flask server"""
    server_file = Path(__file__).parent / "server.py"
    
    if not server_file.exists():
        print("❌ server.py not found")
        return False
    
    try:
        print("🚀 Starting BridgeUI Backend Server...")
        os.execv(sys.executable, [sys.executable, str(server_file)])
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

if __name__ == "__main__":
    print("🔧 BridgeUI Backend Startup")
    
    if install_requirements():
        start_server()
    else:
        print("❌ Backend startup failed")
        sys.exit(1)