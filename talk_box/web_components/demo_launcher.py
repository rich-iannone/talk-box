#!/usr/bin/env python3
"""
Talk Box React Chat Demo Launcher

This script starts both the FastAPI backend server and React frontend development server,
then opens the demo interface to showcase the new React chat component.
"""

import atexit
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Global process tracking for cleanup
processes = []


def cleanup_processes():
    """Clean up all spawned processes on exit."""
    print("\n🧹 Cleaning up processes...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
    print("✅ Cleanup complete")


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print(f"\n⚠️  Received signal {signum}")
    cleanup_processes()
    sys.exit(0)


# Register cleanup handlers
atexit.register(cleanup_processes)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def find_project_root():
    """Find the talk-box project root directory."""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists() and (current / "talk_box").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find talk-box project root")


def check_node_installed():
    """Check if Node.js is installed."""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    print("❌ Node.js not found. Installing via Homebrew...")
    try:
        subprocess.run(["brew", "install", "node"], check=True)
        print("✅ Node.js installed successfully")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Failed to install Node.js. Please install manually:")
        print("   brew install node")
        return False


def setup_react_deps(react_dir):
    """Install React dependencies if needed."""
    if not (react_dir / "node_modules").exists():
        print("📦 Installing React dependencies...")
        try:
            subprocess.run(["npm", "install"], cwd=react_dir, check=True)
            print("✅ React dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install React dependencies")
            return False
    else:
        print("✅ React dependencies already installed")
    return True


def start_fastapi_server(server_dir, venv_python):
    """Start the FastAPI server."""
    print("🔧 Starting FastAPI server...")
    try:
        proc = subprocess.Popen(
            [venv_python, "chat_server.py"],
            cwd=server_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        processes.append(proc)

        # Wait for server to start
        for _ in range(30):  # 30 second timeout
            if proc.poll() is not None:
                print("❌ FastAPI server failed to start")
                return False

            try:
                import requests

                response = requests.get("http://127.0.0.1:8000/health", timeout=1)
                if response.status_code == 200:
                    print("✅ FastAPI server running at http://127.0.0.1:8000")
                    return True
            except Exception:
                pass

            time.sleep(1)

        print("❌ FastAPI server timeout")
        return False

    except Exception as e:
        print(f"❌ Failed to start FastAPI server: {e}")
        return False


def start_react_server(react_dir):
    """Start the React development server."""
    print("⚛️  Starting React development server...")
    try:
        proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=react_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        processes.append(proc)

        # Wait for server to start
        for _ in range(60):  # 60 second timeout for first-time setup
            if proc.poll() is not None:
                print("❌ React server failed to start")
                return False

            try:
                import requests

                response = requests.get("http://localhost:5173", timeout=1)
                if response.status_code == 200:
                    print("✅ React server running at http://localhost:5173")
                    return True
            except Exception:
                pass

            time.sleep(1)

        print("❌ React server timeout")
        return False

    except Exception as e:
        print(f"❌ Failed to start React server: {e}")
        return False


def main():
    """Main demo launcher function."""
    print("🚀 Talk Box React Chat Demo Launcher")
    print("=" * 50)

    try:
        # Find project structure
        project_root = find_project_root()
        server_dir = project_root / "talk_box" / "web_components" / "python_server"
        react_dir = project_root / "talk_box" / "web_components" / "react-chat"
        demo_file = project_root / "talk_box" / "web_components" / "demo.html"
        venv_python = project_root / ".venv" / "bin" / "python3"

        print(f"📁 Project root: {project_root}")

        # Check dependencies
        if not venv_python.exists():
            venv_python = project_root / ".venv" / "bin" / "python"
            if not venv_python.exists():
                print("❌ Virtual environment not found. Please run:")
                print("   python -m venv .venv")
                print("   source .venv/bin/activate")
                print("   pip install -e .")
                return 1

        if not check_node_installed():
            return 1

        if not server_dir.exists():
            print(f"❌ Server directory not found: {server_dir}")
            return 1

        if not react_dir.exists():
            print(f"❌ React directory not found: {react_dir}")
            return 1

        # Setup React dependencies
        if not setup_react_deps(react_dir):
            return 1

        # Start servers
        print("\n🔄 Starting servers...")

        if not start_fastapi_server(server_dir, venv_python):
            return 1

        if not start_react_server(react_dir):
            return 1

        # Open demo
        print("\n🌐 Opening demo interface...")
        time.sleep(2)  # Give servers a moment to fully start

        # Open the React interface
        webbrowser.open("http://localhost:5173")

        # Also open the demo page if it exists
        if demo_file.exists():
            webbrowser.open(f"file://{demo_file}")

        print("\n✅ Demo launched successfully!")
        print("\n🔗 Available interfaces:")
        print("   • React Chat: http://localhost:5173")
        print("   • FastAPI Docs: http://127.0.0.1:8000/docs")
        print("   • Health Check: http://127.0.0.1:8000/health")
        if demo_file.exists():
            print(f"   • Demo Page: file://{demo_file}")

        print("\n⚡ Both servers are running!")
        print("   Press Ctrl+C to stop all servers and exit")

        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
