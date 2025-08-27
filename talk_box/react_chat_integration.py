"""
React Chat Interface Integration for Talk Box

This module provides seamless integration between Talk Box ChatBot and the React chat interface.
It adds a new show() mode that automatically starts the required servers and opens the React interface.
"""

import atexit
import signal
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

# Global flag to prevent multiple patching
_REACT_SUPPORT_ADDED = False


class ReactChatServer:
    """Manages the React chat interface servers and lifecycle."""

    def __init__(self, chatbot_config: Dict[str, Any]):
        self.chatbot_config = chatbot_config
        self.fastapi_process: Optional[subprocess.Popen] = None
        self.react_process: Optional[subprocess.Popen] = None
        self.project_root: Optional[Path] = None
        self._setup_cleanup()

    def _setup_cleanup(self):
        """Setup cleanup handlers for graceful shutdown."""
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle signals for cleanup."""
        self.cleanup()

    def cleanup(self):
        """Clean up all spawned processes."""
        for process in [self.fastapi_process, self.react_process]:
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=3)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass

    def _find_project_root(self) -> Path:
        """Find the talk-box project root directory."""
        if self.project_root:
            return self.project_root

        current = Path(__file__).parent
        while current != current.parent:
            if (current / "pyproject.toml").exists() and (current / "talk_box").exists():
                self.project_root = current
                return current
            current = current.parent
        raise FileNotFoundError("Could not find talk-box project root")

    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        # Check Node.js
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Node.js required but not found. Install with: brew install node")
            return False

        # Check if React dependencies are installed
        project_root = self._find_project_root()
        react_dir = project_root / "talk_box" / "web_components" / "react-chat"

        if not (react_dir / "node_modules").exists():
            print("📦 Installing React dependencies...")
            try:
                subprocess.run(["npm", "install"], cwd=react_dir, check=True, capture_output=True)
                print("✅ React dependencies installed")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install React dependencies: {e}")
                return False

        return True

    def _check_server_running(self, url: str) -> bool:
        """Check if a server is already running at the given URL."""
        try:
            import requests

            response = requests.get(url, timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _start_fastapi_server(self) -> bool:
        """Start the FastAPI backend server."""
        # First check if it's already running
        if self._check_server_running("http://127.0.0.1:8000/health"):
            print("✅ FastAPI server already running")
            return True

        project_root = self._find_project_root()
        server_dir = project_root / "talk_box" / "web_components" / "python_server"
        venv_python = project_root / ".venv" / "bin" / "python3"

        if not venv_python.exists():
            venv_python = project_root / ".venv" / "bin" / "python"

        if not venv_python.exists():
            print("❌ Virtual environment not found. Please activate your venv first.")
            return False

        try:
            self.fastapi_process = subprocess.Popen(
                [str(venv_python), "chat_server.py"],
                cwd=server_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for server to start
            for _ in range(15):
                if self.fastapi_process.poll() is not None:
                    return False

                if self._check_server_running("http://127.0.0.1:8000/health"):
                    return True

                time.sleep(1)

            return False

        except Exception:
            return False

    def _start_react_server(self) -> bool:
        """Start the React development server."""
        # First check if it's already running
        if self._check_server_running("http://localhost:5173"):
            print("✅ React server already running")
            return True

        project_root = self._find_project_root()
        react_dir = project_root / "talk_box" / "web_components" / "react-chat"

        try:
            print(f"🔍 Starting React server in: {react_dir}")

            # Start the process with proper output handling
            self.react_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=react_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )

            # Wait for server to start with better monitoring
            startup_timeout = 30  # Increased timeout
            for i in range(startup_timeout):
                # Check if process is still alive
                if self.react_process.poll() is not None:
                    # Process exited unexpectedly
                    stdout, stderr = self.react_process.communicate()
                    print("❌ React server process exited unexpectedly:")
                    if stdout:
                        print(f"STDOUT: {stdout}")
                    if stderr:
                        print(f"STDERR: {stderr}")
                    return False

                # Check if server is responding
                if self._check_server_running("http://localhost:5173"):
                    print("✅ React server started successfully")
                    return True

                # Print progress periodically
                if i % 5 == 0 and i > 0:
                    print(f"⏳ Waiting for React server... ({i}/{startup_timeout})")

                time.sleep(1)

            print("❌ React server failed to start within timeout")

            # Try to get some output for debugging
            if self.react_process and self.react_process.poll() is None:
                print("🔍 Process still running, trying to get output...")
                try:
                    # Non-blocking read attempt
                    import select

                    if hasattr(select, "select"):
                        ready, _, _ = select.select([self.react_process.stdout], [], [], 1)
                        if ready:
                            output = self.react_process.stdout.read()
                            if output:
                                print(f"React output: {output}")
                except Exception as debug_e:
                    print(f"Debug read failed: {debug_e}")

            return False

        except Exception as e:
            print(f"❌ Failed to start React server: {e}")
            return False

    def _send_config_to_server(self) -> bool:
        """Send the bot configuration to the FastAPI server."""
        try:
            import requests

            response = requests.post(
                "http://127.0.0.1:8000/config",
                json=self.chatbot_config,
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"⚠️ Warning: Could not send bot config to server: {e}")
            return False

    def launch(self) -> bool:
        """Launch the React chat interface."""
        print("🚀 Starting React Chat Interface...")

        try:
            # Check dependencies
            if not self._check_dependencies():
                return False

            # Start FastAPI server
            print("🔧 Starting backend server...")
            if not self._start_fastapi_server():
                print("❌ Failed to start backend server")
                return False

            # Start React server
            print("⚛️ Starting React interface...")
            if not self._start_react_server():
                print("❌ Failed to start React server")
                return False

            # Send bot configuration to server
            print("🔧 Configuring bot settings...")
            self._send_config_to_server()

            # Open browser
            print("🌐 Opening React chat interface...")
            time.sleep(1)  # Brief pause
            webbrowser.open("http://localhost:5173")

            print("✅ React chat interface launched!")
            print("   📱 Interface: http://localhost:5173")
            print("   🔧 API: http://127.0.0.1:8000")
            print("   💡 Press Ctrl+C in terminal to stop servers")

            return True

        except KeyboardInterrupt:
            print("\n❌ Launch interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Error during launch: {e}")
            return False


def add_react_chat_support():
    """
    Add React chat support to the ChatBot class.
    This function patches the ChatBot.show() method to support "react" mode.
    """
    global _REACT_SUPPORT_ADDED

    # Prevent multiple patching
    if _REACT_SUPPORT_ADDED:
        return True

    try:
        from talk_box.builder import ChatBot

        # Check if already patched by looking for our enhanced method
        if hasattr(ChatBot.show, "_react_enhanced"):
            _REACT_SUPPORT_ADDED = True
            return True

        # Store the original show method
        original_show = ChatBot.show

        def enhanced_show(self, mode: str = "help") -> None:
            """Enhanced show method with React chat support."""
            if mode == "react":
                # Launch React chat interface
                config = self.get_config_summary() if hasattr(self, "get_config_summary") else {}
                server = ReactChatServer(config)

                try:
                    server.launch()
                except Exception as e:
                    print(f"❌ Failed to launch React chat: {e}")
                    print("� Tip: Make sure React dev server is available")
            else:
                # Fall back to original show method for other modes
                original_show(self, mode)

        # Mark as enhanced to prevent re-patching
        enhanced_show._react_enhanced = True

        # Replace the show method
        ChatBot.show = enhanced_show

        _REACT_SUPPORT_ADDED = True
        print("✅ React chat support added to ChatBot.show() method")
        print("💡 Usage: bot.show('react')")

    except ImportError:
        print("❌ Could not import ChatBot class")
        return False

    return True


# Note: Auto-initialization removed to prevent circular imports
# React support will be added when explicitly requested
