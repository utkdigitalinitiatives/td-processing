"""ocr-pipeline-studio -- entry point.

Run it with:  python app.py

What happens here, and why it is arranged this way:

Flask and pywebview both want to own the thread they run on. pywebview's
``start()`` must be on the *main* thread -- that is a hard requirement of the
native GUI toolkits it wraps on every platform. Flask's ``run()`` blocks
forever. Only one of them can have the main thread, so Flask goes onto a
background thread and pywebview keeps the main one.

That background thread is a **daemon** thread. A daemon thread does not keep
the process alive once the main thread finishes, which is exactly what we want
here: when the user closes the window, pywebview's start() returns, the main
thread ends, and the Flask thread is torn down with it. Without daemon=True
the server would keep running with no window attached and the app would look
like it had failed to quit.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

# The project root has to be importable so ``pipeline`` and ``server`` resolve
# no matter what directory the app was launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import webview  # noqa: E402  (import after sys.path is set up)

from server import create_app  # noqa: E402

# 127.0.0.1 -- never 0.0.0.0. Binding to 0.0.0.0 would put an unauthenticated
# file-reading, subprocess-spawning server on every network this machine is
# connected to. The loopback address makes it reachable only from this
# computer, which is the entire security model of this app.
HOST = "127.0.0.1"

WINDOW_TITLE = "OCR Pipeline Studio"


def find_free_port() -> int:
    """Ask the OS for an unused port instead of hardcoding one.

    A fixed port would collide with anything else the user happens to be
    running, and the failure would look like the app is broken. Binding to
    port 0 lets the kernel pick a free one, which we then read back.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def start_flask(app, port: int) -> threading.Thread:
    """Start Flask on a daemon background thread. See the module docstring."""

    def serve() -> None:
        # debug/reloader off: the reloader works by re-executing the process,
        # which would spawn a second pywebview window and a second job store.
        app.run(host=HOST, port=port, debug=False,
                use_reloader=False, threaded=True)

    thread = threading.Thread(target=serve, daemon=True, name="flask-server")
    thread.start()
    return thread


def wait_until_serving(port: int, timeout: float = 15.0) -> bool:
    """Block until the server accepts connections, or give up.

    Without this the window can open and request the page before Flask is
    listening, which shows the user a connection error on a perfectly healthy
    app. Polling a socket is enough -- if it accepts, Flask is up.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((HOST, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def main() -> int:
    app = create_app(PROJECT_ROOT)
    port = find_free_port()

    start_flask(app, port)
    if not wait_until_serving(port):
        print("The local server did not start in time. Nothing was launched.",
              file=sys.stderr)
        return 1

    url = "http://%s:%d/" % (HOST, port)
    print("ocr-pipeline-studio is running at %s" % url)
    print("Close the window to quit.")

    webview.create_window(WINDOW_TITLE, url, width=1400, height=900,
                          min_size=(1000, 640))

    # Blocks on the main thread until the window is closed.
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
