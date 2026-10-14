"""ocr-pipeline-studio entry point. Run with: python app.py

pywebview's start() must be on the main thread -- a hard requirement of the
native GUI toolkits it wraps -- and Flask's run() blocks forever, so Flask goes
on a background thread. Daemon, so closing the window ends the process instead
of leaving a server running with no window attached.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

# Make the project root importable so pipeline/ and server/ resolve no matter
# what directory the app was launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WINDOW_TITLE = "OCR Pipeline Studio"

# Loopback only -- never 0.0.0.0. This server reads files and spawns
# subprocesses with no authentication, so being unreachable from the network
# is the whole security model of this app.
HOST = "127.0.0.1"


def claim_single_instance() -> bool:
    """True if this is the only copy; False when one is already running.

    Now that the app is a desktop icon, and the window takes a few seconds to
    appear, clicking the icon again is the natural thing to do -- and a second
    copy means a second PaddleOCR. Three copies of one batch on a 16 GB machine
    is how a run dies of MemoryError partway through rather than finishing,
    which is not a failure anyone could be expected to diagnose.

    A named mutex rather than a lock file: Windows releases it however the
    process exits, so a crash cannot leave the app unable to start again.

    Checked before the log is opened below, because a second copy opening that
    file would truncate the log of the copy that is actually running.
    """
    if sys.platform != "win32":
        return True
    import ctypes

    ERROR_ALREADY_EXISTS = 183
    kernel32 = ctypes.windll.kernel32
    # Session-local (no "Global\\" prefix): one copy per signed-in user, which
    # is what matters here. The handle is deliberately never closed -- it is
    # meant to be held for as long as the process lives.
    if not kernel32.CreateMutexW(None, True, "ocr-pipeline-studio-single-instance"):
        return True  # Cannot tell; better to run than to refuse to start.
    if kernel32.GetLastError() != ERROR_ALREADY_EXISTS:
        return True

    # Bring the copy already running to the front, so a second click does
    # something visible rather than appearing to do nothing at all.
    try:
        user32 = ctypes.windll.user32
        existing = user32.FindWindowW(None, WINDOW_TITLE)
        if existing:
            user32.ShowWindow(existing, 9)   # SW_RESTORE, in case minimized
            user32.SetForegroundWindow(existing)
    except Exception:
        pass  # Nothing to raise yet: the first copy is still starting up.
    return False


IS_ONLY_INSTANCE = claim_single_instance()

# Launched from the desktop shortcut the app runs under pythonw.exe, which
# gives the process no stdout or stderr at all: prints vanish and werkzeug's
# logger has nowhere to write. Done here, before anything that logs is
# imported, so a launch that fails leaves something to read instead of
# nothing. Opened "w" -- the last launch is what anyone needs, and the file
# cannot grow without bound -- which is only safe because a second copy has
# already bowed out above without reaching this line.
#
# Kept on this PC beside the venv (see launcher/launch.cmd), not in workdir/:
# the project folder may be a share several PCs run from, and each launch
# would wipe the log of whichever PC started before it.
if IS_ONLY_INSTANCE and (sys.stdout is None or sys.stderr is None):
    _local = os.environ.get("LOCALAPPDATA")
    if _local:
        _log_path = Path(_local) / "OCR Pipeline Studio" / "launch.log"
    else:
        _log_path = PROJECT_ROOT / "workdir" / "launch.log"
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(_log_path, "w", encoding="utf-8",
                                   errors="replace", buffering=1)

import webview  # noqa: E402  (import after sys.path is set up)

from server import create_app  # noqa: E402


def find_free_port() -> int:
    """Ask the OS for an unused port by binding to port 0.

    A hardcoded port would collide with whatever else the user is running, and
    the failure would look like a broken app.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def start_flask(app, port: int) -> threading.Thread:
    """Start Flask on a daemon background thread. See the module docstring."""

    def serve() -> None:
        # Reloader off: it re-executes the process, which would spawn a
        # second window and a second job store.
        app.run(host=HOST, port=port, debug=False,
                use_reloader=False, threaded=True)

    thread = threading.Thread(target=serve, daemon=True, name="flask-server")
    thread.start()
    return thread


def wait_until_serving(port: int, timeout: float = 15.0) -> bool:
    """Block until the server accepts connections, or give up.

    Without this the window can request the page before Flask is listening,
    showing a connection error on a perfectly healthy app.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((HOST, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def confirm_close(app, window) -> bool:
    """Ask before closing if anything is unsaved; False keeps the window open.

    Edits, applied differences and Keep/Remove choices live in memory until
    Save changes, and closing ends the process holding them. The server is
    asked rather than the page: this runs on the window's own thread, where
    waiting on the page's JavaScript could hang. Never blocks closing if the
    check itself fails.
    """
    from server.routes import unsaved_changes

    try:
        pending = unsaved_changes(app.config["JOB_STORE"])
    except Exception:
        return True
    if not pending:
        return True

    # Named so the user knows what they would lose, but capped: a batch can
    # hold dozens of documents.
    names = sorted({item["name"] for item in pending})
    listed = "\n".join("  - " + name for name in names[:8])
    if len(names) > 8:
        listed += "\n  - and %d more" % (len(names) - 8)
    message = ("These documents have changes that have not been saved:\n\n%s\n\n"
               "Close anyway and lose them? Choose Cancel to go back and press "
               "Save changes." % listed)
    return window.create_confirmation_dialog("Unsaved changes", message)


def pick_folder(app, window):
    """Ask for a run folder, starting in this machine's workdir.

    Returns None when cancelled or failed; the route treats both as "nothing
    chosen" rather than an error.
    """
    try:
        chosen = window.create_file_dialog(
            webview.FOLDER_DIALOG, directory=str(app.config["WORKDIR"]))
    except Exception:
        return None
    if not chosen:
        return None
    # Platforms return either a path or a sequence of them.
    return chosen if isinstance(chosen, str) else chosen[0]


def main() -> int:
    # Decided at import, before the log was opened -- see claim_single_instance.
    if not IS_ONLY_INSTANCE:
        return 0

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

    window = webview.create_window(WINDOW_TITLE, url, width=1400, height=900,
                                   min_size=(1000, 640))
    window.events.closing += lambda: confirm_close(app, window)

    # Lets the server ask for a folder when opening a past run, which may have
    # been copied from another machine. Injected here so server/ never imports
    # pywebview, same as the close check.
    app.config["PICK_FOLDER"] = lambda: pick_folder(app, window)

    # Blocks on the main thread until the window is closed.
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
