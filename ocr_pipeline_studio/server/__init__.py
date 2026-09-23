"""The Flask application factory.

A factory rather than a module-level ``app = Flask(__name__)`` so the app is
built only once app.py knows where the project lives on disk, and so a second
app -- a test with a throwaway workdir -- shares no state with the first.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify

from server.jobs import JobStore

# Where the pipeline expects to find Ollama. Not configurable from the UI:
# pointing it at a non-local address is what this app is meant to never do.
OLLAMA_URL = "http://localhost:11434"


def create_app(project_root: Optional[Path] = None) -> Flask:
    """Build the Flask app. ``project_root`` defaults to this package's
    parent, which is the project root when the app is run normally."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent

    app = Flask(__name__)

    # One JobStore per app, on the config rather than in a global, so routes
    # and the background worker share an instance while a second app stays
    # independent.
    app.config["JOB_STORE"] = JobStore()
    app.config["PROJECT_ROOT"] = root
    app.config["UI_DIR"] = root / "ui"
    app.config["WORKDIR"] = root / "workdir"
    app.config["OLLAMA_URL"] = OLLAMA_URL

    # A scanned thesis can be large; Flask's default limit would reject one.
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB

    app.config["WORKDIR"].mkdir(parents=True, exist_ok=True)

    # Imported here, not at module top, so importing server never pulls in the
    # pipeline (and through it PaddleOCR) as a side effect.
    from server.routes import bp
    app.register_blueprint(bp)

    @app.errorhandler(Exception)
    def unhandled(exc):
        """Answer every failure as JSON, the way the UI expects.

        Flask's own 500 page is HTML, which the front end cannot read, so a bug
        would surface as a parse error instead of what went wrong. The
        traceback still goes to the log.
        """
        from werkzeug.exceptions import HTTPException

        if isinstance(exc, HTTPException):
            return jsonify({"error": exc.description}), exc.code
        app.logger.exception(exc)
        return jsonify({"error": "%s: %s" % (type(exc).__name__, exc)}), 500

    _warm_normalizer()

    return app


def _warm_normalizer() -> None:
    """Import the OCR script's text normalizer ahead of time, off-thread.

    The review screen's one-click merge needs it, and the import costs about
    nine seconds because it drags in PaddleOCR -- a delay that would otherwise
    land on the user's first click. A pipeline run takes minutes, so this is
    long finished by the time anyone reaches the review screen. Failures are
    swallowed: the lazy import simply happens later, on demand.
    """
    def warm() -> None:
        try:
            from pipeline.runner import _get_normalizer
            _get_normalizer()
        except Exception:
            pass

    threading.Thread(target=warm, daemon=True, name="warm-normalizer").start()
