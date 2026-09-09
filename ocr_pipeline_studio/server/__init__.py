"""The Flask application factory.

Why a factory instead of a module-level ``app = Flask(__name__)``:

  * The app object has to be created *after* we know where the project lives
    on disk, and app.py wants to pass those paths in rather than have this
    module guess them.
  * A factory can be called twice with different settings -- once by app.py
    for the real window, once by a test with a throwaway workdir -- without
    either run leaking state into the other.
  * A module-level app is created at import time, which means an import in the
    wrong order can start doing real work before app.py is ready for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from flask import Flask

from server.jobs import JobStore

# Where the pipeline expects to find Ollama. Not configurable from the UI on
# purpose: pointing this at a non-local address is exactly the thing this app
# promises never to do.
OLLAMA_URL = "http://localhost:11434"


def create_app(project_root: Optional[Path] = None) -> Flask:
    """Build the Flask app.

    ``project_root`` defaults to the directory this package sits in, which is
    the project root when the app is run normally.
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent

    app = Flask(__name__)

    # One JobStore for the whole process, stashed on the config so routes and
    # the background worker reach the same instance. There is no global here:
    # everything goes through the app object, which is what makes a second
    # app in a test genuinely independent.
    app.config["JOB_STORE"] = JobStore()
    app.config["PROJECT_ROOT"] = root
    app.config["UI_DIR"] = root / "ui"
    app.config["WORKDIR"] = root / "workdir"
    app.config["OLLAMA_URL"] = OLLAMA_URL

    # A whole scanned thesis can be large; the default limit would reject one.
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB

    app.config["WORKDIR"].mkdir(parents=True, exist_ok=True)

    # Imported here, not at module top, so that importing ``server`` never
    # pulls in the pipeline (and through it PaddleOCR) as a side effect.
    from server.routes import bp
    app.register_blueprint(bp)

    return app
