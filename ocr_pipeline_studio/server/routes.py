"""Every HTTP route the app exposes.

All of it is served to exactly one client: the pywebview window running on
this machine. There is no authentication anywhere in this file, which is only
safe because the server is bound to 127.0.0.1 (see server/__init__.py) and is
therefore unreachable from the network.
"""

from __future__ import annotations

import io
import json
import threading
import traceback
import zipfile
from pathlib import Path

from flask import (Blueprint, current_app, jsonify, request,
                   send_file, send_from_directory)
from werkzeug.utils import secure_filename

from pipeline import runner
from server import jobs as jobstate

bp = Blueprint("main", __name__)


# --------------------------------------------------------------------------
# Static UI
# --------------------------------------------------------------------------

@bp.get("/")
def index():
    """Serve the single page the whole app lives in."""
    return send_from_directory(current_app.config["UI_DIR"], "index.html")


@bp.get("/ui/<path:filename>")
def ui_asset(filename: str):
    """Serve app.js, app.css and the vendored Markdown renderer.

    send_from_directory is used rather than open() because it refuses paths
    that escape the directory, which matters for any route with a
    user-supplied path segment.
    """
    return send_from_directory(current_app.config["UI_DIR"], filename)


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------

@bp.get("/models")
def models():
    """Which VLM models are actually pulled on this machine, plus the modes.

    The dropdown is built from this rather than from a hardcoded name, so it
    always reflects reality. The pipeline's default model is reported too, so
    the UI can preselect it when it happens to be installed.
    """
    listing = runner.list_local_models(current_app.config["OLLAMA_URL"])
    return jsonify({
        "models": listing["models"],
        "source": listing["source"],
        "error": listing["error"],
        "default_model": runner.DEFAULT_VLM_MODEL,
        "modes": [
            {"id": key, "label": value["label"], "help": value["help"]}
            for key, value in runner.VLM_MODES.items()
        ],
        "default_mode": runner.DEFAULT_VLM_MODE,
    })


@bp.get("/ollama")
def ollama():
    """Live Ollama reachability, polled by the UI banner."""
    return jsonify(runner.ollama_status(current_app.config["OLLAMA_URL"]))


# --------------------------------------------------------------------------
# Running a job
# --------------------------------------------------------------------------

@bp.post("/upload")
def upload():
    """Accept dropped PDFs, create a job, and start the pipeline off-thread.

    The response goes back as soon as the files are on disk. The actual work
    happens on a background thread, because a full OCR + VLM run takes
    minutes: doing it inline would hold the HTTP connection open the whole
    time and leave the window looking frozen.
    """
    uploaded = request.files.getlist("files")
    if not uploaded:
        return jsonify({"error": "No files were uploaded."}), 400

    model = request.form.get("model") or runner.DEFAULT_VLM_MODEL
    mode = request.form.get("mode") or runner.DEFAULT_VLM_MODE
    if mode not in runner.VLM_MODES:
        return jsonify({"error": "Unknown VLM mode: %s" % mode}), 400

    # Preflight before anything is written: if Ollama is down or the chosen
    # model is missing, say so now, in plain words, rather than letting the
    # job fail deep inside the pipeline.
    check = runner.preflight(model, current_app.config["OLLAMA_URL"], mode)
    if not check["ok"]:
        return jsonify({"error": check["message"]}), 409

    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    job = store.create(current_app.config["WORKDIR"])

    uploads_dir = job.workdir / runner.UPLOADS_DIRNAME
    uploads_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for storage in uploaded:
        # secure_filename strips directory components and unsafe characters.
        # The browser supplies this name, so it is never trusted as a path.
        name = secure_filename(storage.filename or "")
        if not name.lower().endswith(".pdf"):
            continue
        storage.save(uploads_dir / name)
        saved.append(name)

    if not saved:
        return jsonify({"error": "None of the dropped files were PDFs."}), 400

    store.update(
        job.id,
        files=saved,
        model=model,
        mode=mode,
        file_total=len(saved),
        status=jobstate.STATUS_QUEUED,
        message="Queued.",
    )
    for name in saved:
        store.set_file_state(job.id, name, "queued")

    # daemon=True so a half-finished job can never keep the process alive
    # after the window is closed.
    thread = threading.Thread(
        target=_run_job,
        args=(current_app._get_current_object(), job.id),
        daemon=True,
        name="pipeline-%s" % job.id,
    )
    thread.start()

    return jsonify({"job_id": job.id, "files": saved, "model": model, "mode": mode})


def _run_job(app, job_id: str) -> None:
    """Background worker: drive the pipeline and translate its events into
    job-store updates.

    Runs inside an app context so it can read config the same way a route
    does. Every exception is caught and recorded on the job: an unhandled
    exception on a background thread would otherwise vanish silently and
    leave the UI polling a job that never moves.
    """
    with app.app_context():
        store: jobstate.JobStore = app.config["JOB_STORE"]
        job = store.get(job_id)
        if job is None:
            return

        # Why a document produced no draft, keyed by filename. Filled in from
        # the script's own warnings so the queue can state the real reason
        # instead of guessing at one.
        problems: dict = {}

        def report(**event) -> None:
            """Translate one pipeline event into job state.

            Kept as a single function with a dispatch on ``event`` so all the
            UI-facing wording lives in one place.
            """
            kind = event.get("event")

            if kind == "log":
                store.append_log(job_id, event.get("detail", ""))
                return

            if kind == "stage":
                stage = event.get("stage")
                status = {
                    "dedupe": jobstate.STATUS_DEDUPE,
                    "ocr": jobstate.STATUS_OCR,
                    "collecting": jobstate.STATUS_COLLECTING,
                }.get(stage, jobstate.STATUS_OCR)
                store.update(job_id, status=status, message=event.get("detail", stage))

            elif kind == "dedupe_start":
                store.update(job_id, current_file=event.get("filename"),
                             file_index=event.get("index", 0),
                             message="Checking for repeated pages")
                store.set_file_state(job_id, event["filename"], "dedupe")

            elif kind == "dedupe_done":
                removed = event.get("removed", 0)
                detail = ("%d repeated page(s) removed" % removed) if removed else "no repeats"
                store.set_file_state(job_id, event["filename"], "deduped", detail)
                store.append_log(job_id, "[dedupe] %s: %s" % (event["filename"], detail))

            elif kind == "file_index":
                store.update(job_id, file_index=event.get("index", 0),
                             file_total=event.get("total", 0))

            elif kind == "file_start":
                name = event.get("filename")
                store.update(job_id, current_file=name, page_current=0, page_total=0,
                             message="Reading %s" % name)
                store.set_file_state(job_id, name, "ocr")

            elif kind == "page_total":
                store.update(job_id, page_total=event.get("total", 0))

            elif kind == "page_done":
                # Only the fields this event actually carries are updated, so
                # a missing total cannot blank out the one page_total already
                # established -- and nothing is read off the job unlocked.
                fields = {"page_current": event.get("current", 0)}
                if event.get("total"):
                    fields["page_total"] = event["total"]
                store.update(job_id, **fields)

            elif kind == "file_problem":
                # Recorded against the file the script is currently on, so the
                # queue row can say why this document produced nothing.
                current = store.get(job_id)
                if current is not None and current.current_file:
                    problems[current.current_file] = event.get("reason", "")

            elif kind == "vlm_phase":
                # The VLM phase runs once, after all OCR, across the batch.
                store.update(job_id, status=jobstate.STATUS_OCR,
                             page_current=0, page_total=0,
                             message=event.get("detail", "VLM pass"))

            elif kind == "vlm_doc":
                store.update(job_id, current_file=event.get("filename"),
                             message="%s %s (%d page(s))" % (
                                 event.get("verb", "reviewing").capitalize(),
                                 event.get("filename"), event.get("pages", 0)))

        try:
            result = runner.run_pipeline(
                job.workdir,
                model=job.model,
                mode=job.mode,
                ollama_url=app.config["OLLAMA_URL"],
                report=report,
            )

            documents = result["documents"]
            for doc in documents:
                store.set_file_state(job_id, doc["filename"], "done",
                                     "%d flag(s)" % len(doc["flags"]))

            # A document the OCR script skipped (no abstract heading found)
            # produces no draft, so it never appears in ``documents``. Say so
            # rather than leaving it showing "queued" forever.
            produced = {d["filename"] for d in documents}
            skipped = [name for name in job.files if name not in produced]
            for name in skipped:
                store.set_file_state(job_id, name, "skipped",
                                     problems.get(name, "produced no output - see the log"))

            if documents:
                message = "Finished - %d document(s) ready to review." % len(documents)
            else:
                # Report the reason the script actually gave. Distinct reasons
                # are joined so a mixed batch does not hide one behind another.
                reasons = sorted(set(problems.get(n, "") for n in skipped)) or [""]
                detail = "; ".join(r for r in reasons if r)
                message = "Finished, but nothing was extracted"
                message += (": %s." % detail) if detail else " - see the log."

            store.update(job_id,
                         status=jobstate.STATUS_DONE,
                         documents=documents,
                         message=message,
                         page_current=0, page_total=0,
                         current_file=None)

        except Exception as exc:
            store.append_log(job_id, traceback.format_exc())
            store.update(job_id,
                         status=jobstate.STATUS_ERROR,
                         error=str(exc),
                         message="Failed: %s" % exc)


@bp.get("/status/<job_id>")
def status(job_id: str):
    """Poll one job's progress. The UI calls this on a timer while running."""
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    snapshot = store.snapshot(job_id)
    if snapshot is None:
        return jsonify({"error": "No such job."}), 404
    return jsonify(snapshot)


@bp.get("/jobs")
def job_list():
    """Every job this session has run, newest first, for the batch sidebar."""
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    listing = [
        {
            "job_id": job.id,
            "status": job.status,
            "files": job.files,
            "created_at": job.created_at,
            "documents": len(job.documents),
        }
        for job in store.all()
    ]
    listing.sort(key=lambda item: item["created_at"], reverse=True)
    return jsonify({"jobs": listing})


# --------------------------------------------------------------------------
# Review + export
# --------------------------------------------------------------------------

def _find_document(job, name: str):
    """Look a document up by name on a job, or return None.

    Matching on the name recorded in the job -- rather than building a path
    from the request -- is what keeps this route from being a way to read
    arbitrary files off the disk.
    """
    for doc in job.documents:
        if doc["name"] == name:
            return doc
    return None


@bp.get("/document/<job_id>/<path:name>")
def document(job_id: str, name: str):
    """Full content for one document: text, flags, diffs, VLM recovery blocks.

    Returns the in-memory edited text when the user has changed it in this
    session, so switching between documents in the sidebar does not silently
    discard unsaved work.
    """
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "No such job."}), 404

    doc = _find_document(job, name)
    if doc is None:
        return jsonify({"error": "No such document."}), 404

    md_path = job.workdir / runner.OUTPUT_DIRNAME / doc["md_file"]
    on_disk = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

    return jsonify({
        "name": doc["name"],
        "filename": doc["filename"],
        "text": job.edits.get(name, on_disk),
        "saved_text": on_disk,
        "dirty": name in job.edits and job.edits[name] != on_disk,
        "flags": doc["flags"],
        "diffs": doc["diffs"],
        "recovery": doc["recovery"],
        "page_count": doc["page_count"],
        "duplicates_removed": doc["duplicates_removed"],
        "duplicates": doc["duplicates"],
        "model_used": doc["model_used"],
        "mode_used": doc["mode_used"],
        "processed_at": doc["processed_at"],
    })


@bp.post("/edit/<job_id>/<path:name>")
def edit(job_id: str, name: str):
    """Hold an edited document in memory. Nothing is written to disk here.

    The review screen posts here as the user types (debounced). Keeping edits
    in memory until an explicit Save/Export is what the spec asked for, and it
    means a stray keystroke can never damage the pipeline's own output.
    """
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "No such job."}), 404
    if _find_document(job, name) is None:
        return jsonify({"error": "No such document."}), 404

    payload = request.get_json(silent=True) or {}
    job.edits[name] = payload.get("text", "")
    return jsonify({"ok": True, "dirty": True})


@bp.post("/save/<job_id>")
def save(job_id: str):
    """Write in-memory edits out to the .md files on disk.

    This is the only route that overwrites the pipeline's output, and it only
    ever runs when the user presses Save.
    """
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "No such job."}), 404

    payload = request.get_json(silent=True) or {}
    names = payload.get("names") or list(job.edits.keys())

    written = []
    for name in names:
        doc = _find_document(job, name)
        if doc is None or name not in job.edits:
            continue
        path = job.workdir / runner.OUTPUT_DIRNAME / doc["md_file"]
        path.write_text(job.edits[name], encoding="utf-8")
        written.append(doc["md_file"])

    return jsonify({"ok": True, "written": written})


@bp.post("/export")
def export():
    """Download the reviewed output.

    One document comes back as a single .md; several are zipped server-side
    into one download. Either way the bytes are the *possibly edited* text
    from the review screen, not necessarily what is on disk -- exporting is
    meant to give the user what they are looking at.
    """
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    payload = request.get_json(silent=True) or {}
    job_id = payload.get("job_id", "")
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "No such job."}), 404

    names = payload.get("names") or [d["name"] for d in job.documents]
    selected = [d for d in job.documents if d["name"] in names]
    if not selected:
        return jsonify({"error": "Nothing selected to export."}), 400

    def text_for(doc) -> str:
        """Edited text if there is any, otherwise what the pipeline wrote."""
        if doc["name"] in job.edits:
            return job.edits[doc["name"]]
        path = job.workdir / runner.OUTPUT_DIRNAME / doc["md_file"]
        return path.read_text(encoding="utf-8") if path.exists() else ""

    if len(selected) == 1:
        doc = selected[0]
        buffer = io.BytesIO(text_for(doc).encode("utf-8"))
        return send_file(buffer, mimetype="text/markdown",
                         as_attachment=True, download_name=doc["md_file"])

    # Built in memory rather than as a temp file: these are abstracts, a few
    # KB each, so there is nothing to gain from touching the disk.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for doc in selected:
            archive.writestr(doc["md_file"], text_for(doc))
        manifest = job.workdir / runner.OUTPUT_DIRNAME / "manifest.json"
        if manifest.exists():
            archive.writestr("manifest.json", manifest.read_text(encoding="utf-8"))
    buffer.seek(0)
    return send_file(buffer, mimetype="application/zip",
                     as_attachment=True,
                     download_name="ocr-pipeline-studio-%s.zip" % job.id)
