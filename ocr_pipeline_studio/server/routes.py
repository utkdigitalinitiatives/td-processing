"""Every HTTP route the app exposes.

All of it is served to exactly one client: the pywebview window running on
this machine. There is no authentication anywhere in this file, which is only
safe because the server is bound to 127.0.0.1 (see server/__init__.py) and is
therefore unreachable from the network.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
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

    try:
        options = _run_options(request.form.get("model"), request.form.get("mode"),
                               request.form.get("fixes_only"),
                               request.form.get("overrides") or "{}")
    except _Refused as exc:
        return jsonify({"error": exc.message}), exc.status

    # secure_filename strips directory components and unsafe characters.
    # The browser supplies these names, so they are never trusted as paths.
    pdfs = []
    for storage in uploaded:
        name = secure_filename(storage.filename or "")
        if name.lower().endswith(".pdf"):
            pdfs.append((name, storage))

    if not pdfs:
        return jsonify({"error": "None of the dropped files were PDFs."}), 400

    # Named before anything is saved, since the folder is named after them.
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    saved = [name for name, _ in pdfs]
    job = store.create(current_app.config["WORKDIR"], saved)

    uploads_dir = job.workdir / runner.UPLOADS_DIRNAME
    uploads_dir.mkdir(parents=True, exist_ok=True)
    for name, storage in pdfs:
        storage.save(uploads_dir / name)

    return _launch_job(job, saved, options)


@bp.post("/rerun/<job_id>")
def rerun(job_id: str):
    """Run some of a finished job's files again, as a new job.

    For the files a run got wrong: a thesis whose abstract was missed or
    misidentified is rerun with its pages set by hand, and one with no
    abstract at all is rerun as page fixes only. The uploads are copied from
    the old job, so nothing has to be dropped in again, and the old job's
    results are left as they were.
    """
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    old = store.get(job_id)
    if old is None:
        return jsonify({"error": "No such job."}), 404
    if old.status not in jobstate.FINISHED:
        return jsonify({"error": "That job is still running."}), 409

    payload = request.get_json(silent=True) or {}
    # Only names the old job actually holds -- never a path from the request.
    files = [name for name in payload.get("files", []) if name in old.files]
    if not files:
        return jsonify({"error": "None of those files belong to that job."}), 400

    try:
        options = _run_options(payload.get("model"), payload.get("mode"),
                               payload.get("fixes_only"), payload.get("overrides") or {})
    except _Refused as exc:
        return jsonify({"error": exc.message}), exc.status

    job = store.create(current_app.config["WORKDIR"], files)
    uploads_dir = job.workdir / runner.UPLOADS_DIRNAME
    uploads_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copy2(old.workdir / runner.UPLOADS_DIRNAME / name, uploads_dir / name)

    return _launch_job(job, files, options)


class _Refused(Exception):
    """A run request that cannot start, with the status to answer it with."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def _run_options(model, mode, fixes_only, overrides) -> dict:
    """Validate the settings shared by /upload and /rerun.

    ``overrides`` may arrive as a JSON string (a form field) or already
    decoded (a JSON body). Raises _Refused for anything that should stop the
    job from starting.
    """
    fixes_only = str(fixes_only).lower() in ("1", "true", "yes")
    model = model or runner.DEFAULT_VLM_MODEL
    mode = mode or runner.DEFAULT_VLM_MODE
    if mode not in runner.VLM_MODES:
        raise _Refused("Unknown VLM mode: %s" % mode)

    if isinstance(overrides, str):
        try:
            overrides = json.loads(overrides)
        except ValueError:
            raise _Refused("Abstract page overrides could not be read.")
    if not isinstance(overrides, dict):
        raise _Refused("Abstract page overrides must be an object.")
    ranges = {}
    if not fixes_only:
        for name, value in overrides.items():
            try:
                parsed = runner.parse_page_range(value)
            except ValueError as exc:
                raise _Refused("%s: %s" % (name, exc))
            if parsed is not None:
                # Keyed the way the saved upload is named, so it matches.
                ranges[secure_filename(name)] = parsed

    # Preflight before anything is written: if Ollama is down or the chosen
    # model is missing, say so now, in plain words, rather than letting the
    # job fail deep inside the pipeline. A page-fixes-only run never calls
    # Ollama, so it is checked as if the VLM were off.
    check = runner.preflight(model, current_app.config["OLLAMA_URL"],
                             "off" if fixes_only else mode)
    if not check["ok"]:
        raise _Refused(check["message"], 409)

    return {"model": model, "mode": mode, "fixes_only": fixes_only, "overrides": ranges}


def _launch_job(job, files: list, options: dict):
    """Record a job's settings and start the pipeline on a background thread."""
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    store.update(
        job.id,
        files=files,
        model=options["model"],
        mode=options["mode"],
        fixes_only=options["fixes_only"],
        overrides={n: r for n, r in options["overrides"].items() if n in files},
        file_total=len(files),
        status=jobstate.STATUS_QUEUED,
        message="Queued.",
    )
    for name in files:
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

    return jsonify({"job_id": job.id, "files": files, "model": options["model"],
                    "mode": options["mode"], "fixes_only": options["fixes_only"]})


def _fixes_detail(removed: int, rotated: int, review: int) -> str:
    """One queue-row phrase for a file's page fixes; empty when there were none."""
    parts = []
    if removed:
        parts.append("%d repeated page(s) removed" % removed)
    if rotated:
        parts.append("%d page(s) rotated" % rotated)
    if review:
        parts.append("%d to check by hand" % review)
    return ", ".join(parts)


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
                    "fixes": jobstate.STATUS_FIXING,
                    "ocr": jobstate.STATUS_OCR,
                    "collecting": jobstate.STATUS_COLLECTING,
                }.get(stage, jobstate.STATUS_OCR)
                store.update(job_id, status=status, message=event.get("detail", stage))

            elif kind == "fix_start":
                store.update(job_id, current_file=event.get("filename"),
                             file_index=event.get("index", 0),
                             message="Checking for repeated and sideways pages")
                store.set_file_state(job_id, event["filename"], "fixing")

            elif kind == "fix_done":
                detail = _fixes_detail(event.get("removed", 0), event.get("rotated", 0),
                                       event.get("review", 0)) or "no page fixes needed"
                store.set_file_state(job_id, event["filename"], "fixed", detail)
                store.append_log(job_id, "[fixes] %s: %s" % (event["filename"], detail))

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
                overrides=job.overrides,
                fixes_only=job.fixes_only,
            )

            documents = result["documents"]
            for doc in documents:
                # The page fixes are repeated here because "done" replaces the
                # dedupe/rotate details the row showed while the job ran.
                parts = [] if doc["fixes_only"] else ["%d flag(s)" % len(doc["flags"])]
                if doc["abstract_pages"]:
                    parts.append("abstract pages %s" % doc["abstract_pages"]["requested"])
                fixes = _fixes_detail(doc["duplicates_removed"], doc["pages_rotated"],
                                      doc["pages_for_review"])
                if fixes:
                    parts.append(fixes)
                if doc["fixes_only"] and not parts:
                    parts.append("no page fixes needed")
                store.set_file_state(job_id, doc["filename"], "done", ", ".join(parts))

            # A document the OCR script skipped (no abstract heading found)
            # produces no draft, so it never appears in ``documents``. Say so
            # rather than leaving it showing "queued" forever.
            produced = {d["filename"] for d in documents}
            skipped = [name for name in job.files if name not in produced]
            for name in skipped:
                store.set_file_state(job_id, name, "skipped",
                                     problems.get(name, "produced no output - see the log"))

            if documents and job.fixes_only:
                message = "Finished - page fixes ready for %d document(s)." % len(documents)
            elif documents:
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


@bp.post("/open-folder")
@bp.post("/open-folder/<job_id>")
def open_folder(job_id: str = None):
    """Open a job's folder -- or all of ``workdir/`` -- in the file manager.

    The folder comes from the job store or the app's config, never from the
    request, so this can only ever open the app's own folders. That is also
    what makes it safe to hand to the OS: ``os.startfile`` on a folder opens
    it, it does not run anything.
    """
    if job_id is None:
        folder = current_app.config["WORKDIR"]
    else:
        store: jobstate.JobStore = current_app.config["JOB_STORE"]
        job = store.get(job_id)
        if job is None:
            return jsonify({"error": "No such job."}), 404
        folder = job.workdir

    if not folder.is_dir():
        return jsonify({"error": "That folder no longer exists: %s" % folder}), 404

    try:
        _reveal(folder)
    except Exception as exc:
        # Still hand back the path, so the user can find it by hand.
        return jsonify({"error": "Could not open the folder (%s). It is at: %s"
                                 % (exc, folder)}), 500
    return jsonify({"ok": True, "folder": str(folder)})


def _reveal(folder: Path) -> None:
    """Open ``folder`` in the platform's file manager."""
    if sys.platform.startswith("win"):
        os.startfile(str(folder))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])


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


def _span_key(page_no, index) -> str:
    """Identity of one diff span within a document."""
    return "%s:%s" % (page_no, index)


def _diffs_with_state(diffs: list, text: str, offsets: dict) -> list:
    """Copy the diff blocks, tagging each span with which side the document
    currently shows.

    Worked out from the live text on every request rather than stored, so the
    buttons stay honest even after the user has edited the textarea by hand or
    applied a span and changed their mind. ``offsets`` supplies each span's
    last known position, which resolves the cases the text alone cannot.

    A span whose text appears more than once also gets ``occurrences`` to pick
    from, ``selected`` (the copy currently changed, once one has been picked)
    and ``suggested`` (the likeliest copy, while none is picked).
    """
    annotated = []
    for page in diffs:
        spans = []
        for index, span in enumerate(page["spans"]):
            hint = offsets.get(_span_key(page["page"], index))
            spans.append(dict(
                span,
                state=runner.span_state(text, span["ocr"], span["vlm"], hint),
                hint=hint,
            ))
        annotated.append({"page": page["page"], "spans": spans})

    flat = [span for page in annotated for span in page["spans"]]
    guesses = runner.suggest_occurrences(
        text, [(s["ocr"], s["vlm"], s["state"]) for s in flat])
    for span, guess in zip(flat, guesses):
        view = runner.span_choice_view(text, span["ocr"], span["vlm"],
                                       span["state"], span.pop("hint"))
        if view is None:
            continue
        span.update(view)
        offsets_listed = [o["offset"] for o in view["occurrences"]]
        span["suggested"] = (offsets_listed.index(guess)
                             if view["selected"] is None and guess in offsets_listed else None)
    return annotated


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

    # A page-fixes-only document has no text at all -- its product is the PDF.
    on_disk = ""
    if doc["md_file"]:
        md_path = job.workdir / runner.OUTPUT_DIRNAME / doc["md_file"]
        on_disk = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    current = job.edits.get(name, on_disk)

    return jsonify({
        "name": doc["name"],
        "filename": doc["filename"],
        "fixes_only": doc["fixes_only"],
        "abstract_pages": doc["abstract_pages"],
        "text": current,
        "saved_text": on_disk,
        "dirty": name in job.edits and job.edits[name] != on_disk,
        "flags": doc["flags"],
        "diffs": _diffs_with_state(doc["diffs"], current,
                                   job.span_offsets.get(name, {})),
        "recovery": doc["recovery"],
        "page_count": doc["page_count"],
        "duplicates_removed": doc["duplicates_removed"],
        "duplicates": doc["duplicates"],
        "pages_rotated": doc["pages_rotated"],
        "pages_for_review": doc["pages_for_review"],
        "rotations": doc["rotations"],
        "model_used": doc["model_used"],
        "mode_used": doc["mode_used"],
        "processed_at": doc["processed_at"],
    })


# Which copy of a PDF a page image may be drawn from. A fixed map, never a
# path taken from the request: each key names one of the job's own stage
# folders, so the route cannot be pointed anywhere else on disk.
_PAGE_IMAGE_SOURCES = {
    "original": runner.UPLOADS_DIRNAME,
    "fixed": runner.FIXED_DIRNAME,
}

# PyMuPDF is not thread-safe, and Flask answers each thumbnail request on its
# own thread. One lock makes the page renders take turns.
_render_lock = threading.Lock()


@bp.get("/page-image/<job_id>/<source>/<int:page_idx>/<path:name>")
def page_image(job_id: str, source: str, page_idx: int, name: str):
    """A PNG of one page, for the before/after views on the Page fixes tab.

    ``page_idx`` is 0-based within the chosen copy. Rendering respects the
    page's /Rotate, which is exactly what makes a rotation fix visible: the
    original shows the page as scanned, the fixed copy as corrected.
    """
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "No such job."}), 404

    doc = _find_document(job, name)
    folder = _PAGE_IMAGE_SOURCES.get(source)
    if doc is None or folder is None:
        return jsonify({"error": "No such page."}), 404

    pdf_path = job.workdir / folder / doc["filename"]
    if not pdf_path.exists():
        return jsonify({"error": "No such page."}), 404

    import fitz  # PyMuPDF

    with _render_lock:
        pdf = fitz.open(pdf_path)
        try:
            if not 0 <= page_idx < pdf.page_count:
                return jsonify({"error": "No such page."}), 404
            # 60 dpi is plenty to judge orientation or spot a repeated page,
            # and keeps a document with dozens of fixes quick to open.
            png = pdf[page_idx].get_pixmap(dpi=60).tobytes("png")
        finally:
            pdf.close()

    response = send_file(io.BytesIO(png), mimetype="image/png")
    # Uploads never change, and the UI adds a version to a fixed-PDF page's
    # URL whenever its rotation is changed, so caching stays correct.
    response.headers["Cache-Control"] = "private, max-age=3600"
    return response


@bp.post("/rotation/<job_id>/<path:name>")
def set_rotation(job_id: str, name: str):
    """Change how one page is turned in the fixed PDF.

    For the rotation step what the OCR/VLM swap is for the text: the script's
    turn can be undone, a page it was unsure about turned, or a page turned
    the other way. Body: ``{original_idx, turn}`` with turn 0, 90, 180 or 270
    degrees on top of the page as scanned.

    Only pages the rotation step reported on can be changed. That is also
    what keeps the route to this job's own files: the page is looked up in
    the document's record, never taken as a path.
    """
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "No such job."}), 404
    doc = _find_document(job, name)
    if doc is None:
        return jsonify({"error": "No such document."}), 404

    payload = request.get_json(silent=True) or {}
    turn = payload.get("turn")
    if turn not in runner.VALID_TURNS or isinstance(turn, bool):
        return jsonify({"error": "turn must be 0, 90, 180 or 270."}), 400
    record = next((r for r in doc["rotations"]
                   if r["original_idx"] == payload.get("original_idx")), None)
    if record is None:
        return jsonify({"error": "That page was not part of the rotation step."}), 404

    fixed_pdf = job.workdir / runner.FIXED_DIRNAME / doc["filename"]
    original_pdf = job.workdir / runner.UPLOADS_DIRNAME / doc["filename"]
    if not fixed_pdf.exists() or not original_pdf.exists():
        return jsonify({"error": "The PDFs for this document are no longer in its folder."}), 404

    # Shares the page-image lock: a thumbnail must not render mid-save.
    with _render_lock:
        runner.set_page_rotation(original_pdf, fixed_pdf, record["original_idx"],
                                 record["fixed_idx"], turn)

    record["current"] = turn
    record["decided"] = True
    # The counts behind the sidebar chips follow what the PDF now says.
    doc["pages_rotated"] = sum(1 for r in doc["rotations"] if r["current"])
    doc["pages_for_review"] = sum(1 for r in doc["rotations"]
                                  if not r["applied"] and not r["decided"])

    return jsonify({
        "rotations": doc["rotations"],
        "pages_rotated": doc["pages_rotated"],
        "pages_for_review": doc["pages_for_review"],
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
    doc = _find_document(job, name)
    if doc is None:
        return jsonify({"error": "No such document."}), 404
    if not doc["md_file"]:
        return jsonify({"error": "This document has no text - it was a page fixes only run."}), 409

    payload = request.get_json(silent=True) or {}
    job.edits[name] = payload.get("text", "")
    return jsonify({"ok": True, "dirty": True})


@bp.post("/apply/<job_id>/<path:name>")
def apply_spans(job_id: str, name: str):
    """Apply one diff span -- or every span still on the OCR side -- in place.

    This is what saves the user from scrolling the editor to find the words a
    diff is talking about. Like /edit, it only changes the in-memory copy;
    nothing reaches disk until Save or Export.

    The request body takes either a single ``{page, index, direction}`` or
    ``{direction, all: true}`` to sweep every span that is not already on the
    requested side. A single span may also carry ``at_offset``: the copy the
    user picked when its text appears more than once.
    """
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "No such job."}), 404

    doc = _find_document(job, name)
    if doc is None:
        return jsonify({"error": "No such document."}), 404

    payload = request.get_json(silent=True) or {}
    direction = payload.get("direction", "vlm")
    if direction not in ("vlm", "ocr"):
        return jsonify({"error": "direction must be 'vlm' or 'ocr'."}), 400

    if not doc["md_file"]:
        return jsonify({"error": "This document has no text - it was a page fixes only run."}), 409

    md_path = job.workdir / runner.OUTPUT_DIRNAME / doc["md_file"]
    text = job.edits.get(name)
    if text is None:
        text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

    # Build the list of spans to act on.
    if payload.get("all"):
        targets = [
            (page["page"], index, span)
            for page in doc["diffs"]
            for index, span in enumerate(page["spans"])
        ]
    else:
        page_no = payload.get("page")
        index = payload.get("index")
        targets = [
            (page["page"], i, span)
            for page in doc["diffs"] if page["page"] == page_no
            for i, span in enumerate(page["spans"]) if i == index
        ]
        if not targets:
            return jsonify({"error": "No such diff span."}), 404

    offsets = job.span_offsets.setdefault(name, {})

    # Only meaningful for one span: it is a position in the text as it stands.
    at_offset = None if payload.get("all") else payload.get("at_offset")
    if at_offset is not None and (isinstance(at_offset, bool) or not isinstance(at_offset, int)):
        return jsonify({"error": "at_offset must be a whole number."}), 400

    counts = {"applied": 0, "unchanged": 0, "not_found": 0,
              "ambiguous": 0, "unplaceable": 0, "moved": 0}
    skipped = []
    for page_no, index, span in targets:
        key = _span_key(page_no, index)
        hint = offsets.get(key)

        # Skip spans already showing the requested side, so a bulk apply does
        # not report them as failures. A picked copy is exempt: picking a
        # different copy of an applied span moves the change, it is not a no-op.
        if at_offset is None and runner.span_state(text, span["ocr"], span["vlm"], hint) == direction:
            counts["unchanged"] += 1
            continue

        if at_offset is not None:
            text, status, new_hint = runner.apply_span_choice(
                text, span["ocr"], span["vlm"], direction, hint, at_offset)
        else:
            text, status, new_hint = runner.apply_span(
                text, span["ocr"], span["vlm"], direction, hint)
            # A span once picked from repeated text keeps its picker through
            # an ordinary swap back and forth.
            if status == "applied" and hint and hint.get("picked_side"):
                new_hint = dict(new_hint, picked_side=hint["picked_side"])
        counts[status] += 1

        if status == "applied":
            # Remember where this span now sits. Positions of the other spans
            # are deliberately left alone: each record carries the words
            # either side of its span, so it can still be found after an edit
            # elsewhere moves it, with no bookkeeping to get wrong.
            offsets[key] = new_hint
        else:
            skipped.append({"page": page_no, "index": index, "status": status,
                            "ocr": span["ocr"], "vlm": span["vlm"]})

    job.edits[name] = text

    return jsonify({
        "ok": True,
        "text": text,
        "counts": counts,
        "skipped": skipped,
        "diffs": _diffs_with_state(doc["diffs"], text, offsets),
    })


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
        if doc is None or not doc["md_file"] or name not in job.edits:
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

    if job.fixes_only:
        return _export_fixed_pdfs(job, selected)

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


def _export_fixed_pdfs(job, selected: list):
    """Download a page-fixes-only job's corrected PDFs.

    The file in ``fixed/`` is the finished product: repeats removed and
    sideways pages turned. Unlike abstracts these can run to tens of MB, so a
    batch is zipped to a file in the job folder rather than in memory, and
    stored uncompressed -- PDF scans are already compressed.
    """
    fixed_dir = job.workdir / runner.FIXED_DIRNAME
    pdfs = [fixed_dir / d["filename"] for d in selected]
    missing = [p.name for p in pdfs if not p.exists()]
    if missing:
        return jsonify({"error": "Fixed PDF missing for: %s" % ", ".join(missing)}), 404

    if len(pdfs) == 1:
        return send_file(pdfs[0], mimetype="application/pdf",
                         as_attachment=True, download_name=pdfs[0].name)

    zip_path = job.workdir / "fixed-pdfs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as archive:
        for pdf in pdfs:
            archive.write(pdf, pdf.name)
        manifest = job.workdir / runner.OUTPUT_DIRNAME / "manifest.json"
        if manifest.exists():
            archive.write(manifest, "manifest.json")
    return send_file(zip_path, mimetype="application/zip", as_attachment=True,
                     download_name="fixed-pdfs-%s.zip" % job.id)
