"""Every HTTP route the app exposes.

All of it serves exactly one client: the pywebview window on this machine.
There is no authentication in this file, which is only safe because the server
binds to 127.0.0.1 (see app.py) and is unreachable from the network.
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
from pathlib import Path

from flask import (Blueprint, current_app, jsonify, request,
                   send_file, send_from_directory)
from werkzeug.utils import secure_filename

from pipeline import runner
from server import jobs as jobstate

bp = Blueprint("main", __name__)


# ---- Static UI ----

@bp.get("/")
def index():
    """Serve the single page the whole app lives in."""
    return send_from_directory(current_app.config["UI_DIR"], "index.html")


@bp.get("/ui/<path:filename>")
def ui_asset(filename: str):
    """Serve app.js, app.css and the vendored Markdown renderer.

    send_from_directory rather than open(): it refuses paths that escape the
    directory, which matters for any route with a user-supplied path segment.
    """
    return send_from_directory(current_app.config["UI_DIR"], filename)


# ---- Ollama ----

@bp.get("/models")
def models():
    """Which VLM models are pulled on this machine, plus the modes.

    The dropdown is built from this rather than a hardcoded name, so it
    reflects reality. The pipeline's default is reported too, so the UI can
    preselect it when it happens to be installed.
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


# ---- Running a job ----

@bp.post("/upload")
def upload():
    """Accept dropped PDFs, create a job, and start the pipeline off-thread.

    The response returns as soon as the files are on disk. A full OCR + VLM run
    takes minutes, so doing it inline would hold the connection open and leave
    the window looking frozen.
    """
    uploaded = request.files.getlist("files")
    if not uploaded:
        return jsonify({"error": "No files were uploaded."}), 400

    try:
        options = _run_options(request.form.get("model"), request.form.get("mode"),
                               request.form.get("steps"),
                               request.form.get("overrides") or "{}")
    except _Refused as exc:
        return jsonify({"error": exc.message}), exc.status

    # The browser supplies these names, so they are never trusted as paths:
    # secure_filename strips directory components and unsafe characters.
    pdfs = []
    seen = set()
    for storage in uploaded:
        name = secure_filename(storage.filename or "")
        if not name.lower().endswith(".pdf"):
            continue
        # Two files of the same name -- from different folders, or differing
        # only in characters secure_filename strips -- would save over each
        # other, leaving a batch claiming more files than it has.
        if name.lower() in seen:
            continue
        seen.add(name.lower())
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

    For the files a run got wrong: a thesis whose abstract was missed is rerun
    with its pages set by hand, one with no abstract at all as page fixes only.
    The uploads are copied from the old job, so nothing has to be dropped in
    again and the old job's results are left as they were.
    """
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    old = store.get(job_id)
    if old is None:
        return jsonify({"error": "No such job."}), 404
    if old.status not in jobstate.FINISHED:
        return jsonify({"error": "That job is still running."}), 409

    payload = request.get_json(silent=True) or {}
    # Only names the old job holds -- never a path from the request.
    files = [name for name in payload.get("files", []) if name in old.files]
    if not files:
        return jsonify({"error": "None of those files belong to that job."}), 400

    try:
        options = _run_options(payload.get("model"), payload.get("mode"),
                               payload.get("steps"), payload.get("overrides") or {})
    except _Refused as exc:
        return jsonify({"error": exc.message}), exc.status

    job = store.create(current_app.config["WORKDIR"], files)
    uploads_dir = job.workdir / runner.UPLOADS_DIRNAME
    uploads_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copy2(old.workdir / runner.UPLOADS_DIRNAME / name, uploads_dir / name)

    return _launch_job(job, files, options)


# ---- Saving a run, listing past runs, opening one ----
# A run's folder holds its own state (server/jobs.py), so a batch run by one
# person can be reviewed by another, later, elsewhere. Every route that changes
# something calls _persist(); typing is coalesced, since a draft edit arrives
# on every pause and the file can be megabytes.

_PERSIST_EVERY = 3.0  # seconds, for the coalesced writes from /edit
_persist_timers: dict = {}
_persist_lock = threading.Lock()


def _persist(job_id: str) -> None:
    """Write this run's state now."""
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    with _persist_lock:
        timer = _persist_timers.pop(job_id, None)
    if timer is not None:
        timer.cancel()
    jobstate.save_job(store, job_id)


def _persist_soon(app, job_id: str) -> None:
    """Write this run's state within a few seconds, once, however many
    changes arrive in the meantime."""
    with _persist_lock:
        if job_id in _persist_timers:
            return

        def run_it():
            with _persist_lock:
                _persist_timers.pop(job_id, None)
            with app.app_context():
                jobstate.save_job(app.config["JOB_STORE"], job_id)

        timer = threading.Timer(_PERSIST_EVERY, run_it)
        timer.daemon = True
        _persist_timers[job_id] = timer
        timer.start()


@bp.get("/saved-runs")
def saved_runs():
    """Past runs found in workdir/, newest first, for the Run screen's list."""
    root: Path = current_app.config["WORKDIR"]
    runs = []
    if root.is_dir():
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            summary = jobstate.read_summary(folder)
            if summary is not None:
                runs.append(summary)
    runs.sort(key=lambda r: r.get("saved_at") or "", reverse=True)
    return jsonify({"runs": runs})


@bp.post("/open-run")
def open_run():
    """Open a run folder for review, from its saved state or from its files.

    With no ``folder`` in the body the window's folder picker is used, so a run
    copied from another machine can be opened from anywhere. A folder counts as
    a run if it holds the saved state or an ``uploads`` folder; anything else
    is refused rather than opened as an empty review.
    """
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    payload = request.get_json(silent=True) or {}
    given = payload.get("folder")

    if not given:
        picker = current_app.config.get("PICK_FOLDER")
        if picker is None:
            return jsonify({"error": "No folder picker here - pass a folder path."}), 501
        given = picker()
        if not given:
            return jsonify({"cancelled": True})

    folder = Path(str(given)).expanduser().resolve()
    if not folder.is_dir():
        return jsonify({"error": "There is no folder at: %s" % folder}), 404

    # Already open: hand back that job rather than a second copy. Two copies
    # would each hold their own choices and overwrite the other's saved state,
    # last writer winning.
    for job in store.all():
        if job.workdir == folder:
            return jsonify({
                "job_id": job.id, "files": job.files, "model": job.model,
                "mode": job.mode, "steps": job.steps, "fixes_only": job.fixes_only,
                "folder": str(folder), "rebuilt": False,
                "documents": len(job.documents), "already_open": True,
            })

    try:
        state = jobstate.load_state(folder)
    except jobstate.StateError as exc:
        return jsonify({"error": str(exc)}), 400

    if state is None:
        # No saved state: recover what the files themselves still hold.
        if not (folder / runner.UPLOADS_DIRNAME).is_dir():
            return jsonify({"error": "That folder does not look like a run: it has no "
                                     "%s and no uploads folder." % jobstate.STATE_FILE}), 400
        documents = runner.rebuild_documents(folder)
        job = store.register(folder, documents=documents)
        store.update(job.id, message="Opened from a folder - rebuilt from its files, "
                                     "so repeated and sideways pages are not recorded.")
    else:
        job = store.register(folder, payload=state)

    store.append_log(job.id, "[open] %s" % folder)
    return jsonify({
        "job_id": job.id,
        "files": job.files,
        "model": job.model,
        "mode": job.mode,
        "steps": job.steps,
        "fixes_only": job.fixes_only,
        "folder": str(folder),
        "rebuilt": state is None,
        "documents": len(job.documents),
    })


class _Refused(Exception):
    """A run request that cannot start, with the status to answer it with."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def _decode(value, what: str):
    """A form field arrives as a JSON string, a JSON body already decoded."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            raise _Refused("%s could not be read." % what)
    return value


def _run_options(model, mode, steps, overrides) -> dict:
    """Validate the settings shared by /upload and /rerun.

    ``steps`` chooses what to run -- ``{"dedupe", "rotate", "ocr"}``, any
    missing one on -- and ``overrides`` the abstract pages set by hand. Either
    may arrive as a JSON string (a form field) or already decoded (a JSON
    body). Raises _Refused for anything that should stop the job starting.
    """
    steps = _decode(steps, "The steps to run") if steps is not None else {}
    if not isinstance(steps, dict):
        raise _Refused("The steps to run must be an object.")
    steps = runner.normalize_steps(steps)
    if not any(steps.values()):
        raise _Refused("Pick at least one step to run.")
    # For everything downstream that only asks "is there text?"
    fixes_only = not steps["ocr"]

    model = model or runner.DEFAULT_VLM_MODEL
    mode = mode or runner.DEFAULT_VLM_MODE
    if mode not in runner.VLM_MODES:
        raise _Refused("Unknown VLM mode: %s" % mode)

    overrides = _decode(overrides, "Abstract page overrides")
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

    # Preflight before anything is written, so a missing model or a stopped
    # Ollama is reported now rather than deep inside the pipeline. A run
    # without OCR never calls Ollama, so it is checked as if the VLM were off.
    check = runner.preflight(model, current_app.config["OLLAMA_URL"],
                             "off" if fixes_only else mode)
    if not check["ok"]:
        raise _Refused(check["message"], 409)

    return {"model": model, "mode": mode, "steps": steps, "fixes_only": fixes_only,
            "overrides": ranges}


def _launch_job(job, files: list, options: dict):
    """Record a job's settings and start the pipeline on a background thread."""
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    store.update(
        job.id,
        files=files,
        model=options["model"],
        mode=options["mode"],
        steps=options["steps"],
        fixes_only=options["fixes_only"],
        overrides={n: r for n, r in options["overrides"].items() if n in files},
        file_total=len(files),
        status=jobstate.STATUS_QUEUED,
        message="Queued.",
    )
    for name in files:
        store.set_file_state(job.id, name, "queued")

    # daemon=True so a half-finished job cannot keep the process alive after
    # the window is closed.
    thread = threading.Thread(
        target=_run_job,
        args=(current_app._get_current_object(), job.id),
        daemon=True,
        name="pipeline-%s" % job.id,
    )
    thread.start()

    return jsonify({"job_id": job.id, "files": files, "model": options["model"],
                    "mode": options["mode"], "steps": options["steps"],
                    "fixes_only": options["fixes_only"]})


def _busy_message(path: Path, exc: Exception) -> str:
    """Why a PDF could not be rewritten, in words the user can act on."""
    if isinstance(exc, runner.FixedPdfBusy):
        return str(exc)
    return ("%s could not be written (%s). If it is open in a PDF viewer, close it "
            "and try again." % (path.name, exc))


def _fixes_detail(repeats: int, rotated: int, review: int) -> str:
    """One queue-row phrase for a file's page fixes; empty when there were
    none."""
    parts = []
    if repeats:
        parts.append("%d possible repeated page(s) to check" % repeats)
    if rotated:
        parts.append("%d page(s) rotated" % rotated)
    if review:
        parts.append("%d to check by hand" % review)
    return ", ".join(parts)


def _run_job(app, job_id: str) -> None:
    """Background worker: drive the pipeline, translating its events into
    job-store updates.

    Runs inside an app context so it reads config the way a route does. Every
    exception is caught and recorded on the job: an unhandled one on a
    background thread would vanish silently and leave the UI polling a job that
    never moves.
    """
    with app.app_context():
        store: jobstate.JobStore = app.config["JOB_STORE"]
        job = store.get(job_id)
        if job is None:
            return

        # Why a document produced no draft, keyed by filename, taken from the
        # script's own warnings so the queue states the real reason.
        problems: dict = {}

        def report(**event) -> None:
            """Translate one pipeline event into job state.

            One function dispatching on ``event``, so all the UI-facing wording
            lives in one place.
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
                             message=event.get("detail", "Checking pages"))
                store.set_file_state(job_id, event["filename"], "fixing")

            elif kind == "fix_done":
                detail = _fixes_detail(event.get("repeats", 0), event.get("rotated", 0),
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
                # Only the fields this event carries are updated, so a missing
                # total cannot blank out the page_total already established --
                # and nothing is read off the job unlocked.
                fields = {"page_current": event.get("current", 0)}
                if event.get("total"):
                    fields["page_total"] = event["total"]
                store.update(job_id, **fields)

            elif kind == "file_problem":
                # Recorded against the file the script is on, so the queue row
                # can say why this document produced nothing.
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
                steps=job.steps,
            )

            documents = result["documents"]
            for doc in documents:
                # Repeated here because "done" replaces the dedupe/rotate
                # details the row showed while the job ran.
                parts = [] if doc["fixes_only"] else ["%d flag(s)" % len(doc["flags"])]
                if doc["abstract_pages"]:
                    parts.append("abstract pages %s" % doc["abstract_pages"]["requested"])
                fixes = _fixes_detail(doc["duplicates_flagged"], doc["pages_rotated"],
                                      doc["pages_for_review"])
                if fixes:
                    parts.append(fixes)
                if doc["fixes_only"] and not parts:
                    parts.append("no page fixes needed")
                store.set_file_state(job_id, doc["filename"], "done", ", ".join(parts))

            # A document the script skipped -- no abstract heading found --
            # produces no draft and never appears in ``documents``. Say so
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
                # The reason the script actually gave. Distinct reasons are
                # joined so a mixed batch does not hide one behind another.
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
            # The folder now holds everything needed to review it later.
            _persist(job_id)

        except Exception as exc:
            store.append_log(job_id, traceback.format_exc())
            store.update(job_id,
                         status=jobstate.STATUS_ERROR,
                         error=str(exc),
                         message="Failed: %s" % exc)
            # Saved even so: a failed run's folder is still worth opening.
            _persist(job_id)


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
    request, so only the app's own folders can be opened. That is what makes it
    safe to hand to the OS: ``os.startfile`` on a folder opens it, it does not
    run anything.
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


# ---- Review ----

def _find_document(job, name: str):
    """Look a document up by name on a job, or return None.

    Matching the name recorded in the job, rather than building a path from the
    request, is what stops these routes reading arbitrary files off the disk.
    """
    for doc in job.documents:
        if doc["name"] == name:
            return doc
    return None


def _span_key(page_no, index) -> str:
    """Identity of one diff span within a document."""
    return "%s:%s" % (page_no, index)


def _diffs_with_state(diffs: list, text: str, offsets: dict) -> list:
    """Copy the diff blocks, tagging each span with the side the document
    currently shows.

    Worked out from the live text on every request rather than stored, so the
    buttons stay honest after a hand edit or a change of mind. ``offsets``
    supplies each span's last known position, resolving what the text alone
    cannot.

    A span whose text appears more than once also gets ``occurrences`` to pick
    from, ``selected`` (the copy currently changed) and ``suggested`` (the
    likeliest copy, while none is picked).
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

    Returns the in-memory edited text when this session has changed it, so
    switching documents in the sidebar does not discard unsaved work.
    """
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "No such job."}), 404

    doc = _find_document(job, name)
    if doc is None:
        return jsonify({"error": "No such document."}), 404

    # A page-fixes-only document has no text: its product is the PDF.
    on_disk = ""
    if doc["md_file"]:
        md_path = job.workdir / runner.OUTPUT_DIRNAME / doc["md_file"]
        on_disk = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    current = job.edits.get(name, on_disk)

    return jsonify({
        "name": doc["name"],
        "filename": doc["filename"],
        "fixes_only": doc["fixes_only"],
        "fixes_recorded": doc.get("fixes_recorded", True),
        "steps": doc.get("steps"),
        "abstract_pages": doc["abstract_pages"],
        "text": current,
        "saved_text": on_disk,
        "dirty": name in job.edits and job.edits[name] != on_disk,
        "flags": doc["flags"],
        "diffs": _diffs_with_state(doc["diffs"], current,
                                   job.span_offsets.get(name, {})),
        "recovery": doc["recovery"],
        "page_count": doc["page_count"],
        "original_page_count": doc["original_page_count"],
        "duplicates_removed": doc["duplicates_removed"],
        "duplicates_flagged": doc["duplicates_flagged"],
        "duplicates": doc["duplicates"],
        "fixes_pending": doc.get("fixes_pending", False),
        "pending_removals": _pending_removal_count(doc),
        "pages_rotated": doc["pages_rotated"],
        "pages_for_review": doc["pages_for_review"],
        "rotations": doc["rotations"],
        "model_used": doc["model_used"],
        "mode_used": doc["mode_used"],
        "processed_at": doc["processed_at"],
    })


# Which copy of a PDF a page image may be drawn from. A fixed map, never a
# path from the request, so the route cannot be pointed elsewhere on disk.
_PAGE_IMAGE_SOURCES = {
    "original": runner.UPLOADS_DIRNAME,
    "fixed": runner.FIXED_DIRNAME,
}

# PyMuPDF is not thread-safe and Flask answers each thumbnail request on its
# own thread, so the renders take turns.
_render_lock = threading.Lock()


@bp.get("/page-image/<job_id>/<source>/<int:page_idx>/<path:name>")
def page_image(job_id: str, source: str, page_idx: int, name: str):
    """A PNG of one page, for the before/after views on the Page fixes tab.

    ``page_idx`` is 0-based within the chosen copy. Rendering respects the
    page's /Rotate, which is what makes a rotation fix visible: the original
    shows the page as scanned, the fixed copy as corrected.
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
            # Enough to judge orientation or spot a repeated page, and keeps
            # a document with dozens of fixes quick to open.
            png = pdf[page_idx].get_pixmap(dpi=60).tobytes("png")
        finally:
            pdf.close()

    response = send_file(io.BytesIO(png), mimetype="image/png")
    # Uploads never change, and the UI versions a fixed-PDF page's URL
    # whenever its rotation changes, so caching stays correct.
    response.headers["Cache-Control"] = "private, max-age=3600"
    return response


@bp.post("/rotation/<job_id>/<path:name>")
def set_rotation(job_id: str, name: str):
    """Change how one page is turned in the fixed PDF.

    For the rotation step what the OCR/VLM swap is for the text: undo the
    script's turn, turn a page it was unsure about, or turn one the other way.
    Body ``{original_idx, turn}``, turn being 0, 90, 180 or 270 degrees on top
    of the page as scanned.

    Only pages the rotation step reported on can be changed, which is also what
    keeps this route to the job's own files: the page is looked up in the
    document's record, never taken as a path.
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
    if record["fixed_idx"] is None:
        return jsonify({"error": "That page has been removed as a repeat - keep it to turn it."}), 409

    fixed_pdf = job.workdir / runner.FIXED_DIRNAME / doc["filename"]
    original_pdf = job.workdir / runner.UPLOADS_DIRNAME / doc["filename"]
    if not fixed_pdf.exists() or not original_pdf.exists():
        return jsonify({"error": "The PDFs for this document are no longer in its folder."}), 404

    # Shares the page-image lock: a thumbnail must not render mid-save.
    try:
        with _render_lock:
            runner.set_page_rotation(original_pdf, fixed_pdf, record["original_idx"],
                                     record["fixed_idx"], turn)
    except (runner.FixedPdfBusy, OSError) as exc:
        return jsonify({"error": _busy_message(fixed_pdf, exc)}), 409
    except IndexError:
        # The PDFs in the folder no longer match what this run recorded.
        return jsonify({"error": "Page %d is not in %s any more - the PDFs in this "
                                 "folder have changed since the run."
                                 % (record["original_page"], doc["filename"])}), 409

    record["current"] = turn
    record["decided"] = True
    _persist(job_id)
    # The counts behind the sidebar chips follow what the PDF now says.
    doc["pages_rotated"] = sum(1 for r in doc["rotations"] if r["current"])
    doc["pages_for_review"] = sum(1 for r in doc["rotations"]
                                  if not r["applied"] and not r["decided"])

    return jsonify({
        "rotations": doc["rotations"],
        "pages_rotated": doc["pages_rotated"],
        "pages_for_review": doc["pages_for_review"],
    })


@bp.post("/duplicate/<job_id>/<path:name>")
def set_duplicate(job_id: str, name: str):
    """Remove a suspected repeated page from the fixed PDF, or put it back.

    The dedupe step only flags repeats -- it has been wrong too often to act on
    its own -- so a person decides here, as with the rotation switch. Body
    ``{dupe_idx, remove}``; only pages dedupe flagged can be removed, looked up
    in the document's record.

    Only the choice is recorded, so switching back and forth is instant. The
    fixed PDF is rebuilt once, on save (see _save_page_fixes): removing a page
    shifts every page after it, so the rebuild is a full rewrite, too slow to
    run on every click.
    """
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "No such job."}), 404
    doc = _find_document(job, name)
    if doc is None:
        return jsonify({"error": "No such document."}), 404

    payload = request.get_json(silent=True) or {}
    remove = payload.get("remove")
    if not isinstance(remove, bool):
        return jsonify({"error": "remove must be true or false."}), 400
    record = next((d for d in doc["duplicates"]
                   if d["dupe_idx"] == payload.get("dupe_idx")), None)
    if record is None:
        return jsonify({"error": "That page was not flagged as a repeat."}), 404

    record["removed"] = remove
    record["decided"] = True
    doc["fixes_pending"] = _removals_differ_from_file(doc)
    _persist(job_id)

    return jsonify({
        "duplicates": doc["duplicates"],
        "fixes_pending": doc["fixes_pending"],
        "pending_removals": _pending_removal_count(doc),
    })


def _removals_differ_from_file(doc) -> bool:
    """Whether the Keep/Remove choices differ from what the fixed PDF holds.

    ``saved_removed`` is what was removed at the last save, so a difference
    from it is a choice not yet written.
    """
    return any(d["removed"] != d.get("saved_removed", False) for d in doc["duplicates"])


def _pending_removal_count(doc) -> int:
    return sum(1 for d in doc["duplicates"] if d["removed"] != d.get("saved_removed", False))


def _save_page_fixes(job, doc) -> bool:
    """Rebuild one document's fixed PDF with its Keep/Remove choices.

    Returns True if the file was rewritten. Rotations already in the file carry
    over, since the rebuild reapplies every current turn.

    The choices are copied when the rebuild starts and only that copy is
    recorded as saved: Keep/Remove stays clickable during a save, so a click
    landing mid-rebuild is left pending for the next one.
    """
    fixed_pdf = job.workdir / runner.FIXED_DIRNAME / doc["filename"]
    original_pdf = job.workdir / runner.UPLOADS_DIRNAME / doc["filename"]

    # Shares the page-image lock: a thumbnail must not render mid-save, and
    # two saves must not rebuild the same file at once.
    with _render_lock:
        if not _removals_differ_from_file(doc):
            return False
        if not original_pdf.exists():
            raise FileNotFoundError("The uploaded PDF for %s is no longer in this job's folder."
                                    % doc["filename"])
        snapshot = {d["dupe_idx"]: d["removed"] for d in doc["duplicates"]}
        keep = runner.rebuild_fixed_pdf(
            original_pdf, fixed_pdf,
            [dict(d, removed=snapshot[d["dupe_idx"]]) for d in doc["duplicates"]],
            doc["rotations"])
        for d in doc["duplicates"]:
            d["saved_removed"] = snapshot[d["dupe_idx"]]

    # Anything clicked while the file was being written is still pending.
    doc["fixes_pending"] = _removals_differ_from_file(doc)
    doc["page_count"] = len(keep)
    doc["duplicates_removed"] = doc["original_page_count"] - len(keep)
    return True


def unsaved_changes(store: jobstate.JobStore) -> list:
    """Every document, across this session's jobs, with changes not on disk.

    Text counts when the in-memory edit differs from its .md file, page fixes
    when Keep/Remove choices differ from the fixed PDF. Asked before the window
    closes, since all of it lives in memory. Returns
    ``[{"job_id", "name", "text", "page_fixes"}, ...]``.
    """
    found = []
    for job in store.all():
        for doc in job.documents:
            name = doc["name"]
            text = False
            if doc["md_file"] and name in job.edits:
                md_path = job.workdir / runner.OUTPUT_DIRNAME / doc["md_file"]
                on_disk = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
                text = job.edits[name] != on_disk
            page_fixes = _removals_differ_from_file(doc)
            if text or page_fixes:
                found.append({"job_id": job.id, "name": name,
                              "text": text, "page_fixes": page_fixes})
    return found


@bp.get("/unsaved")
def unsaved():
    """What would be lost if the window closed now."""
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    return jsonify({"unsaved": unsaved_changes(store)})


@bp.post("/edit/<job_id>/<path:name>")
def edit(job_id: str, name: str):
    """Hold an edited document in memory. Nothing is written to disk here.

    The review screen posts here as the user types, debounced. Holding edits
    until an explicit Save means a stray keystroke can never damage the
    pipeline's own output.
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
    # Typing arrives on every pause, so these writes are coalesced.
    _persist_soon(current_app._get_current_object(), job_id)
    return jsonify({"ok": True, "dirty": True})


@bp.post("/apply/<job_id>/<path:name>")
def apply_spans(job_id: str, name: str):
    """Apply one diff span -- or every span still on the OCR side -- in place.

    Saves the user from scrolling the editor to find the words a diff is
    talking about. Like /edit, only the in-memory copy changes; nothing reaches
    disk until Save.

    Body is either a single ``{page, index, direction}`` or
    ``{direction, all: true}`` to sweep every span not already on the requested
    side. A single span may also carry ``at_offset``: the copy the user picked
    when its text appears more than once.
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

    # The spans to act on.
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

    # Only meaningful for one span: a position in the text as it stands.
    at_offset = None if payload.get("all") else payload.get("at_offset")
    if at_offset is not None and (isinstance(at_offset, bool) or not isinstance(at_offset, int)):
        return jsonify({"error": "at_offset must be a whole number."}), 400

    counts = {"applied": 0, "unchanged": 0, "not_found": 0,
              "ambiguous": 0, "unplaceable": 0, "moved": 0}
    skipped = []
    for page_no, index, span in targets:
        key = _span_key(page_no, index)
        hint = offsets.get(key)

        # Skip spans already on the requested side, so a bulk apply does not
        # report them as failures. A picked copy is exempt: picking a different
        # copy of an applied span moves the change, it is not a no-op.
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
            # Remember where this span now sits. The other spans are left
            # alone: each record carries the words either side of its own span,
            # so it is still found after an edit elsewhere moves it.
            offsets[key] = new_hint
        else:
            skipped.append({"page": page_no, "index": index, "status": status,
                            "ocr": span["ocr"], "vlm": span["vlm"]})

    job.edits[name] = text
    _persist(job_id)

    return jsonify({
        "ok": True,
        "text": text,
        "counts": counts,
        "skipped": skipped,
        "diffs": _diffs_with_state(doc["diffs"], text, offsets),
    })


@bp.post("/save/<job_id>")
def save(job_id: str):
    """Write everything pending to disk: text edits to the .md files, and
    Keep/Remove choices to the fixed PDFs.

    The only route that overwrites the pipeline's output, and it runs only when
    the user presses Save.
    """
    store: jobstate.JobStore = current_app.config["JOB_STORE"]
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "No such job."}), 404

    payload = request.get_json(silent=True) or {}
    names = payload.get("names") or [d["name"] for d in job.documents]

    written = []
    page_fixes = {}
    # What each document still has unsaved once this save is done: edits and
    # clicks can land mid-save, so the review screen sets its marks from this
    # rather than assuming everything was written.
    saved_text = {}
    still_pending = {}
    for name in names:
        doc = _find_document(job, name)
        if doc is None:
            continue
        if doc["md_file"] and name in job.edits:
            text = job.edits[name]
            path = job.workdir / runner.OUTPUT_DIRNAME / doc["md_file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            written.append(doc["md_file"])
            saved_text[name] = text
        try:
            rebuilt = _save_page_fixes(job, doc)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc), "written": written}), 404
        except (runner.FixedPdfBusy, OSError) as exc:
            # The choices stay pending, so pressing Save again is enough.
            return jsonify({"error": _busy_message(
                job.workdir / runner.FIXED_DIRNAME / doc["filename"], exc),
                "written": written}), 409
        if rebuilt:
            written.append("%s/%s" % (runner.FIXED_DIRNAME, doc["filename"]))
            # The rebuild moved pages, so the review screen needs the new
            # positions and counts for this document.
            page_fixes[name] = {
                "rotations": doc["rotations"],
                "duplicates": doc["duplicates"],
                "page_count": doc["page_count"],
                "duplicates_removed": doc["duplicates_removed"],
            }
        still_pending[name] = {
            "fixes": doc.get("fixes_pending", False),
            "pending_removals": _pending_removal_count(doc),
        }

    _persist(job_id)

    return jsonify({"ok": True, "written": written, "page_fixes": page_fixes,
                    "saved_text": saved_text, "still_pending": still_pending})
