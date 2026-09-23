"""Job and progress tracking, held in memory and saved to each run's folder.

Why in memory and not a database: this is a single-user desktop tool. Only one
process ever touches this state, and the results themselves live on disk as
real files in ``workdir/``. A database here would add a schema, a migration
story, and a file to corrupt, in exchange for nothing.

Why it is also written to a file: a run is reviewed by a person, possibly a
different person, possibly days later. Everything the review screen needs that
is *not* already a file -- the parsed flags and OCR/VLM differences, which copy
of a repeated span was picked, the Keep/Remove and page-turn choices, unsaved
edits, and the run's own settings -- is saved as ``job.json`` in the run's own
folder (see save_job). The folder then holds the whole run and can be copied to
another machine and opened there (see load_job).

Why a lock: the pipeline runs on a background thread so the UI stays
responsive, but Flask answers ``/status`` on a *different* thread. Both touch
the same Job objects. The lock is what keeps a status poll from reading a job
halfway through an update.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Job lifecycle. Kept as plain strings because they are sent to the UI as JSON
# and read there directly; an enum would only have to be converted back.
STATUS_QUEUED = "queued"
STATUS_PREFLIGHT = "preflight"
STATUS_FIXING = "fixing"
STATUS_OCR = "ocr"
STATUS_COLLECTING = "collecting"
STATUS_DONE = "done"
STATUS_ERROR = "error"

# Terminal states -- the UI stops polling when it sees one of these.
FINISHED = (STATUS_DONE, STATUS_ERROR)

# The run's saved state, and a short summary read when listing past runs
# (job.json can run to megabytes on a large batch, which is too much to read
# just to show a row per run).
STATE_FILE = "job.json"
SUMMARY_FILE = "run.json"

# Bumped only when a change would confuse an older build. A file whose version
# is higher than this is refused rather than half-read.
STATE_FORMAT = "ocr-pipeline-studio/job"
STATE_VERSION = 1


@dataclass
class Job:
    """One run of the pipeline over one batch of PDFs.

    ``edits`` is the review screen's scratch space: the textarea contents as
    the user has changed them, keyed by document name. It is deliberately
    separate from what is on disk. Edits only reach the filesystem when the
    user explicitly saves, which is what makes the review screen safe to
    experiment in.
    """
    id: str
    workdir: Path
    files: list = field(default_factory=list)          # original uploaded names
    model: str = ""
    mode: str = ""
    # Which steps this job runs, as {"dedupe", "rotate", "ocr"} booleans.
    steps: dict = field(default_factory=lambda: {"dedupe": True, "rotate": True, "ocr": True})
    # A job without OCR stops after the page fixes: no text. Kept alongside
    # ``steps`` because everything that only asks "is there text?" uses it.
    fixes_only: bool = False
    # Abstract pages set by hand, as {filename: (start, end)} in the uploaded
    # PDF's own page numbers.
    overrides: dict = field(default_factory=dict)

    status: str = STATUS_QUEUED
    message: str = "Waiting to start."
    error: Optional[str] = None

    # Batch position -- "file 2 of 5"
    file_index: int = 0
    file_total: int = 0
    current_file: Optional[str] = None

    # Page position within the current file -- "3 of 12 pages"
    page_current: int = 0
    page_total: int = 0

    # Per-file states for the batch queue display, keyed by filename.
    file_states: dict = field(default_factory=dict)

    documents: list = field(default_factory=list)      # finished output records
    edits: dict = field(default_factory=dict)          # in-memory textarea state

    # Where each diff span was last written, as
    # ``{document name: {"<page>:<index>": offset}}``. This is what lets the
    # review screen swap a span back and forth reliably: once a correction has
    # been applied the same words can occur in two places, and only a
    # remembered position can say which one belongs to that diff.
    span_offsets: dict = field(default_factory=dict)

    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # True for a run opened from a folder rather than run in this window. The
    # review screen says so, since its progress and log belong to the run that
    # produced it, not to this session.
    opened_from_folder: bool = False

    # Bounded so a very long batch cannot grow the log without limit. 400 lines
    # is far more than the UI shows but enough to diagnose a failed run.
    log: deque = field(default_factory=lambda: deque(maxlen=400))

    def progress_label(self) -> str:
        """The single human-readable line the UI puts under the progress bar."""
        if self.status in FINISHED:
            return self.message
        # page_current stays 0 until the first page actually finishes, and
        # "page 0 of 5" reads like a fault rather than a start.
        if self.page_total and self.page_current:
            return "%s - page %d of %d" % (self.message, self.page_current, self.page_total)
        if self.page_total:
            return "%s - %d pages to read" % (self.message, self.page_total)
        return self.message


# Long enough to recognise a thesis call number, short enough to keep Windows
# paths well under their limit once a job's own subfolders are added.
_LABEL_MAX = 40


def _folder_label(filenames: list) -> str:
    """The "what is in it" part of a job folder name.

    The first file's name without its extension, plus how many others came
    with it. Anything other than letters, digits, dot, dash and underscore is
    dropped, so the result is safe as a folder name and in a URL.
    """
    if not filenames:
        return ""
    stem = Path(filenames[0]).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "", stem).strip(".")[:_LABEL_MAX]
    if not stem:
        return ""
    others = len(filenames) - 1
    return stem + ("_and_%d_more" % others if others else "")


class JobStore:
    """Thread-safe registry of Jobs, keyed by job id.

    Every public method takes the lock. Callers never touch ``_jobs``
    directly, so there is exactly one place where concurrent access is
    reasoned about.
    """

    def __init__(self) -> None:
        self._jobs: dict = {}
        self._lock = threading.Lock()

    def create(self, workdir_root: Path, filenames: Optional[list] = None) -> Job:
        """Make a new job with a server-generated id and its own scratch dir.

        The id doubles as the folder name, so it is built to be found by a
        person browsing ``workdir/``: when the job started, then what is in
        it -- ``2026-09-16_143205_Thesis80.W5465_and_2_more``. Sorting by name
        is sorting by time.

        The id is still generated here, on the server, and never taken from
        the client: it becomes a directory name, so letting the browser choose
        it would be a path-traversal hole in a tool that reads and writes
        files. The filename part is reduced to plain characters for the same
        reason.
        """
        base = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        label = _folder_label(filenames or [])
        if label:
            base += "_" + label

        workdir_root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            # Two jobs started in the same second over the same file would
            # otherwise share a folder. mkdir without exist_ok is the check:
            # it fails if the folder is already there, whoever made it.
            job_id = base
            suffix = 1
            while True:
                try:
                    (workdir_root / job_id).mkdir()
                    break
                except FileExistsError:
                    suffix += 1
                    job_id = "%s-%d" % (base, suffix)

            job = Job(id=job_id, workdir=workdir_root / job_id)
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list:
        with self._lock:
            return list(self._jobs.values())

    def update(self, job_id: str, **fields) -> None:
        """Apply attribute updates to a job atomically.

        Used by the background thread for every progress tick, so a status
        poll can never observe a job with, say, a new page number but a stale
        filename.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)

    def append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.log.append(line)

    def set_file_state(self, job_id: str, filename: str, state: str, detail: str = "") -> None:
        """Record per-file status for the batch queue list in the UI."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.file_states[filename] = {"state": state, "detail": detail}

    def state_payload(self, job_id: str) -> Optional[dict]:
        """One run's saveable state, assembled under the lock.

        Carries the run's folder as ``_workdir`` for save_job to write into;
        that key is removed before anything is written, since a saved path
        would be wrong the moment the folder is copied somewhere else.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            payload = _state_payload(job)
            payload["_workdir"] = str(job.workdir)
            return payload

    def register(self, folder: Path, payload: Optional[dict] = None,
                 documents: Optional[list] = None) -> Job:
        """Add a job for a run folder that already exists on disk.

        Used when a past run is opened: from its saved state, or -- for a
        folder with none -- from ``documents`` rebuilt out of the files
        themselves. The id is the folder's own name, suffixed if a run of that
        name is already open, so two copies of the same run can be open at
        once.
        """
        saved = (payload or {}).get("job", {})
        with self._lock:
            job_id = folder.name
            suffix = 1
            while job_id in self._jobs:
                suffix += 1
                job_id = "%s-%d" % (folder.name, suffix)

            job = Job(
                id=job_id,
                workdir=folder,
                files=list(saved.get("files") or []),
                model=saved.get("model", ""),
                mode=saved.get("mode", ""),
                steps=dict(saved.get("steps") or {"dedupe": True, "rotate": True, "ocr": True}),
                fixes_only=bool(saved.get("fixes_only", False)),
                overrides={name: tuple(pages)
                           for name, pages in (saved.get("overrides") or {}).items()},
                # Opened runs are finished by definition: nothing is running.
                status=STATUS_DONE,
                message=saved.get("message") or "Opened from a folder.",
                file_states=dict(saved.get("file_states") or {}),
                documents=documents if documents is not None else (payload or {}).get("documents") or [],
                edits=dict((payload or {}).get("edits") or {}),
                span_offsets=(payload or {}).get("span_offsets") or {},
                created_at=saved.get("created_at") or datetime.now().isoformat(timespec="seconds"),
            )
            job.opened_from_folder = True
            if not job.files:
                job.files = sorted(d["filename"] for d in job.documents)
            self._jobs[job_id] = job
        return job

    def snapshot(self, job_id: str, log_lines: int = 40) -> Optional[dict]:
        """Build the JSON payload for ``/status/<job_id>``.

        Assembled under the lock and returned as a plain dict, so the caller
        serialises a consistent point-in-time copy rather than a live object
        the worker thread is still mutating.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "job_id": job.id,
                "folder": str(job.workdir),
                "opened_from_folder": job.opened_from_folder,
                "status": job.status,
                "finished": job.status in FINISHED,
                "message": job.message,
                "label": job.progress_label(),
                "error": job.error,
                "model": job.model,
                "mode": job.mode,
                "fixes_only": job.fixes_only,
                "steps": dict(job.steps),
                "files": job.files,
                "file_index": job.file_index,
                "file_total": job.file_total,
                "current_file": job.current_file,
                "page_current": job.page_current,
                "page_total": job.page_total,
                "file_states": dict(job.file_states),
                "documents": [
                    # The review screen needs names and flags up front; the
                    # bulky text/diff payloads are fetched per document.
                    {
                        "name": d["name"],
                        "filename": d["filename"],
                        "fixes_only": d["fixes_only"],
                        "page_count": d["page_count"],
                        "duplicates_removed": d["duplicates_removed"],
                        "duplicates_flagged": d["duplicates_flagged"],
                        "pages_rotated": d["pages_rotated"],
                        "pages_for_review": d["pages_for_review"],
                        "model_used": d["model_used"],
                        "flags": d["flags"],
                        "diff_pages": len(d["diffs"]),
                    }
                    for d in job.documents
                ],
                "created_at": job.created_at,
                "log": list(job.log)[-log_lines:],
            }


# --------------------------------------------------------------------------
# Saving a run to its folder, and reading it back
# --------------------------------------------------------------------------

def _state_payload(job: Job) -> dict:
    """Everything about a run that is not already a file in its folder.

    Only names are stored, never paths: ``workdir`` is wherever the folder is
    found when it is opened, and every route builds its paths from that. That
    is what lets a run folder be copied to another machine.
    """
    return {
        "format": STATE_FORMAT,
        "version": STATE_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "job": {
            "id": job.id,
            "files": list(job.files),
            "model": job.model,
            "mode": job.mode,
            "steps": dict(job.steps),
            "fixes_only": job.fixes_only,
            # Tuples would come back from JSON as lists; the pipeline wants
            # (start, end), so they are restored as tuples in load_job.
            "overrides": {name: list(pages) for name, pages in job.overrides.items()},
            "status": job.status,
            "message": job.message,
            "error": job.error,
            "file_states": dict(job.file_states),
            "created_at": job.created_at,
        },
        "documents": job.documents,
        "edits": dict(job.edits),
        "span_offsets": job.span_offsets,
    }


def _summary_payload(payload: dict) -> dict:
    """The few fields the "past runs" list needs."""
    job = payload["job"]
    return {
        "format": STATE_FORMAT,
        "version": STATE_VERSION,
        "saved_at": payload["saved_at"],
        "created_at": job["created_at"],
        "files": job["files"],
        "documents": len(payload["documents"]),
        "status": job["status"],
        "steps": job["steps"],
    }


def _write_json(path: Path, payload: dict) -> None:
    """Write JSON through a temp file, so a reader never sees half of it."""
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(temp, path)


def save_job(store: "JobStore", job_id: str) -> Optional[Path]:
    """Write one run's state into its own folder.

    The payload is built under the store's lock, for the same reason
    snapshot() is -- the worker thread may be mid-update -- and written
    outside it, since writing megabytes must not hold up a status poll.
    """
    payload = store.state_payload(job_id)
    if payload is None:
        return None
    folder = Path(payload.pop("_workdir"))
    if not folder.is_dir():
        return None
    _write_json(folder / STATE_FILE, payload)
    _write_json(folder / SUMMARY_FILE, _summary_payload(payload))
    return folder / STATE_FILE


def read_summary(folder: Path) -> Optional[dict]:
    """One past run's summary row, or None if this folder is not a run."""
    path = folder / SUMMARY_FILE
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if not isinstance(summary, dict):
        return None
    summary["folder"] = str(folder)
    summary["name"] = folder.name
    summary["reviewable"] = (folder / STATE_FILE).is_file()
    return summary


class StateError(Exception):
    """A run folder that cannot be opened, with a message for the user."""


def load_state(folder: Path) -> Optional[dict]:
    """Read a folder's saved state, or None when it has none.

    Raises StateError for a file that is there but unusable, so "no saved
    state, rebuild what we can" stays distinct from "this file is wrong".
    """
    path = folder / STATE_FILE
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise StateError("%s in this folder could not be read (%s)." % (STATE_FILE, exc))
    if not isinstance(payload, dict) or payload.get("format") != STATE_FORMAT:
        raise StateError("%s in this folder was not written by this app." % STATE_FILE)
    if int(payload.get("version", 0)) > STATE_VERSION:
        raise StateError("This run was saved by a newer version of the app (format %s, "
                         "this build reads %s). Update the app to open it."
                         % (payload.get("version"), STATE_VERSION))
    return payload
