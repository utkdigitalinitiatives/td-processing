"""Job and progress tracking, held in memory and saved to each run's folder.

In memory, not a database: one process touches this state and the results
themselves are real files in ``workdir/``.

Also written to a file, because a run may be reviewed later, by someone else,
on another machine. Everything the review screen needs that is not already a
file -- flags, OCR/VLM differences, which copy of a repeated span was picked,
Keep/Remove and page-turn choices, unsaved edits, and the run's settings --
goes into ``job.json`` in the run's own folder, so the folder holds the whole
run (see save_job / load_state).

The lock is what keeps a ``/status`` poll, answered on a Flask thread, from
reading a Job the pipeline's background thread is halfway through updating.
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

# Job lifecycle. Plain strings because they go to the UI as JSON and are read
# there directly; an enum would only have to be converted back.
STATUS_QUEUED = "queued"
STATUS_PREFLIGHT = "preflight"
STATUS_FIXING = "fixing"
STATUS_OCR = "ocr"
STATUS_COLLECTING = "collecting"
STATUS_DONE = "done"
STATUS_ERROR = "error"

# Terminal states -- the UI stops polling when it sees one of these.
FINISHED = (STATUS_DONE, STATUS_ERROR)

# The run's saved state, plus a short summary for listing past runs: job.json
# can run to megabytes on a large batch, too much to read just to show a row.
STATE_FILE = "job.json"
SUMMARY_FILE = "run.json"

# Bumped only when a change would confuse an older build. A higher version is
# refused rather than half-read.
STATE_FORMAT = "ocr-pipeline-studio/job"
STATE_VERSION = 1


@dataclass
class Job:
    """One run of the pipeline over one batch of PDFs.

    ``edits`` is the review screen's scratch space: textarea contents keyed by
    document name, kept separate from what is on disk. Edits only reach the
    filesystem on an explicit save, which is what makes the review screen safe
    to experiment in.
    """
    id: str
    workdir: Path
    files: list = field(default_factory=list)          # original uploaded names
    model: str = ""
    mode: str = ""
    # Which steps this job runs, as {"dedupe", "rotate", "ocr"} booleans.
    steps: dict = field(default_factory=lambda: {"dedupe": True, "rotate": True, "ocr": True})
    # A job without OCR stops after the page fixes: no text. Kept alongside
    # ``steps`` for everything that only asks "is there text?"
    fixes_only: bool = False
    # Abstract pages set by hand, {filename: (start, end)}, numbered as in the
    # uploaded PDF.
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
    # ``{document name: {"<page>:<index>": offset}}``. Applying a correction
    # can make the same words occur twice, and only a remembered position can
    # say which one belongs to that diff.
    span_offsets: dict = field(default_factory=dict)

    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # True for a run opened from a folder rather than run in this window; its
    # progress and log belong to the run that produced it, not this session.
    opened_from_folder: bool = False

    # Bounded so a long batch cannot grow the log without limit. More than the
    # UI shows, but enough to diagnose a failed run.
    log: deque = field(default_factory=lambda: deque(maxlen=400))

    def progress_label(self) -> str:
        """The single human-readable line the UI puts under the progress bar."""
        if self.status in FINISHED:
            return self.message
        # page_current stays 0 until the first page finishes, and "page 0 of
        # 5" reads like a fault rather than a start.
        if self.page_total and self.page_current:
            return "%s - page %d of %d" % (self.message, self.page_current, self.page_total)
        if self.page_total:
            return "%s - %d pages to read" % (self.message, self.page_total)
        return self.message


# Long enough to recognise a thesis call number, short enough to keep Windows
# paths under their limit once a job's subfolders are added.
_LABEL_MAX = 40


def _folder_label(filenames: list) -> str:
    """The "what is in it" part of a job folder name.

    The first file's stem plus how many others came with it. Anything but
    letters, digits, dot, dash and underscore is dropped, so the result is safe
    as a folder name and in a URL.
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

    Every public method takes the lock and callers never touch ``_jobs``
    directly, so concurrent access is reasoned about in one place.
    """

    def __init__(self) -> None:
        self._jobs: dict = {}
        self._lock = threading.Lock()

    def create(self, workdir_root: Path, filenames: Optional[list] = None) -> Job:
        """Make a new job with a server-generated id and its own scratch dir.

        The id doubles as the folder name, built to be found by someone
        browsing ``workdir/``: start time then contents, as in
        ``2026-09-16_143205_Thesis80.W5465_and_2_more``, so sorting by name is
        sorting by time.

        Generated here and never taken from the client: it becomes a directory
        name, so letting the browser choose it would be a path-traversal hole.
        The filename part is reduced to plain characters for the same reason.
        """
        base = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        label = _folder_label(filenames or [])
        if label:
            base += "_" + label

        workdir_root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            # Two jobs started in the same second over the same file would
            # otherwise share a folder. mkdir without exist_ok is the check --
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

        Used for every progress tick, so a status poll can never see a job with
        a new page number but a stale filename.
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

        Carries the run's folder as ``_workdir`` for save_job to write into.
        That key is removed before anything is written: a saved path would be
        wrong the moment the folder is copied elsewhere.
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

        Built from the folder's saved state, or -- for a folder with none --
        from ``documents`` rebuilt out of the files themselves. The id is the
        folder's name, suffixed if a run of that name is already open.
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

        A plain dict assembled under the lock, so the caller serialises a
        point-in-time copy rather than an object the worker is still mutating.
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
                    # Names and flags up front; the bulky text/diff payloads
                    # are fetched per document.
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


# ---- Saving a run to its folder, and reading it back ----

def _state_payload(job: Job) -> dict:
    """Everything about a run that is not already a file in its folder.

    Names only, never paths: ``workdir`` is wherever the folder is found when
    opened and every route builds from that, which is what lets a run folder be
    copied to another machine.
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
            # JSON has no tuples; restored as (start, end) when loaded.
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

    Built under the store's lock, as snapshot() is, but written outside it:
    writing megabytes must not hold up a status poll.
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

    Raises StateError for a file that is there but unusable, keeping "no saved
    state, rebuild what we can" distinct from "this file is wrong".
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
