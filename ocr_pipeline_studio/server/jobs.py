"""In-memory job and progress tracking.

Why in-memory and not a database: this is a single-user desktop tool. Only one
process ever touches this state, it is worthless once the window closes, and
the durable results already live on disk as real files in ``workdir/``. A
database here would add a schema, a migration story, and a file to corrupt, in
exchange for nothing.

Why a lock: the pipeline runs on a background thread so the UI stays
responsive, but Flask answers ``/status`` on a *different* thread. Both touch
the same Job objects. The lock is what keeps a status poll from reading a job
halfway through an update.
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Job lifecycle. Kept as plain strings because they are sent to the UI as JSON
# and read there directly; an enum would only have to be converted back.
STATUS_QUEUED = "queued"
STATUS_PREFLIGHT = "preflight"
STATUS_DEDUPE = "dedupe"
STATUS_ROTATE = "rotate"
STATUS_OCR = "ocr"
STATUS_COLLECTING = "collecting"
STATUS_DONE = "done"
STATUS_ERROR = "error"

# Terminal states -- the UI stops polling when it sees one of these.
FINISHED = (STATUS_DONE, STATUS_ERROR)


@dataclass
class Job:
    """One run of the pipeline over one batch of PDFs.

    ``edits`` is the review screen's scratch space: the textarea contents as
    the user has changed them, keyed by document name. It is deliberately
    separate from what is on disk. Edits only reach the filesystem when the
    user explicitly saves or exports, which is what makes the review screen
    safe to experiment in.
    """
    id: str
    workdir: Path
    files: list = field(default_factory=list)          # original uploaded names
    model: str = ""
    mode: str = ""
    # A page-fixes-only job stops after dedupe + rotation: no OCR, no text.
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


class JobStore:
    """Thread-safe registry of Jobs, keyed by job id.

    Every public method takes the lock. Callers never touch ``_jobs``
    directly, so there is exactly one place where concurrent access is
    reasoned about.
    """

    def __init__(self) -> None:
        self._jobs: dict = {}
        self._lock = threading.Lock()

    def create(self, workdir_root: Path) -> Job:
        """Make a new job with a server-generated id and its own scratch dir.

        The id is generated here, on the server, and never taken from the
        client: it becomes a directory name, so letting the browser choose it
        would be a path-traversal hole in a tool that reads and writes files.
        """
        job_id = uuid.uuid4().hex[:12]
        workdir = workdir_root / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, workdir=workdir)
        with self._lock:
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
                "status": job.status,
                "finished": job.status in FINISHED,
                "message": job.message,
                "label": job.progress_label(),
                "error": job.error,
                "model": job.model,
                "mode": job.mode,
                "fixes_only": job.fixes_only,
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
