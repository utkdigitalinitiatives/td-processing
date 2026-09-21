"""Thin adapters around the original scripts in ``source_scripts/``.

Nothing in this file reimplements what those scripts already do. The copies
that live beside this module (``pipeline/dedupe.py``,
``pipeline/fix_rotation.py`` and ``pipeline/vlm_abstract.py``) are
byte-for-byte the files that were handed to us; this module only *calls* them
and reshapes their output into something a background thread and a JSON API
can work with.

Two very different calling styles are used here, and the reason for each
matters:

``dedupe.py`` and ``fix_rotation.py`` are called **in-process, as a normal
import**. Their core functions (``find_and_export_duplicates()`` and
``find_and_fix_rotations()``) already return real Python lists, so there is
nothing awkward to work around.

``vlm_abstract.py`` is called **as a subprocess, through its own CLI**. That is
deliberate and is the single most important design decision in this file:

  1. Its ``main()`` contains a "deferred VLM phase" that must run strictly
     after every PDF's PaddleOCR work is finished. The script's own comments
     explain why -- an Ollama model resident on the GPU has been observed to
     break PaddleOCR's initialization. Importing ``process_pdf()`` directly
     would skip ``main()`` entirely, so we would have to re-implement that
     phase ourselves. That would be rewriting the script's logic, which we
     were told not to do.
  2. The script already re-invokes *itself* as a subprocess (once per PDF, and
     again once per page for OCR crash isolation). Driving it by CLI is
     therefore the interface it was actually built to expose.
  3. It reports progress by printing to stdout rather than returning values.
     Wrapping it in a pipe lets us read that progress without editing it.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import fitz  # PyMuPDF -- used only to *write* the fixed PDF, see fix_pdf()

# The original detector, imported and used exactly as written.
from pipeline.dedupe import find_and_export_duplicates

# The original rotation fixer, likewise. Importing it only pulls in fitz --
# the script defers its paddleocr import to where the classifier is built.
from pipeline.fix_rotation import ORIENTATION_MODEL, find_and_fix_rotations

# Absolute path to the OCR/VLM script we shell out to. Resolved once at import
# time so a change of working directory later can never break the call.
VLM_SCRIPT = Path(__file__).resolve().parent / "vlm_abstract.py"

# Kept in sync with the script's own DEFAULT_VLM_MODEL / DEFAULT_OLLAMA_URL.
# Duplicated here rather than imported because importing vlm_abstract.py into
# this process would pull in PaddleOCR (slow, and it prints on import).
DEFAULT_VLM_MODEL = "qwen2.5vl:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# The VLM modes we expose in the UI, mapped to the script's real CLI flags.
# Keeping this as data (not an if/elif chain at the call site) means adding a
# mode later is a one-line change here plus one <option> in the UI.
VLM_MODES: dict[str, dict] = {
    "off": {
        "label": "OCR only (no VLM)",
        "flags": [],
        "help": "Fastest. PaddleOCR only -- Ollama is never contacted.",
    },
    "targeted": {
        "label": "VLM recovery on suspicious pages",
        "flags": ["--vlm-review", "--vlm-review-mode", "targeted"],
        "help": "Only re-reads pages with a suspiciously low OCR segment count.",
    },
    "always": {
        "label": "VLM recovery on every page",
        "flags": ["--vlm-review", "--vlm-review-mode", "always"],
        "help": "Re-reads every abstract page and appends the result for review.",
    },
    "diff": {
        "label": "OCR/VLM diff report (report only)",
        "flags": ["--vlm-diff-review"],
        "help": "Runs the VLM on every page and reports what it read differently. "
                "Never edits the draft text.",
    },
    "merge": {
        "label": "OCR/VLM diff + auto-merge safe fixes (recommended)",
        "flags": ["--vlm-diff-merge"],
        "help": "Applies the corrections the script judges safe straight into the "
                "draft, and reports the rest so you can apply them in one click.",
    },
}
DEFAULT_VLM_MODE = "merge"


# --------------------------------------------------------------------------
# Stage 0: Ollama preflight
# --------------------------------------------------------------------------
# Everything here runs *before* a job starts. The point is to fail loudly and
# in plain language at the top of the pipeline, rather than let a job die
# fifteen minutes deep with a connection traceback the user cannot act on.

def ollama_status(ollama_url: str = DEFAULT_OLLAMA_URL, timeout: float = 4.0) -> dict:
    """Ask Ollama what models it has, via the same REST endpoint the OCR
    script uses (``GET /api/tags``).

    Returns ``{"running": bool, "models": [...], "error": str|None}``. Never
    raises -- a preflight check that can itself explode is useless.
    """
    try:
        import requests
    except ImportError:
        return {"running": False, "models": [],
                "error": "The 'requests' package is not installed."}

    try:
        resp = requests.get(ollama_url + "/api/tags", timeout=timeout)
        resp.raise_for_status()
        models = [m.get("name") for m in resp.json().get("models", []) if m.get("name")]
        return {"running": True, "models": models, "error": None}
    except Exception as exc:
        return {
            "running": False,
            "models": [],
            # Phrased for the person looking at the window, not for a log file.
            "error": "Ollama isn't running - start it and try again. (%s)" % exc,
        }


def list_local_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> dict:
    """List the models actually pulled on this machine, for the UI dropdown.

    Shells out to ``ollama list`` as the primary source, because that is the
    command a user would run themselves and its output is what they expect to
    see. Falls back to the REST API when the CLI is not on PATH -- the app can
    still work in that case, since the pipeline only ever needs the HTTP API.
    """
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=15,
            encoding="utf-8", errors="replace",
        )
        if proc.returncode == 0:
            models = []
            # Output is a padded table: NAME  ID  SIZE  MODIFIED. The first
            # whitespace-delimited field of each line after the header is the
            # model tag, which is the only column we need.
            for line in proc.stdout.splitlines()[1:]:
                if not line.strip():
                    continue
                models.append(line.split()[0])
            if models:
                return {"models": models, "source": "ollama list", "error": None}
    except Exception:
        # Fall through to the REST API below -- no need to surface this, since
        # the fallback answers the same question.
        pass

    status = ollama_status(ollama_url)
    return {
        "models": status["models"],
        "source": "api/tags",
        "error": status["error"],
    }


def preflight(model: str, ollama_url: str = DEFAULT_OLLAMA_URL, mode: str = DEFAULT_VLM_MODE) -> dict:
    """Check Ollama is up and ``model`` is present, before any job starts.

    Returns ``{"ok": bool, "message": str|None}``. When the chosen mode makes
    no VLM calls at all ("off"), this passes unconditionally -- refusing to run
    a pure-OCR job because Ollama is down would be wrong.
    """
    if mode == "off":
        return {"ok": True, "message": None}

    status = ollama_status(ollama_url)
    if not status["running"]:
        return {"ok": False, "message": status["error"]}

    # The script matches model tags exactly against this same list, so an
    # exact check here is what actually predicts whether the job will work.
    if model not in status["models"]:
        return {
            "ok": False,
            # Deliberately tells the user to pull it themselves: a first pull
            # can be several GB, which is not something to start mid-job.
            "message": ("The model '%s' isn't pulled on this machine. "
                        "Run:  ollama pull %s" % (model, model)),
        }

    return {"ok": True, "message": None}


# --------------------------------------------------------------------------
# Stage 1: page fixes -- remove repeated pages, turn sideways ones upright
# --------------------------------------------------------------------------
# Both original scripts detect on the uploaded PDF, and their findings are
# then written out together in a single save into fixed/. Each script still
# reads the file in its own way -- their per-page logic is theirs, and is not
# merged or copied here -- but no intermediate copy is ever written: one read
# per script, one write per PDF.

# The confidence levels fix_rotation.py actually applies. "review" pages are
# reported but left alone -- the script's own rule, mirrored here only so the
# pages turned below are exactly the ones it would have turned.
_APPLIED_CONFIDENCE = ("high", "medium")


@dataclass
class PageFixResult:
    """What one PDF's page fixes found and did.

    A dataclass rather than a bare dict because these fields flow straight
    into manifest.json, and a typo in a dict key would only surface much
    later, in the UI, as a missing number.
    """
    source_name: str
    fixed_path: Path
    original_page_count: int
    # 0-based page numbers of the original PDF, in the order they were kept.
    # Position i in the fixed PDF is original page kept_indices[i]. Every page
    # is kept: suspected repeats are only flagged, and removed by a person.
    kept_indices: list = field(default_factory=list)
    # dedupe.py's findings, one dict per suspected repeat:
    #   {dupe_idx, orig_idx, duplicate_page, original_page, score, folio}
    duplicates: list = field(default_factory=list)
    # fix_rotation.py's decisions, numbered as in the original PDF:
    #   {page_idx, page, rotation, confidence, why, score, contradicted, img_agreed}
    rotations: list = field(default_factory=list)

    @property
    def kept_page_count(self) -> int:
        return len(self.kept_indices)

    @property
    def duplicates_removed(self) -> int:
        return self.original_page_count - self.kept_page_count

    @property
    def duplicates_flagged(self) -> int:
        return len(self.duplicates)

    @property
    def pages_rotated(self) -> int:
        return sum(1 for d in self.rotations if d["confidence"] in _APPLIED_CONFIDENCE)

    @property
    def pages_for_review(self) -> int:
        return sum(1 for d in self.rotations if d["confidence"] == "review")


def load_orientation_classifier():
    """Build the orientation model the rotation script uses, once per batch.

    The script builds this inside scan_directory(), which we do not call (it
    only prints its results), so the same construction is repeated here.
    Loading takes about a second, hence doing it once rather than per PDF.
    """
    from paddleocr import DocImgOrientationClassification

    return DocImgOrientationClassification(model_name=ORIENTATION_MODEL)


def fix_pdf(
    pdf_path: Path,
    fixed_dir: Path,
    classifier=None,
    dpi: int = 100,
    min_score: float = 0.70,
    dedupe: bool = True,
    rotate: bool = True,
) -> PageFixResult:
    """Find repeated and sideways pages with the original scripts, then write
    one corrected copy of the PDF into ``fixed_dir``.

    Repeated pages are **flagged, never removed**: dedupe.py's matches have
    proved wrong often enough that every one goes to a person, who removes it
    from the Page fixes tab if it really is a repeat (see
    rebuild_fixed_pdf()). Sideways pages the rotation script is confident
    about are turned, as before; its uncertain ones are left for review.

    Both scripts only detect here -- ``dedupe.py`` has no function that
    removes pages, and ``fix_rotation.py`` runs in its detect-only mode -- so
    the write happens in write_fixed_pdf(), driven entirely by what they
    returned. All of the matching and deciding stays in the original files.
    """
    fixed_dir.mkdir(parents=True, exist_ok=True)

    # Writing into the source's own folder would overwrite the original.
    if pdf_path.parent.resolve() == fixed_dir.resolve():
        raise ValueError("fixed_dir must differ from the folder holding %s" % pdf_path.name)

    # Either check can be left out (``dedupe`` / ``rotate``), when the person
    # running the batch only needs the other. Rotation is the slow one -- it
    # renders every page through the orientation model -- so skipping it also
    # skips loading that model.

    # We pass no output_dir, so it does not write its comparison PDF -- only
    # the detection result is wanted here.
    duplicates = find_and_export_duplicates(pdf_path) if dedupe else []

    # No output_dir and apply=False: the script only reports, it writes nothing.
    rotations = []
    if rotate:
        if classifier is None:
            classifier = load_orientation_classifier()
        rotations = find_and_fix_rotations(pdf_path, classifier, dpi=dpi, min_score=min_score)

    with fitz.open(pdf_path) as doc:
        original_page_count = doc.page_count
    keep = list(range(original_page_count))

    out_path = fixed_dir / pdf_path.name
    turns = {d["page_idx"]: d["rotation"] for d in rotations
             if d["confidence"] in _APPLIED_CONFIDENCE}
    write_fixed_pdf(pdf_path, out_path, keep, turns)

    return PageFixResult(
        source_name=pdf_path.name,
        fixed_path=out_path,
        original_page_count=original_page_count,
        kept_indices=keep,
        duplicates=duplicates,
        rotations=rotations,
    )


def write_fixed_pdf(original_pdf: Path, fixed_pdf: Path, keep: list, turns: dict) -> None:
    """Write the fixed PDF: the pages in ``keep``, each turned by ``turns``.

    ``keep`` is 0-based original page numbers in order; ``turns`` maps an
    original page number to degrees added to that page's own /Rotate (the
    same addition the rotation script's apply step makes). Always built from
    the untouched upload, so earlier choices never compound.

    Written beside the target and swapped in, so the fixed PDF is never left
    half-written if something goes wrong mid-save.
    """
    temp = fixed_pdf.with_name(fixed_pdf.stem + ".tmp.pdf")
    with fitz.open(original_pdf) as doc:
        removing = len(keep) < doc.page_count
        turning = {i: t for i, t in turns.items() if t % 360}

        if not removing and not turning:
            # Nothing to change: an exact copy, over any stale one.
            shutil.copy2(original_pdf, temp)
        else:
            if removing:
                # select() rewrites the document to exactly this page list.
                doc.select(keep)
            position = {original: new for new, original in enumerate(keep)}
            for original, turn in turning.items():
                if original in position:
                    page = doc[position[original]]
                    # Added, not assigned: some PDFs already carry a /Rotate.
                    page.set_rotation((page.rotation + turn) % 360)
            if removing:
                # Removed pages leave unreferenced objects behind; garbage
                # collection is what actually takes them out of the file.
                doc.save(temp, garbage=4, deflate=True)
            else:
                # Turns only: no garbage/deflate, so only the page
                # dictionaries differ from the original.
                doc.save(temp)
    os.replace(temp, fixed_pdf)


def rebuild_fixed_pdf(original_pdf: Path, fixed_pdf: Path, duplicates: list,
                      rotations: list) -> list:
    """Rewrite the fixed PDF from a document's records, after a person
    removes or restores a repeated page.

    Removing a page shifts every page after it, so the whole file is rebuilt
    from the upload with the current removals (``duplicates`` records marked
    ``removed``) and turns (each rotation record's ``current``), and each
    rotation record's ``fixed_idx`` is updated in place -- None for a page
    that is now removed. Returns the kept original page numbers.
    """
    with fitz.open(original_pdf) as doc:
        page_count = doc.page_count
    removed = {d["dupe_idx"] for d in duplicates if d.get("removed")}
    keep = [i for i in range(page_count) if i not in removed]
    # The same guard fix_pdf always had: never a zero-page PDF.
    if not keep:
        keep = list(range(page_count))

    turns = {r["original_idx"]: r["current"] for r in rotations}
    write_fixed_pdf(original_pdf, fixed_pdf, keep, turns)

    position = {original: new for new, original in enumerate(keep)}
    for r in rotations:
        r["fixed_idx"] = position.get(r["original_idx"])
    return keep


def rotation_records(result: Optional[PageFixResult]) -> list:
    """The rotation script's decisions, located in both copies of the PDF.

    ``original_idx`` / ``original_page`` place the page in the uploaded PDF
    (the numbers the user sees when opening their own file); ``fixed_idx``
    places it in the fixed copy, which is shorter by every removed repeat.
    """
    if result is None:
        return []
    position = {original: new for new, original in enumerate(result.kept_indices)}
    records = []
    for d in result.rotations:
        applied = d["confidence"] in _APPLIED_CONFIDENCE
        records.append(dict(
            d,
            applied=applied,
            original_idx=d["page_idx"],
            original_page=d["page_idx"] + 1,
            fixed_idx=position[d["page_idx"]],
            # The turn the fixed PDF has now, on top of the page's own /Rotate.
            # Starts as the script's decision; the user can change it.
            current=d["rotation"] if applied else 0,
            decided=False,
        ))
    return records


VALID_TURNS = (0, 90, 180, 270)


def set_page_rotation(original_pdf: Path, fixed_pdf: Path, original_idx: int,
                      fixed_idx: int, turn: int) -> None:
    """Set one page of the fixed PDF to its original orientation plus ``turn``.

    For correcting the rotation step by hand: undoing a wrong turn, turning a
    page the script was unsure about, or turning one the other way. Measured
    from the uploaded page rather than added to the fixed one, so choosing the
    same option twice is harmless and every option is always reachable.

    Saved incrementally where possible -- only the page dictionary changes,
    exactly as when the rotation script applies a turn itself.
    """
    if turn not in VALID_TURNS:
        raise ValueError("turn must be one of %s" % (VALID_TURNS,))

    with fitz.open(original_pdf) as source:
        base = source[original_idx].rotation

    doc = fitz.open(fixed_pdf)
    try:
        doc[fixed_idx].set_rotation((base + turn) % 360)
        if doc.can_save_incrementally():
            doc.save(fixed_pdf, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
        else:
            # A file PyMuPDF had to repair cannot be appended to; write a
            # fresh copy beside it and swap it in.
            temp = fixed_pdf.with_suffix(".tmp.pdf")
            doc.save(temp)
            doc.close()
            os.replace(temp, fixed_pdf)
    finally:
        if not doc.is_closed:
            doc.close()


def page_fix_fields(result: Optional[PageFixResult]) -> dict:
    """The page-fixes part of a document record.

    Shared by the full pipeline and the page-fixes-only run, so the review
    screen reads the same fields whichever produced the document.
    """
    return {
        "page_count": result.kept_page_count if result else None,
        "original_page_count": result.original_page_count if result else None,
        "duplicates_removed": result.duplicates_removed if result else 0,
        "duplicates_flagged": result.duplicates_flagged if result else 0,
        # Every suspected repeat starts kept; a person decides.
        "duplicates": [dict(d, removed=False, decided=False)
                       for d in (result.duplicates if result else [])],
        "pages_rotated": result.pages_rotated if result else 0,
        "pages_for_review": result.pages_for_review if result else 0,
        "rotations": rotation_records(result),
    }


# --------------------------------------------------------------------------
# Abstract page overrides
# --------------------------------------------------------------------------
# The OCR script accepts ``--override-csv`` naming each PDF's abstract pages
# by hand, for theses whose heading it cannot find or finds in the wrong
# place. People read those page numbers off their own PDF, but the script
# reads the fixed copy, where every removed repeat shifts later pages down
# by one. So the numbers are translated before they are handed over.

_RE_PAGE_RANGE_INPUT = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s*$")


def parse_page_range(value: str) -> Optional[tuple]:
    """``"5-8"`` -> ``(5, 8)``, ``"5"`` -> ``(5, 5)``; blank -> None.

    Raises ValueError on anything else, so a typo is refused up front rather
    than silently ignored and the abstract auto-detected after all.
    """
    if value is None or not str(value).strip():
        return None
    m = _RE_PAGE_RANGE_INPUT.match(str(value))
    if not m:
        raise ValueError("'%s' is not a page range - use a form like 5-8 or 5." % value)
    start = int(m.group(1))
    end = int(m.group(2) or start)
    if start < 1 or end < start:
        raise ValueError("'%s' is not a valid page range." % value)
    return start, end


def map_pages_to_fixed(start: int, end: int, kept_indices: Optional[list]) -> Optional[tuple]:
    """Translate 1-based original page numbers to the fixed copy's.

    The range covers every kept page between ``start`` and ``end``, so a
    removed repeat inside it is simply skipped. Returns None when no page in
    the range survived (all repeats, or past the end of the document).
    """
    if kept_indices is None:
        return start, end
    positions = [
        position
        for position, original in enumerate(kept_indices)
        if start - 1 <= original <= end - 1
    ]
    if not positions:
        return None
    return positions[0] + 1, positions[-1] + 1


def write_override_csv(path: Path, ranges: dict) -> Path:
    """Write ``{filename: (start, end)}`` in the script's own CSV format."""
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "pages"])
        for filename, (start, end) in sorted(ranges.items()):
            writer.writerow([filename, "%d-%d" % (start, end)])
    return path


# --------------------------------------------------------------------------
# Stage 2: the OCR + VLM abstract pass
# --------------------------------------------------------------------------

# Progress markers we look for in the script's stdout. The script was written
# to be watched by a human, not parsed, so these patterns are matched
# defensively: any line that does not match is simply logged and ignored.
_RE_BATCH_INDEX = re.compile(r"\[(\d+)/(\d+)\]")
_RE_PROCESSING = re.compile(r"Processing:\s+(.*\.pdf)\s*$")
_RE_PAGE_RANGE = re.compile(r"Extracting text from pages (\d+) to (\d+)")
_RE_PAGE_DONE = re.compile(r"Processing page (\d+) with PaddleOCR")
_RE_VLM_PHASE = re.compile(r"(VLM review pass|OCR/VLM diff pass)")
_RE_VLM_DOC = re.compile(r"^\s*(.+\.pdf): (reviewing|diffing) (\d+)")

# The three ways the script gives up on a document without writing a draft.
# Each has a genuinely different cause, and telling them apart is the
# difference between a user knowing what to do next and guessing. Mapped to
# plain-language explanations here rather than in the UI, because the wording
# depends on knowing what the script actually does.
_DOCUMENT_PROBLEMS = [
    (re.compile(r"Skipped - could not find abstract"),
     "no abstract heading found in the first 15 pages"),
    (re.compile(r"No text extracted"),
     "OCR read no text on the abstract pages - the OCR engine may be "
     "misconfigured, or the scan may be unreadable at this threshold"),
    (re.compile(r"No paragraph text reconstructed"),
     "text was read but no paragraphs could be rebuilt from it"),
]


def build_abstract_command(
    input_dir: Path,
    out_dir: Path,
    model: str,
    mode: str,
    ollama_url: str,
    confidence_threshold: float = 0.60,
    override_csv: Optional[Path] = None,
) -> list:
    """Assemble the exact argv used to invoke the original OCR script.

    Split out from run_abstract_pass() so it can be logged and checked without
    actually launching a multi-minute OCR run.
    """
    cmd = [
        sys.executable,
        str(VLM_SCRIPT),
        "--input", str(input_dir),
        "--out", str(out_dir),
        "--confidence-threshold", str(confidence_threshold),
        "--vlm-model", model,
        "--ollama-url", ollama_url,
    ]
    if override_csv is not None:
        cmd += ["--override-csv", str(override_csv)]
    cmd += VLM_MODES.get(mode, VLM_MODES[DEFAULT_VLM_MODE])["flags"]
    return cmd


def run_abstract_pass(
    input_dir: Path,
    out_dir: Path,
    model: str = DEFAULT_VLM_MODEL,
    mode: str = DEFAULT_VLM_MODE,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    on_progress: Optional[Callable[[dict], None]] = None,
    override_csv: Optional[Path] = None,
) -> int:
    """Run the original script over every PDF in ``input_dir``.

    A whole directory is handed over in one call, rather than looping over
    files here, precisely because the script's deferred VLM phase is
    batch-wide: it waits until all PaddleOCR work is done before making any
    Ollama call. Calling it once per file would defeat that and reintroduce
    the GPU-contention bug its author designed around.

    ``on_progress`` is called with small dicts as milestones are parsed out of
    stdout. Returns the script's exit code.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Unbuffered + UTF-8 so we see each line as it happens rather than in one
    # burst at the end, and so the script's arrows/checkmarks survive the pipe.
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = build_abstract_command(input_dir, out_dir, model, mode, ollama_url,
                                 override_csv=override_csv)

    def emit(**kwargs) -> None:
        if on_progress is not None:
            on_progress(kwargs)

    emit(event="command", detail=" ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # one interleaved stream is enough for a log
        env=env,
        encoding="utf-8",
        errors="replace",
        bufsize=1,  # line buffered
    )

    page_from = None
    page_to = None
    if proc.stdout is not None:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            emit(event="log", detail=line)

            m = _RE_BATCH_INDEX.search(line)
            if m:
                emit(event="file_index", index=int(m.group(1)), total=int(m.group(2)))

            m = _RE_PROCESSING.search(line)
            if m:
                page_from = None
                page_to = None
                emit(event="file_start", filename=m.group(1))

            m = _RE_PAGE_RANGE.search(line)
            if m:
                page_from, page_to = int(m.group(1)), int(m.group(2))
                emit(event="page_total", total=page_to - page_from + 1)

            m = _RE_PAGE_DONE.search(line)
            if m and page_from is not None:
                # The script prints 1-based page numbers of the *document*; the
                # UI wants "N of M" within the abstract range being processed.
                current = int(m.group(1)) - page_from + 1
                total = (page_to - page_from + 1) if page_to is not None else None
                emit(event="page_done", current=current, total=total)

            for pattern, explanation in _DOCUMENT_PROBLEMS:
                if pattern.search(line):
                    # Attributed to whichever file the script last announced;
                    # it processes strictly one document at a time, so that is
                    # unambiguous.
                    emit(event="file_problem", reason=explanation)
                    break

            if _RE_VLM_PHASE.search(line):
                emit(event="vlm_phase", detail=line.strip())

            m = _RE_VLM_DOC.search(line)
            if m:
                emit(event="vlm_doc", filename=m.group(1),
                     verb=m.group(2), pages=int(m.group(3)))

    proc.wait()
    emit(event="exit", code=proc.returncode)
    return proc.returncode


# --------------------------------------------------------------------------
# Stage 3: reading the script's output back
# --------------------------------------------------------------------------

# The script writes "<stem> draft.txt" -- an HTML fragment, not Markdown.
# Everything below understands that shape.
DRAFT_SUFFIX = " draft.txt"

# Appended review blocks the script may add after the primary text. Both start
# at the first "\n\n<!--", which is also exactly where the script itself splits
# when it re-appends (see _apply_vlm_diff_merges), so this split is safe.
_APPENDED_MARKER = "\n\n<!--"

_RE_RECOVERY_BLOCK = re.compile(
    r"<!-- VLM RECOVERY -- page (\d+) (.*?) -->\n(.*?)(?=\n\n<!--|\Z)",
    re.DOTALL,
)
_RE_DIFF_BLOCK = re.compile(
    r"<!-- OCR/VLM DIFF REPORT -- page (\d+)[^>]*-->\n(.*?)(?=\n\n<!--|\Z)",
    re.DOTALL,
)
_RE_DIFF_LINE = re.compile(
    r'^\s*(?:\[(?P<label>[A-Z ]+)\]\s+)?OCR: "(?P<ocr>.*?)"\s+\|\s+VLM: "(?P<vlm>.*)"\s*$'
)
# Equation placeholders the OCR pass leaves inline when it meets a stacked
# fraction it cannot read. A real, script-produced signal -- not a guess.
_RE_EQUATION_PLACEHOLDER = re.compile(
    r"\[EQUATION(?:\s+\d+)? - VERIFY MANUALLY, PAGE (\d+)\]"
)


def parse_draft(draft_text: str) -> dict:
    """Split one draft file into its primary text and its review artifacts.

    Returns a dict with:
      ``primary``   -- the HTML paragraphs that are the actual output
      ``recovery``  -- VLM re-reads of flagged pages, if the run produced any
      ``diffs``     -- per-page OCR-vs-VLM spans, if the run produced any
      ``flags``     -- per-page review flags, assembled from the above

    Important: there is no numeric per-page confidence score anywhere in this
    output. The OCR script writes its confidence sidecar and then deletes it
    at the end of its own run, so the only per-page signal that survives into
    the draft is *categorical* -- a page was flagged, and for which reason.
    ``flags`` therefore carries reasons, never invented numbers.
    """
    split_at = draft_text.find(_APPENDED_MARKER)
    if split_at == -1:
        primary, appended = draft_text, ""
    else:
        primary, appended = draft_text[:split_at], draft_text[split_at:]

    recovery = [
        {"page": int(page), "reason": reason.strip(), "html": body.strip()}
        for page, reason, body in _RE_RECOVERY_BLOCK.findall(appended)
    ]

    diffs = []
    for page, body in _RE_DIFF_BLOCK.findall(appended):
        spans = []
        for line in body.splitlines():
            m = _RE_DIFF_LINE.match(line)
            if not m:
                continue
            spans.append({
                "label": (m.group("label") or "").strip() or None,
                "ocr": m.group("ocr"),
                "vlm": m.group("vlm"),
            })
        if spans:
            diffs.append({"page": int(page), "spans": spans})

    # --- Assemble per-page flags from the real signals above ----------------
    flags = {}

    def add_flag(page, kind, label):
        # A page can earn several flags; keep the first kind set but let the
        # detail lines accumulate so the sidebar can show everything at once.
        entry = flags.setdefault(page, {"page": page, "kind": kind, "labels": []})
        if label not in entry["labels"]:
            entry["labels"].append(label)

    for block in recovery:
        # "flagged as low-confidence" / "flagged for equation content (...)"
        kind = "equation" if "equation" in block["reason"] else "low_confidence"
        add_flag(block["page"], kind, block["reason"])

    for page in {int(p) for p in _RE_EQUATION_PLACEHOLDER.findall(primary)}:
        add_flag(page, "equation", "contains an unread equation placeholder")

    for block in diffs:
        unresolved = [s for s in block["spans"] if s["label"] in (None, "FLAGGED")]
        if unresolved:
            add_flag(block["page"], "diff",
                     "%d unresolved OCR/VLM difference(s)" % len(unresolved))

    return {
        "primary": primary.strip(),
        "recovery": recovery,
        "diffs": diffs,
        "flags": [flags[p] for p in sorted(flags)],
    }


# --------------------------------------------------------------------------
# Applying a single diff span to the draft text
# --------------------------------------------------------------------------
# The review screen lets the user apply any span the script left FLAGGED with
# one click, instead of hunting for the text in the editor.
#
# The hard part is that a diff span is *raw* text (straight from OCR or from
# the model) while the draft has already been through the script's fixup
# chain -- a span reading "ε = 0.05" appears in the draft as "&epsilon; = 0.05".
# A plain find-and-replace would therefore silently fail on exactly the spans
# most likely to still be flagged. So we reuse the script's own
# _normalize_for_primary_text(), which is the function it uses for this same
# purpose in _apply_vlm_diff_merges(). Reusing it means the UI can never drift
# out of step with how the script itself locates text.

# The literal the script prints for "this side of the diff has no text".
NOTHING_SPAN = "(nothing)"

_normalizer = None


def _get_normalizer():
    """Import the original script's text normalizer, once, on first use.

    Deferred rather than imported at module scope because importing
    vlm_abstract.py pulls in PaddleOCR, which costs about nine seconds. The
    server warms this in a background thread at startup (see server/__init__)
    so the first click does not pay that cost.
    """
    global _normalizer
    if _normalizer is None:
        from pipeline.vlm_abstract import _normalize_for_primary_text
        _normalizer = _normalize_for_primary_text
    return _normalizer


# How much text either side of a span we remember in order to recognise it
# again. Enough to be distinctive in an abstract; short enough that an edit
# nearby does not invalidate it.
_CONTEXT_CHARS = 24

# How many characters of surrounding text have to agree before we will accept
# a candidate as this span's own. A handful of characters can line up by pure
# chance -- a space, a closing tag -- so a winner below this is treated as no
# winner at all.
_MIN_CONTEXT_MATCH = 6


def _common_prefix_len(a: str, b: str) -> int:
    """How many characters ``a`` and ``b`` share from the start."""
    limit = min(len(a), len(b))
    n = 0
    while n < limit and a[n] == b[n]:
        n += 1
    return n


def _common_suffix_len(a: str, b: str) -> int:
    """How many characters ``a`` and ``b`` share from the end."""
    limit = min(len(a), len(b))
    n = 0
    while n < limit and a[len(a) - 1 - n] == b[len(b) - 1 - n]:
        n += 1
    return n


def _offsets_of(text: str, needle: str) -> list:
    """Every position at which ``needle`` occurs in ``text``."""
    positions = []
    at = text.find(needle)
    while at != -1:
        positions.append(at)
        at = text.find(needle, at + 1)
    return positions


# Tokens this short are found inside ordinary text everywhere, so a plain
# substring search says nothing about whether *this* span is still there.
_SHORT_TOKEN = 2
_WORD_CHAR = r"A-Za-z0-9"


def _span_offsets(text: str, needle: str) -> list:
    """Where a diff span's text sits in the document, as the span means it.

    The diff report splits on whitespace, so a span is a whole token. For
    most spans a substring search finds it well enough, but two kinds match
    all over the text and would make a resolved span look unresolved:

    * Punctuation alone (a stray "." the OCR read twice): only a mark not
      attached to a word counts -- "Knoxville..", "word ." -- never the
      ordinary full stop at the end of every sentence.
    * One or two letters (a stray "y"): only as a word on its own, not
      inside other words.
    """
    if not needle:
        return []
    if not re.search("[%s]" % _WORD_CHAR, needle):
        # Not after a word, a closing bracket or quote, or a closing tag --
        # "10<sup>3</sup>." ends a sentence like any other.
        pattern = r"(?<![%s)\]\"'%%>])%s" % (_WORD_CHAR, re.escape(needle))
    elif len(needle) <= _SHORT_TOKEN:
        pattern = r"(?<![%s])%s(?![%s])" % (_WORD_CHAR, re.escape(needle), _WORD_CHAR)
    else:
        return _offsets_of(text, needle)
    return [m.start() for m in re.finditer(pattern, text)]


def _span_present(text: str, needle: str) -> bool:
    return bool(_span_offsets(text, needle))


def make_hint(text: str, offset: int, length: int) -> dict:
    """Record where a span sits, and what sits either side of it.

    Position alone is not enough to find a span again: any edit earlier in the
    document shifts it. The surrounding words move with the span, so they
    identify it even after the text around it has changed.
    """
    return {
        "offset": offset,
        "before": text[max(0, offset - _CONTEXT_CHARS):offset],
        "after": text[offset + length:offset + length + _CONTEXT_CHARS],
    }


def _resolve_offset(text: str, needle: str, hint):
    """Find *this span's own* copy of ``needle``, not just any copy.

    This is what makes a swap reversible. Applying a correction can itself
    make the words non-unique -- changing "PAo" to "PAO" in a document that
    already says "PAO" somewhere else -- and from then on a plain text search
    cannot tell which occurrence belongs to this diff. The remembered position
    and surrounding words can.

    Returns ``(offset, status)`` with status "ok", "not_found" or "ambiguous".
    """
    positions = _span_offsets(text, needle)
    if not positions:
        return -1, "not_found"
    if len(positions) == 1:
        return positions[0], "ok"

    if not hint:
        return -1, "ambiguous"

    # Still exactly where we left it.
    offset = hint.get("offset")
    if offset in positions:
        return offset, "ok"

    # Moved, because of an edit earlier in the document. The words either side
    # of the span moved with it, so they still identify it -- but only
    # partially: an insertion just before the span truncates what is left of
    # the remembered text on that side. So each candidate is scored by how
    # much of its surroundings still agree, rather than demanding an exact
    # match on both sides.
    before = hint.get("before", "")
    after = hint.get("after", "")
    scored = sorted(
        (
            _common_suffix_len(text[:p], before)
            + _common_prefix_len(text[p + len(needle):], after),
            p,
        )
        for p in positions
    )
    best_score, best_position = scored[-1]
    runner_up = scored[-2][0]

    # A clear winner, with enough agreement to be more than coincidence.
    if best_score >= _MIN_CONTEXT_MATCH and best_score > runner_up:
        return best_position, "ok"

    return -1, "ambiguous"


def span_state(text: str, ocr_span: str, vlm_span: str, hint=None) -> str:
    """Which side of this diff the document currently reflects.

    Worked out from the text itself rather than stored as a flag, so it stays
    right even after the user edits the textarea by hand. ``hint`` is the
    span's last known position, which settles cases the text alone cannot --
    notably when one reading contains the other ("Her" inside
    "Her<sup>-</sup>"), where both would otherwise appear present.

    Returns "ocr", "vlm", or "unclear" when neither reading can be located.
    """
    normalize = _get_normalizer()

    # With a record of where this span sits, answer by looking at that spot
    # specifically. Note that "the other reading exists somewhere in the
    # document" is NOT the question -- the same words often appear elsewhere,
    # and treating that as an answer is what used to make a span look already
    # applied when it was not. So every occurrence of both readings is scored
    # on how well its surroundings match what we remember, and the best-placed
    # one wins.
    if hint:
        before = hint.get("before", "")
        after = hint.get("after", "")
        best = None  # (context score, length of match, side)

        for side, raw in (("ocr", ocr_span), ("vlm", vlm_span)):
            if raw == NOTHING_SPAN:
                continue
            needle = normalize(raw)
            for position in _span_offsets(text, needle):
                score = (
                    _common_suffix_len(text[:position], before)
                    + _common_prefix_len(text[position + len(needle):], after)
                )
                # Length breaks a tie so that a reading which contains the
                # other ("Her" inside "Her<sup>-</sup>") is not mistaken for it.
                candidate = (score, len(needle), side)
                if best is None or candidate > best:
                    best = candidate

        if best is not None and best[0] >= _MIN_CONTEXT_MATCH:
            return best[2]

        # Neither reading sits where this span belongs. For a one-sided span
        # that is exactly what "the empty side was applied" looks like.
        if vlm_span == NOTHING_SPAN:
            return "vlm"
        if ocr_span == NOTHING_SPAN:
            return "ocr"

    # No usable position -- fall back to plain presence.
    if vlm_span == NOTHING_SPAN:
        return "ocr" if _span_present(text, normalize(ocr_span)) else "vlm"
    if ocr_span == NOTHING_SPAN:
        return "vlm" if _span_present(text, normalize(vlm_span)) else "ocr"

    ocr_needle = normalize(ocr_span)
    vlm_needle = normalize(vlm_span)
    ocr_present = _span_present(text, ocr_needle)
    vlm_present = _span_present(text, vlm_needle)
    if vlm_present and not ocr_present:
        return "vlm"
    if ocr_present and not vlm_present:
        return "ocr"
    if ocr_present and vlm_present:
        # One reading contains the other, so both "match". The longer one is
        # the specific one, and therefore what the document actually shows.
        if vlm_needle != ocr_needle:
            return "vlm" if len(vlm_needle) > len(ocr_needle) else "ocr"
    return "unclear"


def _restore_point(text: str, hint):
    """Where text that was deleted should go back.

    The span left no words behind to search for, so the only record of where
    it belonged is what sat either side of it. Those neighbours are looked up
    first, which keeps the restore correct even if the document has shifted
    since; the remembered position is the fallback.
    """
    if not hint:
        return None

    before = hint.get("before", "")
    after = hint.get("after", "")

    # The gap is wherever the two remembered sides now meet. Scored the same
    # way as _resolve_offset, so a partial match still counts: an edit
    # elsewhere may have eaten into one side of the remembered context.
    if before and after:
        joins = _offsets_of(text, before + after)
        if len(joins) == 1:
            return joins[0] + len(before)

        tail = before[-8:]
        candidates = _offsets_of(text, tail)
        if candidates:
            scored = sorted(
                (_common_prefix_len(text[p + len(tail):], after), p)
                for p in candidates
            )
            best_score, best_position = scored[-1]
            runner_up = scored[-2][0] if len(scored) > 1 else -1
            if best_score >= _MIN_CONTEXT_MATCH and best_score > runner_up:
                return best_position + len(tail)

    offset = hint.get("offset")
    if offset is not None and 0 <= offset <= len(text):
        return offset
    return None


def apply_span(text: str, ocr_span: str, vlm_span: str, direction: str, hint=None,
               at_offset=None):
    """Rewrite one diff span in ``text`` to the chosen side.

    ``direction`` is "vlm" to take the model's reading or "ocr" to put the
    OCR's reading back. ``hint`` is the record of where this span was last
    written, which is what makes the swap reliably reversible -- see
    _resolve_offset(). ``at_offset`` is a place the user picked from
    span_occurrences() for text that appears more than once; it settles the
    ambiguity in their words rather than ours.

    Returns ``(new_text, status, hint)``. The returned hint describes where
    the span's text now sits and should be passed back in next time; on a
    refusal the incoming hint is handed back unchanged. Status is one of:

      "applied"     -- the swap was made
      "unchanged"   -- that side is already what the document says
      "not_found"   -- the text to replace is not in the document
      "ambiguous"   -- it occurs in several places and we have no record of
                       which one is this span's, so replacing would be a guess
      "unplaceable" -- there is nothing to match against and no remembered
                       position, so there is nowhere to put the text
      "moved"       -- ``at_offset`` no longer points at the text, because the
                       document changed after the places were listed

    Refusing an ambiguous match is the safety property that matters here:
    rewriting the wrong occurrence would corrupt the document somewhere the
    user is not looking.
    """
    normalize = _get_normalizer()

    if direction == "vlm":
        find_raw, put_raw = ocr_span, vlm_span
    else:
        find_raw, put_raw = vlm_span, ocr_span

    replacement = "" if put_raw == NOTHING_SPAN else normalize(put_raw)

    # Restoring something that was deleted: there is no text to search for,
    # but if we remember where it was taken from we can put it back exactly
    # there. Without that record there is no honest place to insert it.
    if find_raw == NOTHING_SPAN:
        gap = _restore_point(text, hint)
        if gap is None or not replacement:
            return text, "unplaceable", None

        # If the gap sits squarely between two tags, this span was a whole
        # paragraph of its own -- a heading, typically -- and its <p> wrapper
        # went with it when it was deleted. Put the wrapper back too, or the
        # restored words would end up loose between paragraphs.
        before_char = text[gap - 1] if gap > 0 else ""
        after_char = text[gap] if gap < len(text) else ""
        if before_char in ("", ">") and after_char in ("", "<"):
            insertion = "<p>" + replacement + "</p>"
            words_at = gap + len("<p>")
        else:
            # Deleting the span also closed up the whitespace around it, so
            # restoring has to reopen it -- otherwise the words fuse onto
            # their neighbours ("noisemore").
            lead = "" if before_char in ("", " ", ">") else " "
            trail = "" if (after_char in ("", " ", "<") or after_char in ".,;:") else " "
            insertion = lead + replacement + trail
            words_at = gap + len(lead)

        new_text = text[:gap] + insertion + text[gap:]
        return new_text, "applied", make_hint(new_text, words_at, len(replacement))

    needle = normalize(find_raw)
    if at_offset is not None:
        # Only honoured while the text is still exactly where it was listed.
        # Anywhere else would be replacing something the user did not pick.
        if needle and 0 <= at_offset and text[at_offset:at_offset + len(needle)] == needle:
            offset, status = at_offset, "ok"
        else:
            return text, "moved", hint
    else:
        offset, status = _resolve_offset(text, needle, hint)

    if status == "not_found":
        # Already showing the other side is the common, harmless case.
        if replacement and _span_present(text, replacement):
            return text, "unchanged", hint
        return text, "not_found", hint
    if status == "ambiguous":
        return text, "ambiguous", hint

    new_text = text[:offset] + replacement + text[offset + len(needle):]

    # Deleting a span leaves debris behind: a doubled space where the words
    # were, a space stranded before punctuation, and -- when the span was a
    # whole paragraph, such as a heading the model was told to omit -- an
    # empty <p></p>. Paragraph text in these drafts is single-spaced, so these
    # cleanups are safe and keep the result looking like the rest of the file.
    #
    # The tidy-up works outwards from the gap the deletion left, so the
    # position handed back still points at the span instead of drifting by
    # however much whitespace got collapsed.
    if not replacement:
        prefix = re.sub(r"[ \t]+$", "", new_text[:offset])
        suffix = re.sub(r"^[ \t]+", "", new_text[offset:])

        # Put a single space back only where real words now sit on both
        # sides; not against a tag boundary, and not before punctuation.
        needs_space = bool(
            prefix and suffix
            and not prefix.endswith(">")
            and not suffix.startswith("<")
            and suffix[0] not in ".,;:"
        )
        joiner = " " if needs_space else ""
        new_text = prefix + joiner + suffix
        offset = len(prefix) + len(joiner)

        # If the span was a whole paragraph -- a heading the model was told to
        # omit -- the now-empty <p></p> goes too. Searched in a tight window
        # around the edit so an empty paragraph elsewhere in the document is
        # never mistaken for this one.
        emptied = re.compile(r"<p>\s*</p>").search(
            new_text, max(0, offset - 8), min(len(new_text), offset + 8))
        if emptied and emptied.start() <= offset <= emptied.end():
            new_text = new_text[:emptied.start()] + new_text[emptied.end():]
            offset = emptied.start()

    return new_text, "applied", make_hint(new_text, offset, len(replacement))


# --------------------------------------------------------------------------
# Choosing among repeated text
# --------------------------------------------------------------------------
# The diff report names each difference only by its words, never by where it
# is. When those words appear more than once, apply_span() cannot know which
# copy is meant, so the user picks -- from each copy shown in its surrounding
# words, with a best guess marked.

# How much surrounding text to show either side of each copy.
_CHOICE_CONTEXT_CHARS = 40

_RE_TAG = re.compile(r"<[^>]*>")


def _readable(fragment: str) -> str:
    """HTML fragment -> plain words, for display only."""
    return re.sub(r"\s+", " ", html.unescape(_RE_TAG.sub(" ", fragment)))


def span_choices(text: str, ocr_span: str, vlm_span: str, state: str, hint=None):
    """The copies a click on this span would have to choose between.

    ``state`` is span_state()'s answer. The click switches to the other side,
    so the text being replaced is the side the document shows now. Returns
    ``(direction, occurrences)`` when that text appears more than once and
    the span's remembered position does not settle which copy is its own;
    otherwise None. Each occurrence is ``{offset, before, match, after}``,
    where ``offset`` is the raw position to send back and the rest is
    readable text.
    """
    if state not in ("ocr", "vlm"):
        return None
    raw = ocr_span if state == "ocr" else vlm_span
    if raw == NOTHING_SPAN:
        return None

    needle = _get_normalizer()(raw)
    positions = _span_offsets(text, needle)
    if len(positions) < 2 or _resolve_offset(text, needle, hint)[1] != "ambiguous":
        return None

    reach = _CHOICE_CONTEXT_CHARS * 2  # raw text includes tags, so read wider
    occurrences = []
    for position in positions:
        end = position + len(needle)
        before = text[max(0, position - reach):position]
        after = text[end:end + reach]
        # Drop a tag cut in half by the window edge.
        if ">" in before and "<" not in before[:before.index(">")]:
            before = before[before.index(">") + 1:]
        if "<" in after and ">" not in after[after.rindex("<"):]:
            after = after[:after.rindex("<")]
        before = _readable(before)
        after = _readable(after)
        # Trimmed to whole words, so the context never starts or ends mid-word.
        if len(before) > _CHOICE_CONTEXT_CHARS:
            before = before[-_CHOICE_CONTEXT_CHARS:]
            before = before[before.find(" ") + 1:] if " " in before else before
        if len(after) > _CHOICE_CONTEXT_CHARS:
            after = after[:_CHOICE_CONTEXT_CHARS]
            after = after[:after.rfind(" ")] if " " in after else after
        occurrences.append({
            "offset": position,
            "before": before.strip(),
            "match": _readable(needle).strip(),
            "after": after.strip(),
        })
    direction = "vlm" if state == "ocr" else "ocr"
    return direction, occurrences


def _picked_side(hint) -> Optional[str]:
    """The side whose repeated text the user picked a copy of, if they did."""
    return hint.get("picked_side") if hint else None


def span_choice_view(text: str, ocr_span: str, vlm_span: str, state: str, hint=None):
    """What the copy picker for this span should show, or None for no picker.

    Two situations get a picker:

    * Not yet picked: the text appears more than once and nothing says which
      copy is this span's -- every copy is listed, none selected.
    * Already picked: the list stays available so a wrong pick can be moved.
      The copies are listed as the document reads with the pick undone, since
      that is the text a new pick applies to, and ``selected`` marks the copy
      currently changed.

    Returns ``{choose_direction, occurrences, selected}``.
    """
    picked = _picked_side(hint)
    if picked and state in ("ocr", "vlm") and state != picked:
        base, status, base_hint = apply_span(text, ocr_span, vlm_span, picked, hint)
        if status != "applied":
            return None
        found = span_choices(base, ocr_span, vlm_span, picked)
        if found is None:
            return None
        direction, occurrences = found
        selected = next((i for i, o in enumerate(occurrences)
                         if o["offset"] == base_hint["offset"]), None)
        return {"choose_direction": direction, "occurrences": occurrences, "selected": selected}

    # A span picked before and since put back lists its copies afresh; its
    # remembered position must not hide the picker.
    found = span_choices(text, ocr_span, vlm_span, state, None if picked else hint)
    if found is None:
        return None
    direction, occurrences = found
    return {"choose_direction": direction, "occurrences": occurrences, "selected": None}


def apply_span_choice(text: str, ocr_span: str, vlm_span: str, direction: str,
                      hint, at_offset: int):
    """Apply a span at the copy the user picked, moving an earlier pick.

    ``at_offset`` is a position from span_choice_view(). If this span already
    has a pick applied, that pick is undone first -- the listed positions
    describe the text with it undone -- and the new copy changed instead, so
    correcting a wrong pick is one click. On any refusal the text is handed
    back exactly as it came in.

    Returns ``(new_text, status, hint)``; the hint records the picked side,
    which is what keeps the picker available afterwards.
    """
    picked = _picked_side(hint)
    base = text
    if picked and span_state(text, ocr_span, vlm_span, hint) not in (picked, "unclear"):
        base, status, _ = apply_span(text, ocr_span, vlm_span, picked, hint)
        if status != "applied":
            return text, "moved", hint

    new_text, status, new_hint = apply_span(base, ocr_span, vlm_span, direction, None,
                                            at_offset=at_offset)
    if status != "applied":
        return text, status, hint
    replaced = "ocr" if direction == "vlm" else "vlm"
    return new_text, "applied", dict(new_hint, picked_side=replaced)


def suggest_occurrences(text: str, spans: list) -> list:
    """A best guess at where each span sits, for marking one choice "likely".

    ``spans`` is ``[(ocr_span, vlm_span, state), ...]`` in report order. The
    report lists differences page by page, top to bottom, which is also the
    order of the text -- so walking both together, each span most likely sits
    at the first copy of its text after the previous span's. Returns one
    offset (or None) per span. A guess, never applied on its own.
    """
    normalize = _get_normalizer()
    cursor = 0
    guesses = []
    for ocr_span, vlm_span, state in spans:
        raw = ocr_span if state == "ocr" else vlm_span if state == "vlm" else NOTHING_SPAN
        needle = normalize(raw) if raw != NOTHING_SPAN else ""
        at = next((o for o in _span_offsets(text, needle) if o >= cursor), -1)
        if at == -1:
            guesses.append(None)
            continue
        guesses.append(at)
        cursor = at + len(needle)
    return guesses


def collect_outputs(
    draft_dir: Path,
    output_dir: Path,
    fix_results: dict,
    model_used: str,
    mode_used: str,
    abstract_pages: Optional[dict] = None,
) -> list:
    """Turn the script's "<stem> draft.txt" files into the app's own outputs.

    Writes, per document:
      ``<stem>.md``         -- the draft's primary text, verbatim
      ``<stem>.pages.json`` -- per-page review flags, only if any exist

    The ``.md`` extension with HTML inside it is intentional and was agreed up
    front: the OCR script's output is an HTML fragment whose <sup>/<sub> tags
    and character entities are the entire point of the pipeline. Converting it
    to Markdown would destroy that, so the file keeps the HTML payload
    byte-for-byte and the UI's Markdown renderer is configured with
    ``html: true``, which passes inline HTML straight through.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = []

    for draft_path in sorted(draft_dir.glob("*" + DRAFT_SUFFIX)):
        stem = draft_path.name[: -len(DRAFT_SUFFIX)]
        parsed = parse_draft(draft_path.read_text(encoding="utf-8"))

        md_path = output_dir / (stem + ".md")
        md_path.write_text(parsed["primary"], encoding="utf-8")

        # Only write the sidecar when there is something real in it -- an empty
        # file would imply "checked, all clean" for runs where the chosen mode
        # never produced flags at all.
        pages_path = output_dir / (stem + ".pages.json")
        if parsed["flags"]:
            pages_path.write_text(
                json.dumps({"source": stem, "flags": parsed["flags"]}, indent=2),
                encoding="utf-8",
            )
        elif pages_path.exists():
            pages_path.unlink()

        filename = stem + ".pdf"
        documents.append(dict(
            {
                "name": stem,
                "md_file": md_path.name,
                "filename": filename,
                "fixes_only": False,
                "fixes_recorded": True,
                "abstract_pages": (abstract_pages or {}).get(filename),
                "model_used": model_used,
                "mode_used": mode_used,
                "processed_at": datetime.now().isoformat(timespec="seconds"),
                "flags": parsed["flags"],
                "diffs": parsed["diffs"],
                "recovery": parsed["recovery"],
            },
            **page_fix_fields(fix_results.get(filename)),
        ))

    return documents


def rebuild_documents(job_folder: Path) -> list:
    """Document records read back from a run folder that has no saved state.

    For a folder written before the app saved ``job.json``, or one assembled
    by hand. What the files still hold is recovered: the draft gives the
    flags, the OCR/VLM differences and the VLM recovery blocks (through the
    same parse_draft() the pipeline uses), and ``output/<stem>.md`` gives the
    text as last saved.

    Two things cannot come back, because nothing on disk records them: which
    pages were flagged as repeats, and which were turned. Those lists are
    empty and ``fixes_recorded`` is False, so the review screen can say the
    folder has no record of them rather than implying none were found.

    Unlike collect_outputs() this writes nothing -- in particular it must not
    overwrite an ``.md`` that holds someone's saved edits.
    """
    drafts_dir = job_folder / DRAFTS_DIRNAME
    output_dir = job_folder / OUTPUT_DIRNAME
    uploads_dir = job_folder / UPLOADS_DIRNAME

    documents = []
    for draft_path in sorted(drafts_dir.glob("*" + DRAFT_SUFFIX)):
        stem = draft_path.name[: -len(DRAFT_SUFFIX)]
        parsed = parse_draft(draft_path.read_text(encoding="utf-8"))

        # The .md is the text as last saved. A folder that was never saved
        # from the review screen has none, so the draft's own text is written
        # out once -- that is creating what is missing, never overwriting
        # someone's saved edits.
        md_path = output_dir / (stem + ".md")
        if not md_path.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            md_path.write_text(parsed["primary"], encoding="utf-8")

        documents.append(dict(
            {
                "name": stem,
                "md_file": md_path.name,
                "filename": stem + ".pdf",
                "fixes_only": False,
                "fixes_recorded": False,
                "steps": None,
                "abstract_pages": None,
                "model_used": None,
                "mode_used": None,
                "processed_at": None,
                "flags": parsed["flags"],
                "diffs": parsed["diffs"],
                "recovery": parsed["recovery"],
            },
            **page_fix_fields(None),
        ))
    # A page-fixes-only run has no drafts at all: its documents are the PDFs.
    if not documents and uploads_dir.is_dir():
        for pdf in sorted(uploads_dir.glob("*.pdf")):
            documents.append(dict(
                {
                    "name": pdf.stem,
                    "md_file": None,
                    "filename": pdf.name,
                    "fixes_only": True,
                    "fixes_recorded": False,
                    "steps": None,
                    "abstract_pages": None,
                    "model_used": None,
                    "mode_used": None,
                    "processed_at": None,
                    "flags": [],
                    "diffs": [],
                    "recovery": [],
                },
                **page_fix_fields(None),
            ))
    return documents


def collect_fix_outputs(pdfs: list, fix_results: dict) -> list:
    """Document records for a page-fixes-only run: no OCR, so no text.

    ``md_file`` is None and the text fields are empty, which is how the
    routes and the review screen tell these apart. The product of such a run
    is the corrected PDF in ``fixed/`` itself.
    """
    processed_at = datetime.now().isoformat(timespec="seconds")
    return [
        dict(
            {
                "name": pdf.stem,
                "md_file": None,
                "filename": pdf.name,
                "fixes_only": True,
                "fixes_recorded": True,
                "abstract_pages": None,
                "model_used": None,
                "mode_used": None,
                "processed_at": processed_at,
                "flags": [],
                "diffs": [],
                "recovery": [],
            },
            **page_fix_fields(fix_results.get(pdf.name)),
        )
        for pdf in pdfs
    ]


# --------------------------------------------------------------------------
# The whole pipeline, start to finish
# --------------------------------------------------------------------------

# Subdirectory names inside workdir/<job_id>/. Named constants because both
# this module and the export route need to agree on them.
UPLOADS_DIRNAME = "uploads"
FIXED_DIRNAME = "fixed"
DRAFTS_DIRNAME = "drafts"
OUTPUT_DIRNAME = "output"


# The steps a run can include, all on unless the person running it says not.
STEP_NAMES = ("dedupe", "rotate", "ocr")


def normalize_steps(steps: Optional[dict]) -> dict:
    """``{"dedupe", "rotate", "ocr"}`` as booleans; missing ones default on.

    Unknown keys are dropped, so the dict can be stored and sent back to the
    UI as it is.
    """
    steps = steps or {}
    return {name: bool(steps.get(name, True)) for name in STEP_NAMES}


def run_pipeline(
    workdir: Path,
    model: str,
    mode: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    report: Optional[Callable[..., None]] = None,
    overrides: Optional[dict] = None,
    steps: Optional[dict] = None,
) -> dict:
    """Run the chosen steps over every uploaded PDF: the page fixes (repeats
    flagged, sideways pages turned), then one batch OCR/VLM pass, then collect
    the results.

    ``steps`` is ``{"dedupe", "rotate", "ocr"}`` booleans -- see
    normalize_steps() -- so a batch runs only what it needs. Without OCR the
    run stops after the page fixes, for theses with no abstract to read.
    ``overrides`` maps an uploaded filename to its abstract pages as
    ``(start, end)``, numbered as in the uploaded PDF.

    ``report`` is a plain callback taking keyword arguments. Passing a
    callback -- rather than having this module import the job store -- keeps
    the dependency arrow pointing one way: ``server`` knows about
    ``pipeline``, never the reverse. That is also what makes this function
    runnable from a plain script or a test with no Flask involved at all.

    Note the shape of the run: *all* per-file work happens first, and then the
    OCR script is invoked exactly once for the whole batch. That ordering is
    not incidental -- see run_abstract_pass() for why the batch must stay
    whole.
    """
    def emit(**kwargs):
        if report is not None:
            report(**kwargs)

    uploads_dir = workdir / UPLOADS_DIRNAME
    fixed_dir = workdir / FIXED_DIRNAME
    drafts_dir = workdir / DRAFTS_DIRNAME
    output_dir = workdir / OUTPUT_DIRNAME

    pdfs = sorted(uploads_dir.glob("*.pdf"))
    if not pdfs:
        raise ValueError("No PDFs were uploaded for this job.")

    steps = normalize_steps(steps)

    def finish(documents: list, exit_code: int) -> dict:
        # Recorded on each document so the review screen can tell a check
        # that found nothing from one that was never run.
        for d in documents:
            d["steps"] = dict(steps)
        write_manifest(output_dir, documents)
        return {"documents": documents, "exit_code": exit_code}

    # --- Pass 1: page fixes, one file at a time, one write each -----------
    # OCR reads the fixed copies; with neither page fix chosen it reads the
    # uploads as they are, and no copy of each PDF is made at all.
    fix_results = {}
    ocr_input = uploads_dir
    if steps["dedupe"] or steps["rotate"]:
        doing = " and ".join(
            what for what, on in (("repeated", steps["dedupe"]), ("sideways", steps["rotate"])) if on)
        emit(event="stage", stage="fixes", detail="Checking for %s pages" % doing)
        classifier = load_orientation_classifier() if steps["rotate"] else None
        for index, pdf in enumerate(pdfs, 1):
            emit(event="fix_start", index=index, total=len(pdfs), filename=pdf.name,
                 detail="Checking for %s pages" % doing)
            result = fix_pdf(pdf, fixed_dir, classifier=classifier,
                             dedupe=steps["dedupe"], rotate=steps["rotate"])
            fix_results[pdf.name] = result
            emit(event="fix_done", filename=pdf.name,
                 repeats=result.duplicates_flagged,
                 rotated=result.pages_rotated,
                 review=result.pages_for_review)
        ocr_input = fixed_dir

    if not steps["ocr"]:
        emit(event="stage", stage="collecting", detail="Collecting results")
        output_dir.mkdir(parents=True, exist_ok=True)
        return finish(collect_fix_outputs(pdfs, fix_results), 0)

    # --- Abstract page overrides, translated to the fixed copy ------------
    override_csv = None
    abstract_pages = {}
    csv_ranges = {}
    for pdf in pdfs:
        requested = (overrides or {}).get(pdf.name)
        if not requested:
            continue
        start, end = requested
        fixed = fix_results.get(pdf.name)
        mapped = map_pages_to_fixed(start, end, fixed.kept_indices if fixed else None)
        if mapped is None:
            emit(event="log", detail="[override] %s: pages %d-%d were all removed or "
                 "past the end - finding the abstract automatically instead"
                 % (pdf.name, start, end))
            continue
        csv_ranges[pdf.name] = mapped
        abstract_pages[pdf.name] = {"requested": "%d-%d" % (start, end),
                                    "used": "%d-%d" % mapped}
        emit(event="log", detail="[override] %s: abstract pages %d-%d (%d-%d in the fixed copy)"
             % (pdf.name, start, end, mapped[0], mapped[1]))
    if csv_ranges:
        override_csv = write_override_csv(workdir / "overrides.csv", csv_ranges)

    # --- Pass 2: the OCR + VLM abstract pass over the whole batch ---------
    emit(event="stage", stage="ocr", detail="Reading abstracts")
    exit_code = run_abstract_pass(
        ocr_input, drafts_dir,
        model=model, mode=mode, ollama_url=ollama_url,
        on_progress=lambda payload: emit(**payload),
        override_csv=override_csv,
    )

    # --- Pass 3: turn the script's drafts into this app's outputs ---------
    emit(event="stage", stage="collecting", detail="Collecting results")
    documents = collect_outputs(drafts_dir, output_dir, fix_results, model, mode,
                                abstract_pages)
    return finish(documents, exit_code)


def write_manifest(output_dir: Path, documents: list) -> Path:
    """Write manifest.json for the whole job.

    Shape: a top-level ``documents`` list of per-file records, each carrying
    the fields asked for (filename, page_count, duplicates_removed,
    model_used, processed_at). A list rather than a bare object because one
    job can hold a whole batch, and a flat object could only describe one file.
    The heavier diff/recovery payloads are left out -- they are review data
    served over the API, not part of the durable record.
    """
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "documents": [
            {
                "filename": d["filename"],
                "page_count": d["page_count"],
                "duplicates_removed": d["duplicates_removed"],
                "model_used": d["model_used"],
                "processed_at": d["processed_at"],
            }
            for d in documents
        ],
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
