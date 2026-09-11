"""Thin adapters around the two original scripts in ``source_scripts/``.

Nothing in this file reimplements what those scripts already do. The two
copies that live beside this module (``pipeline/dedupe.py`` and
``pipeline/vlm_abstract.py``) are byte-for-byte the files that were handed to
us; this module only *calls* them and reshapes their output into something a
background thread and a JSON API can work with.

Two very different calling styles are used here, and the reason for each
matters:

``dedupe.py`` is called **in-process, as a normal import**. Its core function
``find_and_export_duplicates()`` already returns a real Python list, so there
is nothing awkward to work around.

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

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import fitz  # PyMuPDF -- used only to *write* the deduped PDF, see dedupe_pdf()

# The original detector, imported and used exactly as written.
from pipeline.dedupe import find_and_export_duplicates

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
# Stage 1: dedupe
# --------------------------------------------------------------------------

@dataclass
class DedupeResult:
    """What one PDF's dedupe pass produced.

    A dataclass rather than a bare dict because these fields flow straight
    into manifest.json, and a typo in a dict key would only surface much
    later, in the UI, as a missing number.
    """
    source_name: str
    deduped_path: Path
    original_page_count: int
    kept_page_count: int
    duplicates: list = field(default_factory=list)

    @property
    def duplicates_removed(self) -> int:
        return self.original_page_count - self.kept_page_count


def dedupe_pdf(pdf_path: Path, deduped_dir: Path) -> DedupeResult:
    """Detect repeated pages with the original script, then write a copy of
    the PDF with those pages dropped.

    Note on the split of responsibilities: ``dedupe.py`` *detects* duplicates
    and can export a side-by-side comparison PDF, but it has no function that
    removes pages. Rather than edit it, the removal is done here, driven
    entirely by the ``dupe_idx`` values the original returned. All of the
    actual matching logic -- the length guard, the folio gate, the fuzzy
    ratio threshold -- stays in the original file, untouched.
    """
    deduped_dir.mkdir(parents=True, exist_ok=True)

    # The original returns a list of dicts, one per duplicate page found:
    #   {dupe_idx, orig_idx, duplicate_page, original_page, score, folio}
    # We pass no output_dir, so it does not write its comparison PDF -- only
    # the detection result is wanted here.
    duplicates = find_and_export_duplicates(pdf_path)

    # 0-based indices of pages to drop. A set because several later pages can
    # each match the same earlier original, and a page must only be dropped
    # once.
    dupe_indices = {int(d["dupe_idx"]) for d in duplicates}

    out_path = deduped_dir / pdf_path.name
    doc = fitz.open(pdf_path)
    try:
        original_page_count = doc.page_count
        keep = [i for i in range(original_page_count) if i not in dupe_indices]

        # Defensive: if detection somehow flagged everything, keep the file
        # intact rather than writing a zero-page PDF that would break OCR.
        if not keep:
            keep = list(range(original_page_count))

        # select() rewrites the document to exactly this page list, in order.
        doc.select(keep)
        doc.save(out_path, garbage=4, deflate=True)
        kept_page_count = len(keep)
    finally:
        doc.close()

    return DedupeResult(
        source_name=pdf_path.name,
        deduped_path=out_path,
        original_page_count=original_page_count,
        kept_page_count=kept_page_count,
        duplicates=duplicates,
    )


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
    cmd += VLM_MODES.get(mode, VLM_MODES[DEFAULT_VLM_MODE])["flags"]
    return cmd


def run_abstract_pass(
    input_dir: Path,
    out_dir: Path,
    model: str = DEFAULT_VLM_MODEL,
    mode: str = DEFAULT_VLM_MODE,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    on_progress: Optional[Callable[[dict], None]] = None,
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

    cmd = build_abstract_command(input_dir, out_dir, model, mode, ollama_url)

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


def span_state(text: str, ocr_span: str, vlm_span: str) -> str:
    """Which side of this diff the document currently reflects.

    Derived from the text itself rather than remembered in a flag, so it stays
    correct even after the user edits the textarea by hand. Returns "ocr",
    "vlm", or "unclear" when both or neither side can be found.
    """
    normalize = _get_normalizer()

    # A one-sided span: the model either added text or dropped it. Presence of
    # the side that does have text is what settles the question.
    if vlm_span == NOTHING_SPAN:
        return "ocr" if normalize(ocr_span) in text else "vlm"
    if ocr_span == NOTHING_SPAN:
        return "vlm" if normalize(vlm_span) in text else "ocr"

    ocr_present = normalize(ocr_span) in text
    vlm_present = normalize(vlm_span) in text
    if vlm_present and not ocr_present:
        return "vlm"
    if ocr_present and not vlm_present:
        return "ocr"
    return "unclear"


def apply_span(text: str, ocr_span: str, vlm_span: str, direction: str):
    """Rewrite one diff span in ``text`` to the chosen side.

    ``direction`` is "vlm" to take the model's reading or "ocr" to put the
    OCR's reading back. Returns ``(new_text, status)`` where status is:

      "applied"     -- the swap was made
      "unchanged"   -- that side is already what the document says
      "not_found"   -- the text to replace is not in the document
      "ambiguous"   -- it appears more than once, so replacing would be a guess
      "unplaceable" -- the side to remove is "(nothing)", i.e. this is a pure
                       insertion with no anchor text to find

    The uniqueness requirement is the same rule the script applies to its own
    merges. Refusing an ambiguous match is the entire safety property here:
    replacing the wrong occurrence would corrupt the document somewhere the
    user is not looking.
    """
    normalize = _get_normalizer()

    if direction == "vlm":
        find_raw, put_raw = ocr_span, vlm_span
    else:
        find_raw, put_raw = vlm_span, ocr_span

    if find_raw == NOTHING_SPAN:
        # Nothing to search for. The script auto-merges pure insertions using
        # neighbouring words as an anchor; reproducing that here would mean
        # reimplementing its anchor logic, so this is reported honestly
        # instead of guessed at.
        return text, "unplaceable"

    needle = normalize(find_raw)
    replacement = "" if put_raw == NOTHING_SPAN else normalize(put_raw)

    occurrences = text.count(needle)
    if occurrences == 0:
        # Already the other way round is the common, harmless case.
        other = "" if put_raw == NOTHING_SPAN else normalize(put_raw)
        if other and other in text:
            return text, "unchanged"
        return text, "not_found"
    if occurrences > 1:
        return text, "ambiguous"

    new_text = text.replace(needle, replacement, 1)

    # Deleting a span leaves debris behind: a doubled space where the words
    # were, a space stranded before punctuation, and -- when the span was a
    # whole paragraph, such as a heading the model was told to omit -- an
    # empty <p></p>. Paragraph text in these drafts is single-spaced, so these
    # cleanups are safe and keep the result looking like the rest of the file.
    if not replacement:
        new_text = re.sub(r"  +", " ", new_text)
        new_text = re.sub(r"\s+([.,;:])", r"\1", new_text)
        new_text = re.sub(r"<p>\s*</p>", "", new_text)

    return new_text, "applied"


def collect_outputs(
    draft_dir: Path,
    output_dir: Path,
    dedupe_results: dict,
    model_used: str,
    mode_used: str,
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

        dedupe = dedupe_results.get(stem + ".pdf")
        documents.append({
            "name": stem,
            "md_file": md_path.name,
            "filename": stem + ".pdf",
            "page_count": dedupe.kept_page_count if dedupe else None,
            "original_page_count": dedupe.original_page_count if dedupe else None,
            "duplicates_removed": dedupe.duplicates_removed if dedupe else 0,
            "duplicates": dedupe.duplicates if dedupe else [],
            "model_used": model_used,
            "mode_used": mode_used,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
            "flags": parsed["flags"],
            "diffs": parsed["diffs"],
            "recovery": parsed["recovery"],
        })

    return documents


# --------------------------------------------------------------------------
# The whole pipeline, start to finish
# --------------------------------------------------------------------------

# Subdirectory names inside workdir/<job_id>/. Named constants because both
# this module and the export route need to agree on them.
UPLOADS_DIRNAME = "uploads"
DEDUPED_DIRNAME = "deduped"
DRAFTS_DIRNAME = "drafts"
OUTPUT_DIRNAME = "output"


def run_pipeline(
    workdir: Path,
    model: str,
    mode: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    report: Optional[Callable[..., None]] = None,
) -> dict:
    """Run dedupe on every uploaded PDF, then one batch OCR/VLM pass, then
    collect the results.

    ``report`` is a plain callback taking keyword arguments. Passing a
    callback -- rather than having this module import the job store -- keeps
    the dependency arrow pointing one way: ``server`` knows about
    ``pipeline``, never the reverse. That is also what makes this function
    runnable from a plain script or a test with no Flask involved at all.

    Note the shape of the run: *all* dedupe work happens first, and then the
    OCR script is invoked exactly once for the whole batch. That ordering is
    not incidental -- see run_abstract_pass() for why the batch must stay
    whole.
    """
    def emit(**kwargs):
        if report is not None:
            report(**kwargs)

    uploads_dir = workdir / UPLOADS_DIRNAME
    deduped_dir = workdir / DEDUPED_DIRNAME
    drafts_dir = workdir / DRAFTS_DIRNAME
    output_dir = workdir / OUTPUT_DIRNAME

    pdfs = sorted(uploads_dir.glob("*.pdf"))
    if not pdfs:
        raise ValueError("No PDFs were uploaded for this job.")

    # --- Pass 1: dedupe, one file at a time -------------------------------
    emit(event="stage", stage="dedupe", detail="Removing repeated pages")
    dedupe_results = {}
    for index, pdf in enumerate(pdfs, 1):
        emit(event="dedupe_start", index=index, total=len(pdfs), filename=pdf.name)
        result = dedupe_pdf(pdf, deduped_dir)
        dedupe_results[pdf.name] = result
        emit(event="dedupe_done", filename=pdf.name,
             removed=result.duplicates_removed,
             kept=result.kept_page_count,
             original=result.original_page_count)

    # --- Pass 2: the OCR + VLM abstract pass over the whole batch ---------
    emit(event="stage", stage="ocr", detail="Reading abstracts")
    exit_code = run_abstract_pass(
        deduped_dir, drafts_dir,
        model=model, mode=mode, ollama_url=ollama_url,
        on_progress=lambda payload: emit(**payload),
    )

    # --- Pass 3: turn the script's drafts into this app's outputs ---------
    emit(event="stage", stage="collecting", detail="Collecting results")
    documents = collect_outputs(drafts_dir, output_dir, dedupe_results, model, mode)
    write_manifest(output_dir, documents)

    return {"documents": documents, "exit_code": exit_code}


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
