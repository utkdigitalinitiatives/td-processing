# Usage

## Setup

Requires **Python 3.11** — `paddlepaddle` 3.0.0 has no 3.12/3.13 wheel yet.

```
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

(Already have `.venv`? Just run the activate line.)

## Basic run

Add PDFs to `inputs/`, then:

```
py .\abstract_ocr_paddle_cuda.py --input inputs --out outputs
```

This writes one HTML (or TXT) draft per PDF to `outputs/`. Abstract page
boundaries are auto-detected; if detection gets a document wrong, add a row
to `override.csv` (`filename,pages`, e.g. `Thesis76.K355.pdf,5-8`) and pass
`--override-csv override.csv`.

Other flags worth knowing:
- `--confidence-threshold 0.60` — drop OCR detections below this score.
- `--force-single-paragraph` — merge the draft into one `<p>` block.

## Optional: VLM review pass (Ollama)

PaddleOCR alone occasionally misses or misreads a line. A local vision-language
model can double-check it — this is opt-in and never required.

**Setup:** install [Ollama](https://ollama.com), have it running, and pull a
model (default is `qwen2.5vl:3b`: `ollama pull qwen2.5vl:3b`). If Ollama isn't
reachable, these flags are silently skipped rather than failing the run.

- `--vlm-review` — review pages that look suspicious (sparse OCR output) and
  append the VLM's reading as a separate block per page, for manual
  comparison. Never edits the primary draft text.
  - `--vlm-review-mode targeted|always` — `targeted` (default) only reviews
    flagged pages; `always` reviews every abstract page.
  - `--vlm-trigger-threshold N` — segment-count floor that flags a page in
    targeted mode (default 20).
- `--vlm-diff-review` — independent of `--vlm-review`: runs the VLM on
  *every* abstract page and prints a word-level diff against OCR's text at
  the bottom of the draft. Also never edits the primary text.
- `--vlm-diff-merge` — experimental, builds on `--vlm-diff-review`: also
  applies the diff spans judged safe (e.g. a missed superscript, a dropped
  Greek letter) directly into the primary draft, unmarked. Everything else
  stays report-only for a human to check.
- `--vlm-model NAME` / `--ollama-url URL` — override the model tag / server
  address (defaults: `qwen2.5vl:3b`, `http://localhost:11434`).

## Converting output to .docx

From `cmd` (not PowerShell):

```
"C:\Program Files\LibreOffice\soffice" --headless --convert-to docx *.txt
```

## Known limitations

- Does not detect new paragraphs or remove line breaks in every case —
  review generated drafts before use.
