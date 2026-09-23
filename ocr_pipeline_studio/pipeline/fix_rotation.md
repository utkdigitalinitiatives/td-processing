# fix_rotation.py

Finds pages where a table or figure was printed sideways and rotates them
upright. Lossless — it sets one key in the PDF, it does not re-save the scan.

## Setup

**If you already have the venv:**

```powershell
cd ocr-pipeline-studio
.\.venv\Scripts\Activate.ps1
```

**If you don't:** you need Python **3.11** — not 3.12 or 3.13, paddleocr has no
installer for those. Check with `py --list`, and get 3.11 from python.org if
it's missing. Then, in the folder holding `fix_rotation.py`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install PyMuPDF numpy paddleocr==3.2.0 paddlex==3.2.1 paddlepaddle==3.0.0
```

Those version numbers are exact on purpose. A newer paddlepaddle reports every
page as blank instead of failing, so the run looks fine and does nothing.

The first run downloads a 7 MB model. That's the only time it needs internet.

## Use

Run these in order. `-r` includes subfolders; drop it if you don't want that.
The path can also be a single PDF.

**1. See what it would do.** Changes nothing.

```powershell
python fix_rotation.py "C:\path\to\theses" -r
```

**2. Check its work.** Also writes `ROTATED_<name>.pdf` holding just the pages
it wants to turn, already turned. Open it — if they read upright, it's right.

```powershell
python fix_rotation.py "C:\path\to\theses" -r -o "C:\path\to\review"
```

**3. Apply.** Corrected copies into a different folder, originals untouched:

```powershell
python fix_rotation.py "C:\path\to\theses" -r --apply -o "C:\path\to\rotated"
```

Or overwrite the originals, keeping a `.bak` of each:

```powershell
python fix_rotation.py "C:\path\to\theses" -r --in-place
```

## Output

```
[Thesis80.M327.pdf] - Found 30 sideways page(s), 2 for review:
  * Page 44 -> /Rotate 90 (high: text layer + image agree)
  ? Page 3  -> /Rotate 180 (review: image only, near-blank page, score 0.87)
```

`*` gets rotated. `?` does **not** — the evidence was weak, so it's left for
you. Worth skimming; a real sideways page turns up there occasionally.

## Notes

- Running it twice is safe. It won't undo its own work.
- `--apply` won't write into the source folder — that's what `--in-place` is for.
- It won't flip an upside-down page unless both checks agree. Those go to `?`.
- It fixes what you see, not the invisible OCR text underneath.
- `--min-score` (default `0.70`) — lower it to catch more, raise it to be stricter.
