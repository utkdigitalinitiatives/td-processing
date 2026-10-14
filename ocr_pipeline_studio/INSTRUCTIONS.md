# Usage
## Setup

Double-click **`create-shortcut.cmd`** in this folder. It puts an
**OCR Pipeline Studio** icon on your Desktop, and that icon is the only thing
you need from then on.

Open it. The first time, it installs everything the app needs — Python 3.11 if
you do not already have it, then the packages, then the OCR engine — in a
window showing what it is doing. That downloads about 3 GB and takes 10 to 20
minutes. Every launch after that opens the app straight away.

If setup stops part way, nothing is broken and nothing already downloaded is
lost: open the shortcut again and it carries on from where it stopped.

On **macOS** there is no `.lnk`. Drag `launcher/launch.command` into the Dock,
or right-click it and choose Make Alias to put one on the Desktop; the first run
sets itself up the same way.

`create-shortcut.cmd` is the only file here meant to be double-clicked. What it
points at lives in **`launcher/`** and needs no attention: `launch.cmd` starts
the app with the venv's own interpreter, handing off to `setup.cmd` when there
is nothing installed yet.

### The vision model

The OCR/VLM comparison needs [Ollama](https://ollama.com) installed and
running. Setup offers to pull the model for you if Ollama is there, and points
you at the download if it is not — the app itself runs either way, and shows a
banner telling you which.

```
ollama pull qwen2.5vl:3b
```

That pull is about 3 GB, so it is not something to start mid-job.

## Doing it by hand

Setup does the following; there is no need to do it yourself, but here it is
for reference, or if you want to see where something failed. Open PowerShell in
this folder (`ocr_pipeline_studio/`, **not** the repository root — that is the
one mistake that produces a half-installed environment) and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

On **macOS** the first three are the equivalent three, and there is no fourth:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Python 3.11 exactly** — `paddlepaddle` / `paddleocr` publish no 3.12/3.13
wheels. Check with `py --list`; you want a line saying `-V:3.11`.

The fourth command is the OCR runtime, and on Windows and Linux it is not
optional: `requirements.txt` deliberately leaves paddle out there, so without
it OCR has nothing to run on. PyPI does not carry it past 2.6.2, which is why
it comes from Paddle's own index. Despite the name it still runs on the CPU —
the app hides the GPU unless `OCR_USE_GPU=1` — and this build is the one used
because it is several times faster at the same work: a 50-thesis batch ran in
43m instead of 1h16m. It adds ~2.5 GB; `pip install paddlepaddle==3.0.0` is a
working but slower substitute, and the one setup falls back to if that index
cannot be reached. macOS has no CUDA build, so `requirements.txt` installs that
plain wheel there and there is no fourth command.

A venv made by hand is for running from a terminal only: the shortcut does not
use it. The shortcut keeps its own environment on each PC, in
`%LOCALAPPDATA%\OCR Pipeline Studio\.venv`, so that the project folder can sit
on a network share — see "Running from a shared folder" below. It treats that
environment as finished only once `launcher/setup.cmd` has written the
`.setup-complete` stamp into it.

If PowerShell refuses to run `Activate.ps1` over execution policy, run this
once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## Running it

Double-click the Desktop shortcut, or from a terminal with the venv
activated — `(.venv)` showing in your prompt:

```powershell
python app.py
```

A native window opens; close it to quit, and the server shuts down with it.

Started from the shortcut there is no console window, so anything the app
prints goes to **`%LOCALAPPDATA%\OCR Pipeline Studio\launch.log`** instead —
paste that into Explorer's address bar to open it. That is the first place to
look if the icon seems to do nothing.

### Running from a shared folder

The project folder can live on a network share that several PCs open. Code
and `workdir/` are shared, so every PC sees the same runs; the Python
environment is not, because a venv only works on the PC that built it. On each
PC:

1. Open the shared folder and double-click `create-shortcut.cmd`.
2. Double-click the new Desktop shortcut. The first time on each PC this runs
   setup (10-20 minutes); after that it opens the app straight away.

On a Mac, open `launcher/launch.command` from the share instead; its
environment goes in `~/Library/Application Support/OCR Pipeline Studio/`. If
macOS refuses to open it from the network, right-click it and choose Open, or
run `chmod +x` on it once.

Two people can run at once — each run gets its own folder — but if two people
open and save the *same* run, the last save wins.

**The first run is slower.** The first time OCR happens, PaddleOCR downloads
its detection and recognition models into `C:\Users\<you>\.paddlex\` (a few
hundred MB), once. If the first document seems stuck on page 1 for several
minutes, this is why.

## Run screen

**1. Tick the steps to run** — any combination, all ticked by default:

| Step | What it does |
| --- | --- |
| Check for repeated pages | Flags pages that look like repeats of earlier ones. Quick. |
| Turn sideways pages | Finds sideways pages and turns them upright. The slowest step, since it looks at every page — skip it for batches you know are upright. |
| Read the abstract (OCR) | Finds and reads each abstract. Needs the model and VLM pass below. |

Untick OCR and the model and VLM pass grey out, Ollama need not be running,
and the run stops after the page fixes — for theses with no abstract, the
corrected PDFs in the run's `fixed/` folder are the result. With neither page
step ticked, OCR reads your PDFs exactly as dropped.

**2. Pick a vision model and a VLM pass.** The dropdown lists what
`ollama list` reports, so it only offers models you actually have.

| Mode | What it does |
| --- | --- |
| OCR only | PaddleOCR alone. Fastest, never contacts Ollama. |
| Recovery on suspicious pages | Re-reads only pages whose OCR looks unusually sparse. |
| Recovery on every page | Re-reads every abstract page. |
| OCR/VLM diff report | Runs the model on every page and reports what it read differently, without touching the draft. |
| **Diff + auto-merge safe fixes** | Default. Same comparison, but corrections the script judges safe are applied for you. The rest are listed on the **OCR vs VLM** tab, one click each. |

**3. Drop your PDFs and press Run pipeline.** Page steps run first, then OCR.
The vision-model pass runs at the very end, across the whole batch at once —
deliberately, see [How this is put together](#how-this-is-put-together).

**Abstract in the wrong place?** Each dropped file has a box for its abstract
pages. Type them as numbered in your PDF (`5-8`, or `5` for one page) and the
app uses those instead of searching for the heading. Blank searches as usual.

**Rerun.** When a run finishes, every file in the queue gets a **Rerun**
button with its own pages box. It starts a new run on just that file, using
whatever is ticked under **Steps to run** — untick OCR for page fixes only, or
tick only OCR to read the abstract again. The previous run is left alone.

## Review screen

The sidebar lists each finished document with chips for pages worth checking:
`low-conf` (OCR flagged the page as sparse), `equation` (a stacked equation it
could not read, left as a placeholder) and `diff` (the vision model disagreed
somewhere on the page).

The main pane has the raw text on the left and a live preview on the right.
Above it: **OCR vs VLM** for the differences side by side, **VLM recovery**
for the model's own transcription of flagged pages, and **Page fixes** for
what was changed in the PDF before OCR.

### Applying differences in one click

Each difference is two buttons: what the OCR read, and what the model read.
Whichever the document currently says is outlined and not clickable. Click the
other side and it goes straight into the document — you never have to find the
text in the editor. Clicking back reverts it. **Apply all remaining VLM
fixes** and **Revert all to OCR** do the same in bulk and report what they did.

**When the words appear more than once**, the report only says *what* changed,
not *where*, so the app does not guess. The row lists every copy in its
surrounding words, likeliest one marked, and you click the copy to change.
Picked the wrong one? Open it again and click the right copy: the change moves
there and the first copy goes back to how it was.

One kind of difference cannot be applied automatically, and the app says so:
*nothing to match against* — the model added words the OCR missed entirely, so
there is no existing text to swap out.

### Page fixes

- Each page that looks like a repeat, beside the page it matched. Nothing has
  been removed: compare the two and use **Keep / Remove**. Switching is
  instant; the fixed PDF is only rewritten when you press **Save changes**.
- Each sideways page that was turned, as scanned and as corrected.
- Pages that *may* be sideways but were left alone because the evidence was
  weak. Check these yourself.

Pages in the last two groups have a **None / 90° / 180° / 270°** switch. The
turn the fixed PDF has now is pressed, the script's suggestion is marked, and
clicking another changes that page straight away. Only the page's rotation
setting changes — the scan is never re-encoded, so nothing is lost.

Page numbers here are your PDF's own. Removing, keeping or turning a page only
changes the fixed PDF; it does not re-read an abstract already read. To pick
up such a change, **Rerun** that file with OCR ticked.

### Saving

Applying, reverting and typing all stay in memory until you press **Save
changes**, which writes the text back to the `.md` files in the run's
`output/` folder along with any Keep/Remove choices. **Open folder** takes you
there. Close the window with anything unsaved and the app lists the documents
and asks first.

## Where the files go

Each run gets its own folder under `workdir/`, named for when it started and
what was in it — `2026-09-16_143205_Thesis80.W5465_and_2_more` — so sorting by
name sorts by time. Two runs started in the same second get `-2`, `-3`.

```
workdir/<date>_<time>_<first file>/
├── uploads/    the PDFs exactly as you dropped them
├── fixed/      sideways pages turned, repeats you removed taken out — what
│               OCR reads, and the PDFs to keep
├── drafts/     the OCR script's own raw output
├── output/     <doc>.md, <doc>.pages.json, manifest.json
├── job.json    the review itself: differences, flags, your Keep/Remove and
│               page-turn choices, unsaved edits, the run's settings
├── run.json    a few lines about the run, for the "past runs" list
└── overrides.csv   abstract pages you typed, as the OCR script reads them
```

`manifest.json` records per document: `filename`, `page_count`,
`duplicates_removed` (0 at the end of a run — removals are yours to make),
`model_used` and `processed_at`. `<doc>.pages.json` is written only when a
document has flagged pages, and lists the page numbers and each reason.

### Handing a run over

A run folder holds the whole run, so one person can run batches and another
review them later:

1. Run the batch. `job.json` is saved after the run and after every change you
   make while reviewing, so there is nothing to export.
2. Open the app on any machine that has it and use **Or open a past run**.
   Runs in that machine's `workdir/` are listed; **Choose a folder...** opens
   one from anywhere — a shared drive, a USB stick.
3. The Review screen comes back as it was: your text, the differences with
   which side each is on (including which copy of a repeated difference was
   picked), and the Page fixes switches still working.

Copy the whole folder, not parts of it. A folder with no `job.json` — a run
from an older version, or one assembled by hand — still opens: the text comes
from `output/<doc>.md` and the differences are read back out of `drafts/`.
What cannot come back is which pages were flagged as repeats and which were
turned, since nothing on disk records that; the app says so on the Page fixes
tab, and the corrected PDF in `fixed/` is unaffected.

## Two things worth knowing

**The `.md` files contain HTML** — `<p>`, `<sup>`, `<sub>`, entities like
`&alpha;`. That is not a mistake. Producing that markup is the OCR script's
whole purpose, and converting it to plain Markdown would throw the work away.
The preview renders it because its Markdown renderer passes HTML through.

**There is no confidence score.** The OCR script computes per-word confidence
internally but writes it to a temporary file and deletes it when its run ends.
What survives is categorical, not numeric: a page was flagged, and for what
reason. The sidebar chips reflect exactly that. Inventing a percentage would
be worse than showing none.

## How this is put together

For anyone reading the code — the comments in each file go into more detail:

- **`app.py`** starts Flask on a background daemon thread and gives the main
  thread to pywebview, which the GUI toolkits require. It binds to `127.0.0.1`
  only, never `0.0.0.0`: there is no login anywhere, and the app is safe only
  because nothing off this machine can reach it.
- **`server/jobs.py`** tracks jobs in memory under a lock, because the worker
  thread writes progress while Flask reads it to answer `/status`.
- **`pipeline/runner.py`** is the adapter layer. `dedupe.py` and
  `fix_rotation.py` are imported and called normally. `vlm_abstract.py` is
  driven through its command line instead, for three reasons: its deferred
  vision-model phase lives in `main()` and must run after all OCR is done (a
  loaded Ollama model breaks PaddleOCR's startup); it already re-launches
  itself as a subprocess; and it reports progress by printing, which a pipe
  can read without editing it.

The original dedupe script *detects* duplicate pages but has no function that
removes them. The removal is done in `runner.py`, driven entirely by the page
indices the original reports. None of the matching logic was touched.

## Troubleshooting

**"Ollama isn't running"** — start Ollama and wait a few seconds; the banner
clears itself. Or pick "OCR only", which never contacts it.

**"The model ... isn't pulled on this machine"** — run the `ollama pull`
command the message gives you.

**A document finishes as "skipped"** — the OCR script found no abstract
heading in the first 15 pages. It looks for `ABSTRACT`, `INTRODUCTION`,
`INTRO`, `PURPOSE`, `PREFACE` or `SUMMARY` on a line of its own. Open the
pipeline log on the Run screen to see what it found.

**Installing takes forever / fails on `paddlepaddle`** — confirm you are in
the 3.11 venv: `python --version` should say 3.11.x. The runtime is ~2.5 GB,
so that command is slow the first time.

**No NVIDIA card in the machine** — fine: the app hides the GPU and runs on
the CPU regardless. If the CUDA build will not install there, use
`pip install paddlepaddle==3.0.0`; slower, but everything works.

**Every page reports "0 text segments found" and no document appears** —
something upgraded the OCR packages past the pinned versions. Newer
paddlepaddle (3.3.x) fails on every page with an internal
`ConvertPirAttribute2RuntimeAttribute` error that it reports as *zero text*
rather than as a crash, so the run looks healthy and quietly produces nothing.
Reinstall the pinned set:

```powershell
pip install -r requirements.txt --force-reinstall
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ --force-reinstall
```

The `paddle*` versions are exact for this reason — the two in
`requirements.txt` and the runtime above. Do not loosen them without running
a real document through end to end afterwards.
