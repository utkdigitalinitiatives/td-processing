# OCR Pipeline Studio

A local desktop app for pulling abstracts out of scanned PDF theses.

Drop PDFs on the window. The app flags pages that look repeated, turns
sideways pages upright, runs the pages through PaddleOCR, optionally has a
local vision model re-read them, and then gives you an editor with a live
preview so you can fix up the result before saving it.

**Everything runs on this machine.** The only network traffic is to Ollama on
`localhost:11434`. Nothing is ever sent to the internet.

---

## What you need before you start

1. **Python 3.11, specifically.** Not 3.12 or 3.13. The OCR engine
   (`paddlepaddle` / `paddleocr`) only publishes installers for 3.11.
   Check what you have:

   ```
   py --list
   ```

   You want a line saying `-V:3.11`. If it is missing, install Python 3.11
   from python.org and re-run that command.

2. **Ollama**, running, with a vision model pulled. Get it from
   [ollama.com](https://ollama.com), then pull the model this pipeline was
   tuned against:

   ```
   ollama pull qwen2.5vl:3b
   ```

   The app will not download models for you on its own -- a first pull can be
   several gigabytes, which is not something to start in the middle of a job.

---

## Setup (once)

Open PowerShell in this folder and run these four commands in order.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

What each one does:

| Command | What it does |
| --- | --- |
| `py -3.11 -m venv .venv` | Creates a private Python 3.11 installation in a `.venv` folder, so this project's packages cannot conflict with anything else on your computer. |
| `.\.venv\Scripts\Activate.ps1` | Switches your terminal over to using that private Python. Your prompt gains a `(.venv)` prefix when it works. |
| `pip install -r requirements.txt` | Downloads the libraries listed in `requirements.txt`. This one takes a while -- PaddleOCR is a large download. |

If PowerShell refuses to run `Activate.ps1` with a message about execution
policies, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## Running it

With the venv activated (`(.venv)` showing in your prompt):

```powershell
python app.py
```

A native window opens. Close the window to quit -- the server shuts down with
it.

**The first run is slower than the rest.** The first time OCR happens,
PaddleOCR downloads its detection and recognition models into
`C:\Users\<you>\.paddlex\` (a few hundred MB). That happens once; afterwards
it loads them from disk. If the very first document seems stuck on page 1 for
several minutes, this is why. This is the only time the app reaches past your
own machine other than to Ollama, and it is PaddleOCR's own setup step, not
something the app sends anywhere.

Every following session is just two commands:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

---

## Using it

**Run screen**

1. Tick the **Steps to run** -- any combination, all ticked by default:

   | Step | What it does |
   | --- | --- |
   | Check for repeated pages | Flags pages that look like repeats of earlier ones. Quick. |
   | Turn sideways pages | Finds sideways pages and turns them upright. The slowest step, since it looks at every page -- skip it for batches you know are upright. |
   | Read the abstract (OCR) | Finds and reads each abstract. Needs the model and VLM pass below. |

   Untick OCR and the model and VLM pass are greyed out, Ollama does not need
   to be running, and the run stops after the page fixes -- for theses that
   have no abstract, the corrected PDFs in the run's `fixed/` folder are the
   result (**Open folder** takes you there). With neither page step ticked,
   OCR reads your PDFs exactly as dropped.

2. Pick a vision model. The dropdown lists what `ollama list` reports, so it
   only ever offers models you actually have. Then pick a VLM pass:

   | Mode | What it does |
   | --- | --- |
   | OCR only | PaddleOCR alone. Fastest, never contacts Ollama. |
   | Recovery on suspicious pages | Re-reads only pages whose OCR looks unusually sparse. |
   | Recovery on every page | Re-reads every abstract page. |
   | OCR/VLM diff report | Runs the model on every page and reports what it read differently, without touching the draft. |
   | **Diff + auto-merge safe fixes** | Default. Same comparison, but the corrections the script judges safe are applied to the draft for you. The rest are listed on the **OCR vs VLM** tab, one click each. |

3. Drop your PDFs and press **Run pipeline**.

The page steps run first, then OCR. Repeats are **never removed
automatically** -- the matching gets it wrong too often -- so you decide on the
Page fixes tab.

**Abstract in the wrong place?** Each dropped file has a box for its abstract
pages. Type them as numbered in your PDF (`5-8`, or `5` for one page) and the
app uses those instead of searching for the heading. Leave it blank to search
as usual.

When a run finishes, every file in the queue gets a **Rerun** button with
its own pages box. It starts a new run on just that file, without dropping it
in again, using whatever is ticked under **Steps to run** -- so untick OCR
first for page fixes only, or tick only OCR to read the abstract again. The
previous run's results are left alone.

The progress bar shows which file is being processed and which page within it.
The vision-model pass runs at the very end, after all OCR is finished, across
the whole batch at once -- that is deliberate, see "How this is put together"
below.

**Review screen**

The sidebar lists each finished document with coloured chips for pages worth
checking: `low-conf` (the OCR flagged the page as sparse), `equation` (a
stacked equation it could not read, left as a placeholder), and `diff` (the
vision model disagreed with the OCR somewhere on the page).

The main pane has the raw text on the left and a live preview on the right.
Type in the left, the right updates as you go. More tabs sit above it:
**OCR vs VLM** shows the differences side by side, **VLM recovery** shows
the model's own transcription of flagged pages, and **Page fixes** shows
what was changed in the PDF before OCR:

- each page that looks like a repeat, beside the earlier page it matched.
  Nothing has been removed: compare the two and use **Keep / Remove** under
  the pair. Switching is instant and you can change your mind as often as you
  like; the fixed PDF is only rewritten when you press **Save changes**, which
  takes out every page marked Remove and puts back any you kept again;
- each sideways page that was turned, as scanned and as corrected;
- pages that *may* be sideways but were left alone because the evidence was
  weak. Check these yourself.

Every page in the last two groups has a switch under it -- **None / 90° /
180° / 270°** -- like flipping between OCR and VLM text. The turn the fixed
PDF has now is pressed, the script's suggestion is marked, and clicking
another changes that page in the fixed PDF straight away. Use it to undo a
wrong turn, turn a page the script was unsure about, or turn one the other
way. Only the page's rotation setting changes, never the scan. In a full run
the abstract has already been read by then, so if you change a page inside
it, **Rerun** with OCR ticked reads it again.

Page numbers on that tab are your PDF's own. The sidebar chips
`possible repeat(s)`, `removed`, `rotated` and `rotation check` show the
counts. Removing or keeping a page only changes the fixed PDF; like a turn, it
does not re-read an abstract that was already read.

### Applying differences in one click

On the **OCR vs VLM** tab each difference is shown as two buttons: what the
OCR read on the left, what the vision model read on the right. Whichever one
the document currently says is outlined and not clickable. **Click the other
side and it goes straight into the document** -- you never have to find the
text in the editor yourself. Clicking back reverts it.

**Apply all remaining VLM fixes** does that for every difference still on the
OCR side at once, and **Revert all to OCR** undoes the lot. Both report what
they did underneath.

**When the words appear more than once.** The diff report only says *what*
changed, not *where*, so if the same words occur several times in the abstract
the app will not guess which copy is meant. Instead the row lists every copy
in its surrounding words -- with the likeliest one marked, going by the order
the differences were read in -- and you click the one to change. Only that
copy changes, and clicking the other side afterwards puts back exactly that
copy.

Once you pick, the list collapses to a single line saying which copy was
changed. Picked the wrong one? Open it again and click the right copy: the
change moves there, and the copy you picked first goes back to how it was.

One kind of difference still cannot be applied automatically, and the app
says so rather than pretending otherwise:

- *nothing to match against* -- the model added words the OCR missed
  entirely, so there is no existing text to swap out.

Nothing here touches the files on disk. Applying, reverting and typing all
stay in memory until you press **Save changes**, which writes them back to the
`.md` files in the run's `output/` folder, along with any Keep/Remove choices
on the Page fixes tab. **Open folder** takes you there.

If you close the window with anything unsaved, the app lists the documents
and asks first -- **Cancel** keeps the window open so you can save.

---

## Where the files go

Each run gets its own folder under `workdir/`, named for when it started and
what was in it -- `2026-09-16_143205_Thesis80.W5465`, or
`2026-09-16_143205_Thesis80.W5465_and_2_more` for a batch -- so sorting the
folder by name sorts it by time. Two runs started in the same second get
`-2`, `-3` on the end.

The **Open folder** button in the top bar opens the current run's folder (or
`workdir/` itself before any run), and the Progress panel shows its full path.

```
workdir/<date>_<time>_<first file>/
├── uploads/    the PDFs exactly as you dropped them
├── fixed/      the PDFs with sideways pages turned, and any repeats you
│               removed taken out -- what OCR reads, and the PDFs to keep
├── drafts/     the OCR script's own raw output
├── output/     <doc>.md, <doc>.pages.json, manifest.json
└── overrides.csv   abstract pages you typed, if any, as the OCR script reads them
```

Both fixes are found on the PDF you dropped and written out together, in a
single save per file. Turning a page only changes its rotation setting inside
the PDF; the scanned images themselves are never re-encoded, so nothing is
lost.

`workdir/` is git-ignored, so nothing you process is ever committed.

`manifest.json` records, per document: `filename`, `page_count`,
`duplicates_removed` (at the end of the run, always 0 -- removals are
yours to make), `model_used` and `processed_at`.

`<doc>.pages.json` is only written when a document actually has flagged pages.
It lists the page numbers and the reason each was flagged. **It does not
contain a confidence score** -- see the note below.

---

## A note on the `.md` files

The `.md` files contain HTML: `<p>`, `<sup>`, `<sub>` and character entities
like `&alpha;`. That is not a mistake. The OCR script's whole purpose is
producing that markup -- superscripts, subscripts and Greek letters correctly
tagged -- and converting it to plain Markdown would throw that work away. The
preview renders it correctly because the Markdown renderer is configured to
pass HTML through untouched.

## A note on confidence

The OCR script computes per-word confidence internally, but it writes that
data to a temporary file and deletes it when its run ends. What survives into
the output is *categorical*, not numeric: a page was flagged, and for what
reason. The sidebar chips reflect exactly that. The app deliberately does not
show a confidence percentage, because inventing one would be worse than
showing none.

---

## How this is put together

For anyone reading the code (the comments in each file go into more detail):

- **`app.py`** starts Flask on a background daemon thread and gives the main
  thread to pywebview. Both want to own a thread and only one can have the
  main one; GUI toolkits require it to be pywebview.
- **`server/__init__.py`** is an application factory, `create_app()`. It binds
  to `127.0.0.1` only, never `0.0.0.0`, which is the app's entire security
  model: there is no login anywhere, and it is safe only because nothing off
  this machine can reach it.
- **`server/jobs.py`** tracks jobs in memory with a lock, because the worker
  thread writes progress while Flask reads it to answer `/status`.
- **`pipeline/runner.py`** is the adapter layer. `dedupe.py` and
  `fix_rotation.py` are imported and called normally. `vlm_abstract.py` is driven through its command line
  instead, for three reasons: its deferred vision-model phase lives in
  `main()` and must run after all OCR is done (a loaded Ollama model breaks
  PaddleOCR's startup); it already re-launches itself as a subprocess; and it
  reports progress by printing, which a pipe can read without editing it.
- **`pipeline/dedupe.py`, `pipeline/fix_rotation.py` and
  `pipeline/vlm_abstract.py`** are the original scripts, unmodified. The
  originals are also still in `source_scripts/`.

One thing worth knowing about the dedupe step: the original script *detects*
duplicate pages but has no function that removes them. The removal is done in
`runner.py`, driven entirely by the page indices the original reports. None of
the matching logic was touched.

---

## Troubleshooting

**"Ollama isn't running"** -- start Ollama and wait a few seconds; the banner
clears itself. Or pick "OCR only" mode, which never contacts it.

**"The model ... isn't pulled on this machine"** -- run the `ollama pull`
command the message gives you.

**A document finishes as "skipped"** -- the OCR script could not find an
abstract heading in the first 15 pages. It looks for `ABSTRACT`,
`INTRODUCTION`, `INTRO`, `PURPOSE`, `PREFACE` or `SUMMARY` on a line of its
own. Open the pipeline log on the Run screen to see what it found.

**Installing takes forever / fails on `paddlepaddle`** -- confirm you are in
the 3.11 venv: `python --version` should say 3.11.x.

**Every page reports "0 text segments found" and no document appears** --
something has upgraded the OCR packages past the pinned versions. The newer
paddlepaddle (3.3.x) fails on every page with an internal
`ConvertPirAttribute2RuntimeAttribute` error that it reports as *zero text*
rather than as a crash, so the run looks healthy and quietly produces
nothing. Reinstall the pinned set:

```powershell
pip install -r requirements.txt --force-reinstall
```

The three `paddle*` pins in `requirements.txt` are exact for this reason. The
comments in that file explain what each one is guarding against. Do not
loosen them without running a real document through end to end afterwards.
