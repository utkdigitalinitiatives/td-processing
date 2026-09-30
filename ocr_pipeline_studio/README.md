# OCR Pipeline Studio

A local desktop app for pulling abstracts out of scanned thesis PDFs. Drop
PDFs on the window; it flags repeated pages, turns sideways pages upright,
runs the pages through OCR, optionally has a local vision model re-read them,
and hands you an editor with a live preview for fixing the result before
saving.

**Everything runs on this machine.** The only network traffic is to Ollama on
`localhost:11434`, plus PaddleOCR's one-time model download on first run.

This is one of the tools in [td-processing](../README.md); the OCR script it
drives lives beside it in [`auto_abstract/`](../auto_abstract/).

## How it works

1. **Page fixes** — repeated pages are flagged and sideways pages turned
   upright, then written out together as one corrected copy. Repeats are never
   removed automatically; the matching gets it wrong too often, so you decide.
2. **OCR** — [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) reads each
   abstract it finds, through the `auto_abstract` script. Set the pages by hand,
   per file, when detection gets one wrong.
3. **Optional VLM pass** — a local vision-language model (via
   [Ollama](https://ollama.com)) re-reads suspicious pages, or diffs its reading
   against OCR's on every page: reporting the differences, or merging the ones
   judged safe.
4. **Review** — an editor with live preview, the OCR/VLM differences as
   one-click swaps, and a Page fixes tab with Keep/Remove and rotation
   switches. Nothing reaches disk until you press **Save changes**.

## Quick start

Double-click **`create-shortcut.cmd`**, then open the **OCR Pipeline Studio**
icon it puts on your Desktop. The first run installs everything — Python 3.11
if it is missing, the packages, and the OCR engine — in a window you can watch;
about 3 GB and 10-20 minutes, once. After that the icon just opens the app.

macOS: drag `launcher/launch.command` into the Dock and open it; the first run
sets itself up the same way.

The OCR/VLM comparison also wants [Ollama](https://ollama.com) with
`ollama pull qwen2.5vl:3b`; setup offers to pull it, and the app runs without
it either way.

To set up by hand instead, or to see exactly what setup does, see
**[INSTRUCTIONS.md](INSTRUCTIONS.md)**.

## Project structure

```
app.py                Entry point — Flask on a background thread, pywebview on the main one
create-shortcut.cmd   Run once to put an OCR Pipeline Studio icon on the Desktop
launcher/             What that icon runs; nothing in here needs opening by hand
launcher/launch.cmd     Starts the app with the venv's own pythonw, no console
launcher/setup.cmd      First-run install, called by launch.cmd when the venv is missing
launcher/launch.command The macOS equivalent of both
server/               The local API: create_app(), job tracking, routes
pipeline/             Vendored copies of the original scripts: dedupe, fix_rotation, vlm_abstract
pipeline/runner.py    Adapter layer — calls those scripts, reshapes their output
ui/                   The single page the whole app lives in
workdir/              One folder per run (gitignored — see below)
```

`pipeline/vlm_abstract.py` started as a byte-for-byte copy of
[`../auto_abstract/abstract_ocr_paddle_cuda.py`](../auto_abstract/abstract_ocr_paddle_cuda.py)
— the app vendors the script it drives rather than importing it across the
repository — but **the two have deliberately diverged.** The app's copy
carries the speed work: one long-lived OCR worker per batch instead of a
process per page, a cap on the image the text detector sees, and page images
cropped to their text before they go to the vision model. On a 50-thesis
batch that is 43m against 1h16m. `auto_abstract/` is unchanged and slower.

So do **not** re-sync them by copying either over the other; that would undo
the speed work or drag it somewhere it has not been tested. A fix that
belongs in both — anything about how the abstract itself is found or
written — has to be applied to each.

## About the data

`workdir/` is gitignored on purpose: it holds the PDFs you drop and everything
produced from them, which may not be this repo's to publish even in local
history. Each run gets its own folder, recording nothing about where it lives,
so it can be copied to another machine and opened there as it was.

## License

[MIT](LICENSE)
