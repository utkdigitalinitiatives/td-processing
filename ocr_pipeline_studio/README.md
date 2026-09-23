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

See **[INSTRUCTIONS.md](INSTRUCTIONS.md)** for setup and full usage.

Requires **Python 3.11** — `paddlepaddle` publishes no 3.12/3.13 wheels — and
[Ollama](https://ollama.com) running with `ollama pull qwen2.5vl:3b`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## Project structure

```
app.py                Entry point — Flask on a background thread, pywebview on the main one
server/               The local API: create_app(), job tracking, routes
pipeline/             Vendored copies of the original scripts: dedupe, fix_rotation, vlm_abstract
pipeline/runner.py    Adapter layer — calls those scripts, reshapes their output
source_scripts/       The same originals, as handed over
ui/                   The single page the whole app lives in
workdir/              One folder per run (gitignored — see below)
```

`pipeline/vlm_abstract.py`, `source_scripts/abstract_ocr_paddle_cuda.py` and
[`../auto_abstract/abstract_ocr_paddle_cuda.py`](../auto_abstract/abstract_ocr_paddle_cuda.py)
are the same file, byte for byte — the app vendors the script it drives rather
than importing it across the repository. Change one, change all three.

## About the data

`workdir/` is gitignored on purpose: it holds the PDFs you drop and everything
produced from them, which may not be this repo's to publish even in local
history. Each run gets its own folder, recording nothing about where it lives,
so it can be copied to another machine and opened there as it was.

## License

[MIT](LICENSE)
