# Auto Abstract

Extracts abstracts from scanned thesis PDFs and turns them into clean,
markup-tagged HTML/TXT drafts — OCR plus a battery of fixups for the kind of
noise old typewritten academic documents produce: dropped superscripts and
subscripts, Greek letters, stacked equations, zero/O and l/1 misreads,
page-boundary paragraph breaks, and more.

## How it works

1. **OCR** — [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) reads each
   detected abstract page, with per-page process isolation so one page's
   crash doesn't lose the rest of the document.
2. **Cleanup** — a series of targeted fixups correct known OCR failure modes
   (scientific notation, chemical formulas, casing, stray punctuation,
   equation regions) and reconstruct paragraphs from the raw OCR lines.
3. **Optional VLM second pass** — a local vision-language model (via
   [Ollama](https://ollama.com)) can review pages that look suspicious, or
   diff its own reading against OCR's for every page, either just reporting
   the differences or merging in the ones judged safe. Fully opt-in.

Output is an HTML draft with inline `<sup>`/`<sub>` and formatting tags
(see `docspec.txt`-style conventions), ready for a human pass or conversion
to `.docx`.

## Quick start

See **[INSTRUCTIONS.md](INSTRUCTIONS.md)** for setup and full usage,
including the optional VLM review flags.

```
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
py .\abstract_ocr_paddle_cuda.py --input inputs --out outputs
```

## Project structure

```
abstract_ocr_paddle_cuda.py   Main script — OCR, fixups, VLM review/merge, CLI
override.csv                  Optional filename,pages overrides for abstract detection
inputs/  outputs/              Working I/O dirs (gitignored — see below)
completed_drafts/  temp folder/  Local working data (gitignored)
```

## About the data

`inputs/`, `outputs/`, `completed_drafts/`, and `temp folder/` are gitignored
on purpose: they hold scanned thesis PDFs and generated drafts, which may not
be this repo's to publish even in local history. Nothing in those folders is
required to run the script — they're just where your own PDFs go.

## License

[MIT](LICENSE)
