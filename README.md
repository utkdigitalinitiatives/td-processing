# td-processing

Tools for processing scanned thesis documents.

- **[auto_abstract/](auto_abstract/)** — OCR-based abstract extraction from
  scanned thesis PDFs, with optional VLM-assisted review/merge. See
  [auto_abstract/README.md](auto_abstract/README.md) for details.
- **[ocr_pipeline_studio/](ocr_pipeline_studio/)** — a local desktop app that
  wraps that extraction in a drop-files window: it flags repeated pages, turns
  sideways pages upright, runs the OCR, and hands you an editor with a live
  preview for fixing the result before saving. See
  [ocr_pipeline_studio/README.md](ocr_pipeline_studio/README.md) for details.
