"""Read one PDF page several ways and report which ones find text.

For a PC where every page comes back "PaddleOCR detected no text": it shows
whether the problem is the CPU path, the CPU acceleration library, or the
models, and writes a report to send back. Run through launcher/ocr-check.cmd.

    python ocr_check.py THESIS.pdf [PAGE]

Each way runs in its own process, so one that crashes cannot stop the rest.
"""
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = {
    "app": "As the app runs it (CPU)",
    "cpu-noaccel": "CPU, acceleration (oneDNN) off",
    "gpu": "NVIDIA GPU",
}


def read_page(pdf, page, mode):
    """Child: OCR the page one way and print the line count."""
    if mode == "gpu":
        os.environ["OCR_USE_GPU"] = "1"
    sys.path.insert(0, str(HERE))
    import fitz
    import vlm_abstract as v  # hides the GPU on import unless OCR_USE_GPU=1

    pix = v.render_page_image(fitz.open(pdf), page - 1, dpi=350)
    if mode == "cpu-noaccel":
        from paddleocr import PaddleOCR
        v.paddle_ocr = PaddleOCR(
            lang="en", use_doc_orientation_classify=True, use_doc_unwarping=False,
            use_textline_orientation=True, text_det_thresh=0.3, text_det_box_thresh=0.5,
            text_det_unclip_ratio=1.5, text_det_limit_type="max",
            text_det_limit_side_len=2048, device="cpu", enable_mkldnn=False)
    else:
        v.init_paddle_ocr()
    words = v.paddle_ocr_page(pix)
    note = v.last_ocr_page_problem or ""
    print("@@RESULT@@ %d text segments %s" % (len(words), ("(" + note + ")") if note else ""))


def system_info():
    lines = ["Computer: %s" % platform.node(),
             "Windows: %s" % platform.platform(),
             "Processor: %s" % platform.processor()]
    try:
        import paddle
        import paddleocr
        lines.append("Paddle %s (CUDA build: %s), PaddleOCR %s" % (
            paddle.__version__, paddle.is_compiled_with_cuda(), paddleocr.__version__))
    except Exception as e:
        lines.append("Paddle import failed: %s" % e)
    models = Path.home() / ".paddlex" / "official_models"
    lines.append("Models folder: %s (%s)" % (models, "present" if models.exists() else "missing"))
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        lines.append("NVIDIA: %s" % (gpu.stdout.strip() or gpu.stderr.strip() or "none"))
    except Exception:
        lines.append("NVIDIA: nvidia-smi not found")
    return lines


def main():
    if len(sys.argv) == 4:
        read_page(sys.argv[1], int(sys.argv[2]), sys.argv[3])
        return
    pdf = sys.argv[1]
    page = sys.argv[2] if len(sys.argv) > 2 else "1"
    report = ["OCR check - %s, page %s" % (Path(pdf).name, page), ""] + system_info() + [""]
    print("\n".join(report), flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for mode, label in MODES.items():
        print("Trying: %s ..." % label, flush=True)
        proc = subprocess.run([sys.executable, __file__, pdf, page, mode], capture_output=True,
                              text=True, encoding="utf-8", errors="replace", env=env)
        lines = proc.stdout.splitlines() + proc.stderr.splitlines()
        found = [l.split("@@RESULT@@", 1)[1].strip() for l in lines if "@@RESULT@@" in l]
        if found:
            outcome = found[0]
        else:
            # PaddleOCR's own status lines say nothing about why it failed.
            chatter = ("Creating model", "Model files already", "Warning", "warn", "INFO", "ccache")
            useful = [l.strip() for l in lines if l.strip() and not any(c in l for c in chatter)]
            errors = [l for l in useful if "Error" in l or "error" in l]
            if errors:
                outcome = "FAILED: %s" % errors[-1][:200]
            else:
                outcome = "CRASHED (exit code %s, no error message)" % proc.returncode
        report.append("%-32s %s" % (label, outcome))
        print("   " + outcome, flush=True)

    out = Path.home() / "Desktop" / "ocr-check-report.txt"
    try:
        out.write_text("\n".join(report) + "\n", encoding="utf-8")
    except OSError:
        out = Path(os.environ.get("TEMP", ".")) / "ocr-check-report.txt"
        out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n" + "=" * 70)
    print("\n".join(report[-len(MODES):]))
    print("\nReport saved to: %s" % out)


if __name__ == "__main__":
    main()
