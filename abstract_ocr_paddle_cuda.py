import warnings
# Suppress harmless dependency version warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*urllib3.*')
warnings.filterwarnings('ignore', message='.*chardet.*')

# Force UTF-8 on stdout/stderr to avoid UnicodeEncodeError on Windows console
# Helps batch processing of PDFs with non-ASCII characters in text or metadata.
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import difflib
import re
import csv
import json
import base64
import time
import math
import html
import inspect
import subprocess
import tempfile
import numpy as np
import tqdm
from PIL import Image
import fitz  # PyMuPDF
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal
from dotenv import load_dotenv

VALID_ABSTRACT_HEADINGS = ["ABSTRACT", "INTRODUCTION", "INTRO", "PURPOSE", "PREFACE", "SUMMARY"]  # Add more allowed headings here

# Force CPU-only behavior before any Paddle/PaddleOCR imports.
# This prevents CUDA probing on machines with GPU-linked Paddle builds.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FLAGS_selected_gpus"] = ""
os.environ["PADDLE_DEVICE"] = "cpu"
# Prefer legacy/non-PIR execution paths to avoid ConvertPirAttr issues
# seen with mismatched Paddle/PaddleOCR builds.
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("FLAGS_enable_new_ir_in_executor", "0")

# Silence MuPDF's own warning/error output so tagged PDFs don't spam stderr
try:
    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
except Exception:
    pass

# ---- Load env
load_dotenv()

# ---- PaddleOCR 
try:
    from paddleocr import PaddleOCR
except ImportError:
    print("PaddleOCR not found. Install with: pip install paddlepaddle paddleocr")
    raise

# Global PaddleOCR instance - lazy initialized
paddle_ocr = None


def filter_supported_paddle_kwargs(desired_kwargs: dict):
    """Return kwargs limited to PaddleOCR's supported signature."""
    try:
        sig = inspect.signature(PaddleOCR.__init__)
        allowed = set(sig.parameters)
        allowed.discard('self')
        filtered = {k: v for k, v in desired_kwargs.items() if k in allowed}
        unsupported = sorted(k for k in desired_kwargs if k not in allowed)
        return filtered, unsupported
    except Exception:
        return desired_kwargs, []

def init_paddle_ocr() -> PaddleOCR:
    """Initialize PaddleOCR in CPU-only compatibility mode."""
    global paddle_ocr

    print("Initializing PaddleOCR (CPU mode) ...")
    
    desired_kwargs = {
        'lang': 'en',
        'use_doc_orientation_classify': True,
        'use_doc_unwarping': False,
        'use_textline_orientation': True,
        # NOTE: current PaddleOCR/PaddleX builds use the 'text_det_*' names below,
        # not the legacy 'det_db_*' names. A generous unclip_ratio matters here:
        # too small and the text detector fails to merge a full line's bounding
        # box when the line contains a superscript/subscript (e.g. "R2" in "R2
        # for..."), silently dropping that entire line before OCR even runs on it.
        'text_det_thresh': 0.3,
        'text_det_box_thresh': 0.5,
        'text_det_unclip_ratio': 1.5,
    }

    filtered_kwargs, unsupported = filter_supported_paddle_kwargs(desired_kwargs)

    # Explicitly set backend device to CPU before creating OCR instance.
    try:
        import paddle
        paddle.set_device('cpu')
        print("  Paddle device forced to CPU")
    except Exception as e:
        print(f"  Could not force CPU device explicitly: {e}")

    # Initialize PaddleOCR with compatible arguments only
    try:
        paddle_ocr = PaddleOCR(**filtered_kwargs)
        if unsupported:
            print(f"  Skipped unsupported PaddleOCR args: {', '.join(unsupported)}")
        print("✓ PaddleOCR initialized successfully (accuracy mode)!")
    except Exception as e:
        print(f"Error initializing PaddleOCR: {e}")
        raise
    
    return paddle_ocr

# ABSTRACT_HEADING_RE = re.compile(r"^\s*ABSTRACT\s*\.?\s*$", re.IGNORECASE)

def _heading_regex(heading: str) -> re.Pattern:
    return re.compile(r"^\s*" + re.escape(heading) + r"\s*\.?\s*$", re.IGNORECASE)

def _normalize_heading_text(text: str) -> str:
    return re.sub(r"[^A-Z]", "", (text or "").upper())

def _fuzzy_heading_match(line: str, heading: str) -> bool:
    line_norm = _normalize_heading_text(line)
    heading_norm = _normalize_heading_text(heading)
    if not line_norm or not heading_norm:
        return False
    if line_norm[0] != heading_norm[0]:
        return False
    if len(line_norm) < max(4, len(heading_norm) - 3):
        return False
    if len(line_norm) > len(heading_norm) + 3:
        return False
    ratio = difflib.SequenceMatcher(None, heading_norm, line_norm).ratio()
    if len(heading_norm) <= 5:
        return ratio >= 0.90
    if len(heading_norm) <= 8:
        return ratio >= 0.82
    return ratio >= 0.75
STOP_HEADINGS_RE = re.compile(
    r"^\s*(keywords?|key\s*words?|chapter(\s+\d+)?|table of contents|acknowledg(e)?ments?|references?|bibliography|contents)\s*\.?\s*$",
    re.IGNORECASE
)

# Named entities for common Greek & math 
GREEK_MAP = {
    "α": "&alpha;", "β": "&beta;", "γ": "&gamma;", "δ": "&delta;", "ε": "&epsilon;",
    "ζ": "&zeta;",  "η": "&eta;",  "θ": "&theta;", "ι": "&iota;",  "κ": "&kappa;",
    "λ": "&lambda;","μ": "&mu;",   "ν": "&nu;",    "ξ": "&xi;",     "ο": "o",
    "π": "&pi;",    "ρ": "&rho;",  "σ": "&sigma;","τ": "&tau;",    "υ": "&upsilon;",
    "φ": "&phi;",   "χ": "&chi;",  "ψ": "&psi;",  "ω": "&omega;",
    "Γ": "&Gamma;","Δ": "&Delta;","Θ": "&Theta;","Λ": "&Lambda;","Ξ": "&Xi;",
    "Π": "&Pi;",   "Σ": "&Sigma;","Υ": "&Upsilon;","Φ": "&Phi;","Ψ": "&Psi;","Ω": "&Omega;"
}
MATH_MAP = {
    "≤": "&le;","≥": "&ge;","±": "&plusmn;","×": "&times;","÷": "&divide;","≈": "&asymp;",
    "≃": "&simeq;","≅": "&cong;","≠": "&ne;","→": "&rarr;","←": "&larr;","↔": "&harr;",
    "⇒": "&rArr;","⇐": "&lArr;","∞": "&infin;","°": "&deg;","µ": "&micro;","∑": "&sum;",
    "∏": "&prod;","√": "&radic;","∫": "&int;","∂": "&part;","∇": "&nabla;","∈": "&isin;",
    "∉": "&notin;","∩": "&cap;","∪": "&cup;","⊂": "&sub;","⊃": "&sup;","⊆": "&sube;","⊇": "&supe;",
    "·": "&middot;","′": "&prime;","″": "&Prime;"
}

# ---- Known scientific notation, matched by exact text rather than OCR geometry ----
# Problem: OCR often misreads superscripts/subscripts, and we have no visual offset info at the line level to detect them. So we only tag known scientific tokens by exact text match here.
SCIENTIFIC_NOTATION_SUP = {
    "R2": "R<sup>2</sup>", "r2": "r<sup>2</sup>",
    "R3": "R<sup>3</sup>", "r3": "r<sup>3</sup>",
    "cm2": "cm<sup>2</sup>",
}
CHEMICAL_FORMULA_SUB = {
    "H2O": "H<sub>2</sub>O", "CO2": "CO<sub>2</sub>", "O2": "O<sub>2</sub>", "N2": "N<sub>2</sub>",
    "NH3": "NH<sub>3</sub>", "NH4": "NH<sub>4</sub>", "SO2": "SO<sub>2</sub>", "SO4": "SO<sub>4</sub>",
    "NO2": "NO<sub>2</sub>", "NO3": "NO<sub>3</sub>", "CaCO3": "CaCO<sub>3</sub>", "MgCl2": "MgCl<sub>2</sub>",
    "MgSO4": "MgSO<sub>4</sub>", "H2SO4": "H<sub>2</sub>SO<sub>4</sub>", "HNO3": "HNO<sub>3</sub>",
    "CH4": "CH<sub>4</sub>", "C6H12O6": "C<sub>6</sub>H<sub>12</sub>O<sub>6</sub>", "FeCl3": "FeCl<sub>3</sub>",
    "Fe2O3": "Fe<sub>2</sub>O<sub>3</sub>", "Al2O3": "Al<sub>2</sub>O<sub>3</sub>", "CuSO4": "CuSO<sub>4</sub>",
    "AgNO3": "AgNO<sub>3</sub>", "Na2CO3": "Na<sub>2</sub>CO<sub>3</sub>", "K2CO3": "K<sub>2</sub>CO<sub>3</sub>",
    "BaSO4": "BaSO<sub>4</sub>", "VO2": "VO<sub>2</sub>", "C6H6": "C<sub>6</sub>H<sub>6</sub>",
    "C2H5OH": "C<sub>2</sub>H<sub>5</sub>OH",
}
# Isotope notation (leading mass-number superscript before the element
# symbol, plus a trailing superscript "m" for a metastable state if present).
ISOTOPE_NOTATION_SUP = {
    "89Zr": "<sup>89</sup>Zr", "89Zrm": "<sup>89</sup>Zr<sup>m</sup>",
    "89Y": "<sup>89</sup>Y", "89y": "<sup>89</sup>Y",
    "89Nb": "<sup>89</sup>Nb", "89Nbm": "<sup>89</sup>Nb<sup>m</sup>",
    "89Mo": "<sup>89</sup>Mo", "89Mom": "<sup>89</sup>Mo<sup>m</sup>",
    "90Mo": "<sup>90</sup>Mo", "92Mo": "<sup>92</sup>Mo",
}
# Unit/identifier OCR fixes: common misreads confirmed against source scans.
UNIT_OCR_FIXES = {
    "GFa": "GPa",
    "ilvBll2": "ilvB112",
}
# Non-chemical subscript notation confirmed by exact text match (e.g. "K2"
# for a dust-resistivity coefficient). Kept as its own dict, distinct from
# CHEMICAL_FORMULA_SUB, since these aren't chemical formulas.
SUBSCRIPT_NOTATION_SUB = {
    "K2": "K<sub>2</sub>",
}
_KNOWN_SCIENCE_NOTATION = {**SCIENTIFIC_NOTATION_SUP, **CHEMICAL_FORMULA_SUB, **ISOTOPE_NOTATION_SUP, **UNIT_OCR_FIXES, **SUBSCRIPT_NOTATION_SUB}
_SCIENCE_TOKEN_RE = re.compile(r"\b\w+\b")

# Common OCR misreads of scientific phrases that are not single tokens, so we can't catch them with the token regex above. These are applied by exact text match.
SCIENCE_PHRASE_FIXES = {
    "Ca 2+": "Ca<sup>2+</sup>",
    "Mg 2+": "Mg<sup>2+</sup>",
    "2.Oug": "2.0µg",
    "log ft": "log <i>ft</i>",
    "K-2": "K<sub>2</sub>",
    "ft.2": "ft.<sup>2</sup>",
    "Les )": "Les<sup>-</sup>)",
    "Rec )": "Rec<sup>-</sup>)",
    "leu-l": "leu-1",
}

def apply_known_science_notation(text: str) -> str:
    """Tag known statistical/chemical/unit tokens (e.g. 'R2', 'CO2', 'GFa')
    by exact text match, since the OCR line-level boxes give us no visual
    offset to detect them by position. See _KNOWN_SCIENCE_NOTATION and
    SCIENCE_PHRASE_FIXES above."""
    if not text:
        return text
    for phrase, replacement in SCIENCE_PHRASE_FIXES.items():
        if phrase in text:
            text = text.replace(phrase, replacement)
    if _KNOWN_SCIENCE_NOTATION:
        text = _SCIENCE_TOKEN_RE.sub(
            lambda m: _KNOWN_SCIENCE_NOTATION.get(m.group(0), m.group(0)), text
        )
    return text

# ---- Common OCR character-confusion fixes ----
# Applied to reconstructed paragraph text. Deliberately narrow patterns so
# they only fire in unambiguous contexts, leaving real words/formulas alone.
#
# Leading-zero-style decimal fractions (".Ol" -> ".01", ".l45" -> ".145")
_DECIMAL_FRACTION_RE = re.compile(r"(?<![\w.])\.([O0-9lI]{1,4})(?!\w)")
_ZERO_O_MID_RE = re.compile(r"(?<=\d)O(?=\d)")

def fix_zero_o_confusion(text: str) -> str:
    # Fix common OCR confusion between zero and letter O (and l/I) in decimal fractions.
    # Note: l/I are often misrecognized as 1, but we don't want to blindly replace them in words, so we only fix them in decimal fractions (".l45" -> ".145").
    if not text:
        return text
    def _norm(m: "re.Match") -> str:
        token = m.group(1)
        fixed = token.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"}))
        return "." + fixed
    text = _DECIMAL_FRACTION_RE.sub(_norm, text)
    text = _ZERO_O_MID_RE.sub("0", text)
    return text

# Degree symbol OCR confusion: "100oC" or "100O'C" -> "100°C"
_DEGREE_APOSTROPHE_RE = re.compile(r"\b(\d{1,3})[0oO]['’]([CF])\b", re.IGNORECASE)

def fix_degree_celsius(text: str) -> str:
    if not text:
        return text
    return _DEGREE_APOSTROPHE_RE.sub(lambda m: m.group(1) + "0°" + m.group(2).upper(), text)

# Scientific exponent notation: "3 x 10-5" -> "3×10<sup>-5</sup>"
_SCI_EXPONENT_RE = re.compile(r"(?<=[\d.])\s*[xX]\s*10(\d{1,3})(?!\d)")

def fix_scientific_exponent_notation(text: str) -> str:
    if not text:
        return text
    return _SCI_EXPONENT_RE.sub(lambda m: f"×10<sup>{m.group(1)}</sup>", text)

# Common abbreviations that end with a period but are not sentence-ending. We don't want to remove the period from these, but we do want to avoid treating them as sentence boundaries when reconstructing paragraphs.
_ABBREVIATIONS_WITH_PERIOD = {
    "etc", "vs", "dr", "mr", "mrs", "ms", "prof", "fig", "figs", "eq", "eqs",
    "vol", "vols", "al", "cf", "approx", "ca", "no", "nos", "pp", "sp", "spp",
}
_STRAY_PERIOD_RE = re.compile(r"\b([A-Za-z]{2,})\.\s(?=[a-z])")

def fix_stray_period_before_lowercase(text: str) -> str:
    if not text:
        return text
    def _norm(m: "re.Match") -> str:
        word = m.group(1)
        if word.lower() in _ABBREVIATIONS_WITH_PERIOD:
            return m.group(0)
        return word + " "
    return _STRAY_PERIOD_RE.sub(_norm, text)

# ---- Mixed-case term casing fixes

_CASING_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")

def _is_notable_mixed_case(token: str) -> bool:
    # True if the token has at least 2 letters and contains both uppercase and lowercase letters, but is not a simple capitalized word (e.g., "The", "Dna-as-typo").
    letters = [c for c in token if c.isalpha()]
    if len(letters) < 2:
        return False
    has_upper = any(c.isupper() for c in letters)
    has_lower = any(c.islower() for c in letters)
    if not (has_upper and has_lower):
        return False
    # Reject ordinary capitalized words (The, Dna-as-typo) that are not known mixed-case terms.
    if letters[0].isupper() and all(c.islower() for c in letters[1:]):
        return False
    return True

def build_casing_reference(doc, min_occurrences: int = 2, min_dominance: float = 0.7) -> dict:
    # Build a reference dict of known mixed-case terms (dNMP, mRNA, ATPase) from the document text.
    from collections import Counter
    counts: dict = {}
    for i in range(doc.page_count):
        try:
            text = doc.load_page(i).get_text("text")
        except Exception:
            continue
        if not text:
            continue
        for tok in _CASING_TOKEN_RE.findall(text):
            counts.setdefault(tok.lower(), Counter())[tok] += 1

    reference = {}
    for key, variants in counts.items():
        top_variant, top_count = variants.most_common(1)[0]
        total = sum(variants.values())
        if not _is_notable_mixed_case(top_variant):
            continue
        if total < min_occurrences:
            continue
        if (top_count / total) < min_dominance:
            continue
        reference[key] = top_variant
    return reference

def fix_known_term_casing(text: str, casing_reference: dict) -> str:
    # Correct OCR-mangled casing of known mixed-case terms (dNMP, mRNA, ATPase)
    if not text or not casing_reference:
        return text
    return _CASING_TOKEN_RE.sub(
        lambda m: casing_reference.get(m.group(0).lower(), m.group(0)), text
    )

# Collapse stray double periods (". .", ".  .", etc.) down to a single period.
_DOUBLE_PERIOD_RE = re.compile(r"(?<!\.)\.\s?\.(?!\.)")

def fix_double_periods(text: str) -> str:
    """Collapse an OCR-inserted stray extra '.' -- adjacent or separated by
    the single space line-joining inserts -- down to one period.
    """
    if not text:
        return text
    return _DOUBLE_PERIOD_RE.sub(".", text)

@dataclass
class OCRWord:
    page: int
    text: str
    conf: float
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2

# ---------- Utilities ----------
def render_page_image(doc, page_index: int, dpi: int = 400):
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix  # Pixmap

def is_noise(text: str, conf: float) -> bool:
    """Detect if a detected text segment is likely noise/garbage.
    Returns True if we should filter it out.
    """
    # Very low confidence is almost always noise
    if conf < 0.50:
        return True
    
    # Single punctuation or special character (but allow common particles)
    if len(text) == 1 and text not in "aAIi-'.,;:":
        return True
    
    # Very short segments with low-to-moderate confidence
    if len(text) == 2 and conf < 0.70:
        return True
    
    # Pure punctuation/symbols (except hyphen which might separate words)
    if text and len(text.strip()) > 0:
        # Count letters/digits vs punctuation
        alphanumeric = sum(1 for c in text if c.isalnum())
        if alphanumeric == 0:  # All non-alphanumeric
            return True
    
    # Common noise patterns from dirty scans
    noise_patterns = [
        r"^[^a-zA-Z0-9]*$",  # No letters or digits at all
        r"^[_\-\.]+$",       # Just lines/dashes
        r"^[\(\)\[\]\{\}]+$", # Just brackets
    ]
    for pattern in noise_patterns:
        if re.match(pattern, text):
            return True
    
    return False

def paddle_ocr_page(pix, confidence_threshold: float = 0.60) -> List[OCRWord]:
    """Run OCR using PaddleOCR with noise filtering for dirty scans.
    
    Args:
        pix: Pixmap to process
        confidence_threshold: Filter out detections below this confidence (0.0-1.0)
    """
    global paddle_ocr
    if paddle_ocr is None:
        raise RuntimeError("PaddleOCR not initialized. Call init_paddle_ocr() first.")
    
    # Convert Pixmap to numpy array
    img_bytes = pix.tobytes("png")
    img_array = np.array(Image.open(BytesIO(img_bytes)))
    
    # Run PaddleOCR with modern predict API
    try:
        result = paddle_ocr.predict(img_array)
    except Exception as e:
        print(f"PaddleOCR failed: {e}")
        return []
    
    words = []
    # Guard against blank/unreadable pages
    if not result or not isinstance(result, list) or len(result) == 0:
        return words
    
    # New PaddleOCR API returns: [{'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...]}]
    page_result = result[0]
    rec_texts = page_result.get('rec_texts', [])
    rec_scores = page_result.get('rec_scores', [])
    rec_polys = page_result.get('rec_polys', page_result.get('dt_polys', []))
    
    # Iterate through detected text
    for i, text in enumerate(rec_texts):
        try:
            if not text or not text.strip():
                continue
            
            # Get confidence score
            conf = float(rec_scores[i]) if i < len(rec_scores) else 0.95
            
            # Filter out noise and low-confidence detections
            if conf < confidence_threshold:
                continue
            
            if is_noise(text.strip(), conf):
                continue
            
            # Get bounding box polygon [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            if i < len(rec_polys):
                bbox_points = rec_polys[i]
                xs = [point[0] for point in bbox_points]
                ys = [point[1] for point in bbox_points]
                x_min = int(min(xs))
                y_min = int(min(ys))
                x_max = int(max(xs))
                y_max = int(max(ys))
            else:
                # Fallback bbox if polygon missing
                x_min, y_min, x_max, y_max = 0, 0, 100, 20
            
            words.append(OCRWord(page=-1, text=text.strip(), conf=conf, 
                               bbox=(x_min, y_min, x_max, y_max)))
        except Exception as e:
            # Skip malformed entries but continue processing
            continue
    
    return words

# ---- Per-page OCR process isolation ----
# PaddleOCR's CPU inference has been observed to crash with a native access
# violation (an uncatchable segfault, not a Python exception) intermittently.
# To avoid losing the entire PDF processing, we isolate each page's OCR in a separate subprocess. If a crash occurs, we can retry once or skip the page without affecting the rest of the document.
def run_ocr_page_worker(pdf_path: Path, page_index: int, dpi: int, confidence_threshold: float, out_json: Path):
    """Child-process entry point: render one page, OCR it, write results as JSON."""
    with fitz.open(pdf_path) as doc:
        pix = render_page_image(doc, page_index, dpi=dpi)
    init_paddle_ocr()
    words = paddle_ocr_page(pix, confidence_threshold=confidence_threshold)
    payload = [{"text": w.text, "conf": w.conf, "bbox": list(w.bbox)} for w in words]
    out_json.write_text(json.dumps(payload), encoding="utf-8")

def ocr_page_isolated(pdf_path: Path, page_index: int, dpi: int, confidence_threshold: float,
                       max_attempts: int = 2) -> List[OCRWord]:
    """Run OCR for one page in an isolated subprocess, retrying once on crash."""
    script_path = Path(__file__).resolve()
    for attempt in range(1, max_attempts + 1):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "words.json"
            cmd = [
                sys.executable, str(script_path),
                "--ocr-page",
                "--pdf", str(pdf_path),
                "--page", str(page_index),
                "--dpi", str(dpi),
                "--confidence-threshold", str(confidence_threshold),
                "--out-json", str(out_json),
            ]
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0 and out_json.exists():
                try:
                    payload = json.loads(out_json.read_text(encoding="utf-8"))
                except Exception:
                    payload = None
                if payload is not None:
                    return [
                        OCRWord(page=page_index, text=item["text"], conf=item["conf"], bbox=tuple(item["bbox"]))
                        for item in payload
                    ]
            if attempt < max_attempts:
                print(" (crashed, retrying)", end="", flush=True)
    print(" ⚠ OCR crashed twice, skipping page", end="", flush=True)
    return []

# ---- Optional VLM review pass (Ollama, opt-in) ----
# See plan notes for why this is opt-in and not the default. The VLM pass is slower than PaddleOCR, so we only trigger it on pages that are suspiciously sparse (e.g., missing a sentence) or have very low OCR confidence. The VLM prompt is tuned to transcribe old typewritten abstracts with minimal hallucination, but it is still a generative model and can produce errors, so we append its output as a separate HTML block for human review rather than splicing it into the main draft.

DEFAULT_VLM_MODEL = "qwen2.5vl:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_VLM_TRIGGER_THRESHOLD = 20  # segment count floor; see plan notes

try:
    import requests as _requests
except ImportError:
    _requests = None

VLM_PROMPT = (
    "This is a scanned page from an old typewritten academic thesis abstract. "
    "Transcribe ALL the body text on this page exactly as written, word for word. "
    "Formatting rules:\n"
    "- Wrap superscript characters (like exponents or isotope numbers) in <sup></sup> tags.\n"
    "- Wrap subscript characters in <sub></sub> tags.\n"
    "- If you see a mathematical equation laid out as a stacked fraction (numerator "
    "above a line, denominator below), transcribe it as inline text using a slash "
    "for division and <sup>/<sub> tags for any exponents or subscripted variables, "
    "e.g. K<sub>2</sub> = (eta_f * k') / rho_p * S<sup>2</sup> * (1-epsilon)/epsilon<sup>3</sup>.\n"
    "- Do NOT invent, guess, or paraphrase any text you cannot clearly read. If a word "
    "or symbol is illegible, write [ILLEGIBLE] instead of guessing.\n"
    "- Do not include page numbers, headers, or footers.\n"
    "- Output only the transcribed text, no commentary."
)

_vlm_availability_cache: dict = {}

def _vlm_available(model: str, ollama_url: str, attempts: int = 4, retry_delay: float = 2.0) -> bool:
    """Check (cached) whether Ollama is reachable and `model` is pulled.

    Retries a few times with a short delay before giving up. A single
    impatient check isn't reliable right after a machine restart -- Ollama's
    background service can take several seconds to start listening on its
    HTTP port, and a one-shot 3s-timeout GET made right in that window reads
    as "unavailable" even though the service is only moments from being
    ready (confirmed: --vlm-diff-review run alone right after a reboot got
    skipped this way, while --vlm-review run alongside it happened to check
    late enough -- via either extra elapsed time or the shared cache below
    -- to see Ollama already up. That's a timing accident, not a real
    dependency between the two flags, so it's fixed here for both.)
    """
    cache_key = (model, ollama_url)
    if cache_key in _vlm_availability_cache:
        return _vlm_availability_cache[cache_key]
    available = False
    if _requests is not None:
        for attempt in range(1, attempts + 1):
            try:
                resp = _requests.get(f"{ollama_url}/api/tags", timeout=3)
                if resp.status_code == 200:
                    names = {m.get("name") for m in resp.json().get("models", [])}
                    available = model in names
                break  # reachable (even if the model itself isn't pulled) -- no point retrying
            except Exception:
                available = False
            if attempt < attempts:
                time.sleep(retry_delay)
    _vlm_availability_cache[cache_key] = available
    return available

def _page_is_suspicious(segment_count: int, prior_counts: List[int], threshold: int) -> bool:
    """Flag a page as worth a VLM second look.

    Heuristic, not proven: real data (see plan notes) showed 25-28
    segments on typical dense abstract pages but 17 on a page missing a
    whole sentence -- a floor threshold catches that, and comparing
    against this document's own prior pages catches a partial drop that
    doesn't cross the floor on an otherwise-denser document. Expect both
    false positives (a genuinely short abstract) and false negatives (a
    small drop on an already-sparse page) until tuned on more batches.
    """
    if segment_count < threshold:
        return True
    if prior_counts:
        avg_prior = sum(prior_counts) / len(prior_counts)
        if avg_prior > 0 and segment_count < 0.7 * avg_prior:
            return True
    return False

def vlm_transcribe_page(pdf_path: Path, page_index: int, model: str = DEFAULT_VLM_MODEL,
                         ollama_url: str = DEFAULT_OLLAMA_URL, dpi: int = 300,
                         max_attempts: int = 2, first_attempt_timeout: int = 180,
                         retry_timeout: int = 45) -> Optional[str]:
    """Render one page and ask the VLM to transcribe it. Returns None on failure
    after one retry -- never raises, so a flaky VLM call can't fail the batch.

    Timeouts are deliberately asymmetric, calibrated against measured
    qwen2.5vl:3b behavior: a genuine cold start (model not yet loaded into
    VRAM -- the normal case for the *first* call of a deferred VLM phase,
    since the PaddleOCR pass ahead of it takes minutes, well past Ollama's
    idle unload) measured 121.8s once; warm calls measured 7-35s. So
    first_attempt_timeout=180s covers a cold start with real margin, while
    retry_timeout=45s is sized for a warm model -- Ollama keeps processing a
    request server-side even after our client gives up waiting on it, so by
    the time a retry fires, the model is very likely already warm regardless
    of whether the first attempt "succeeded". Worst case: 225s (3.75 min),
    down from a naive 2x180s=360s, without breaking the common cold-start case
    the way a flat 45s/attempt did (that failed on literally every first call).
    """
    if _requests is None:
        return None
    with fitz.open(pdf_path) as doc:
        pix = render_page_image(doc, page_index, dpi=dpi)
    img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

    payload = {
        "model": model,
        "prompt": VLM_PROMPT,
        "images": [img_b64],
        "stream": False,
        # num_ctx matters: the image alone consumes most of Ollama's 4096
        # default context, and the full prompt + response overflows it.
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 2048},
    }
    for attempt in range(1, max_attempts + 1):
        timeout = first_attempt_timeout if attempt == 1 else retry_timeout
        try:
            resp = _requests.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            if text:
                return text
        except Exception:
            pass
    return None

def _unload_vlm_model(model: str, ollama_url: str) -> None:
    """Explicitly unload the model from Ollama (keep_alive=0) once every
    deferred VLM phase in this run is done. Confirmed necessary: Ollama
    keeps a model resident for minutes after last use, and re-running the
    script again in that window -- a normal thing to do while
    iterating/testing, or just processing back-to-back batches -- hits the
    exact same GPU-contention bug the deferred-execution design exists to
    avoid within a single run: PaddleOCR's own init silently fails on every
    page ("0 text segments found") for the *next* run too. Best-effort
    cleanup only, never raises -- not correctness-critical for the run
    that's already finished, just hygiene for whatever runs next.
    """
    if _requests is None:
        return
    try:
        _requests.post(f"{ollama_url}/api/generate", json={"model": model, "keep_alive": 0}, timeout=10)
    except Exception:
        pass

# Human-readable label per flag reason -- see _page_is_suspicious (reason
# "low_confidence") and the equation-placeholder section below (reason
# "equation"). Keeping this as a small lookup, not inline string logic, so
# a new reason only needs one new entry here.
_VLM_FLAG_REASON_LABELS = {
    "low_confidence": "flagged as low-confidence",
    "equation": "flagged for equation content (transcription unverified -- "
                "see the [EQUATION] placeholder(s) in the paragraph above)",
}

def _build_vlm_supplementary_html(vlm_blocks: List[Tuple[int, str, str]], casing_reference: dict, vlm_model: str) -> str:
    """Turn (page_index, reason, raw_vlm_text) triples into HTML-comment-labeled
    supplementary blocks, running each through the same per-paragraph fixup
    chain as normal paragraphs (fix_zero_o_confusion, apply_known_science_notation,
    etc. -- harmless even on already-clean VLM text). See the "Optional VLM
    review pass" section for why these are appended, not spliced into the draft.
    """
    sections = []
    for page_idx, reason, vlm_text in vlm_blocks:
        vlm_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", vlm_text) if p.strip()]
        fixed = []
        for para in vlm_paragraphs:
            t = fix_zero_o_confusion(para)
            t = fix_double_periods(t)
            t = fix_stray_period_before_lowercase(t)
            t = fix_degree_celsius(t)
            t = fix_scientific_exponent_notation(t)
            t = fix_known_term_casing(t, casing_reference)
            t = apply_known_science_notation(t)
            t = replace_greek_math(t)
            t = escape_user_content(t)
            t = escape_ampersands_not_entities(t)
            t = convert_remaining_non_ascii(t)
            fixed.append(t)
        block_html = to_html_paragraphs(fixed).replace("\n", "")
        label = _VLM_FLAG_REASON_LABELS.get(reason, "flagged for review")
        sections.append(
            f"<!-- VLM RECOVERY -- page {page_idx+1} {label} "
            f"({vlm_model}); please verify against the source PDF and merge into "
            f"the paragraph above if correct -->\n{block_html}"
        )
    return "\n\n".join(sections)

# ---- Full OCR/VLM diff pass (--vlm-diff-review, isolated from the recovery
# pathway above) ----
# _page_is_suspicious only catches pages that look sparse -- it can't catch
# a page where OCR produced a normal-looking amount of text that's subtly
# wrong (confirmed on Thesis76.K82: "PAo" for "PAO", a missing superscript
# on "(Her )", on pages with 25/37 segments -- well above the flag
# threshold, so --vlm-review never touches that document at all). This is
# a different, unconditional pass: every page gets sent to the VLM
# regardless of suspicion, and instead of attempting any recovery/merge,
# the two independent transcriptions are diffed and the differences are
# printed as a report -- never touching the primary draft text. Whether a
# given difference "looks like" a dropped sub/superscript versus a VLM
# slip (see the fail/fall finding on K82 -- the VLM's own transcription
# isn't reliable enough to trust automatically either) is left for a human
# to judge for now; classifying that automatically is future work, not
# this pass's job.
def _diff_opcodes(ocr_text: str, vlm_text: str) -> List[Tuple[str, List[str], List[str]]]:
    """Word-level opcodes between OCR's and VLM's text for the same page
    (difflib.SequenceMatcher, word-tokenized). Includes "equal" runs too --
    the merge logic below needs them to build anchor text for insertions;
    callers that only want the differences filter tag != "equal"."""
    ocr_words = ocr_text.split()
    vlm_words = vlm_text.split()
    matcher = difflib.SequenceMatcher(None, ocr_words, vlm_words, autojunk=False)
    return [(tag, ocr_words[i1:i2], vlm_words[j1:j2]) for tag, i1, i2, j1, j2 in matcher.get_opcodes()]

def _diff_ocr_vs_vlm(ocr_text: str, vlm_text: str) -> str:
    """Word-level diff between OCR's and the VLM's independent
    transcription of the same page. Returns "" if they agree."""
    lines = []
    for tag, ocr_words, vlm_words in _diff_opcodes(ocr_text, vlm_text):
        if tag == "equal":
            continue
        ocr_span = " ".join(ocr_words) or "(nothing)"
        vlm_span = " ".join(vlm_words) or "(nothing)"
        lines.append(f'  OCR: "{ocr_span}"  |  VLM: "{vlm_span}"')
    return "\n".join(lines)

def _build_vlm_diff_report_html(diff_blocks: List[Tuple[int, str, str]]) -> str:
    """diff_blocks: (page_index, ocr_text, vlm_text) triples. One section
    per page that has at least one difference; pages that fully agree are
    skipped -- no noise for pages OCR already got right."""
    sections = []
    for page_idx, ocr_text, vlm_text in diff_blocks:
        diff = _diff_ocr_vs_vlm(ocr_text, vlm_text)
        if not diff:
            continue
        sections.append(
            f"<!-- OCR/VLM DIFF REPORT -- page {page_idx+1}, informational only, "
            f"no inline changes made -->\n{diff}"
        )
    return "\n\n".join(sections)

# ---- Inline diff-merge (--vlm-diff-merge, experimental -- see the
# "Inline diff-merge" plan) ----
# Classifies each diff span above and, for the ones judged safe, applies
# the VLM's correction directly into the primary draft text instead of
# only reporting it. The rules below are deliberately narrow and were
# derived from a real diff report (Thesis76.K355), not guessed. Two
# shapes are trusted: a short pure insertion (a dropped sub/superscript-
# sized token, e.g. "(nothing)" -> "rho_p") and a short trailing addition
# on top of text OCR already got right (OCR's span is an exact prefix of
# VLM's, e.g. "ε)/ε" -> "ε)/ε<sup>3</sup>"). Everything else stays
# report-only: multi-word insertions (real content recovery -- that's
# --vlm-review's job), character-confusion substitutions like l/1 (the
# maps' job), and substitutions with no shared prefix (confirmed unsafe
# on real data -- "Fu"/"eta_f", and "ε"/"epsilon" where OCR was already
# the more correct rendering).
_EQUATION_PLACEHOLDER_MARKER = "[EQUATION"
_MERGE_ANCHOR_WORDS = 4
_MERGE_MAX_ADDITION_CHARS = 8
_TAG_RE = re.compile(r"<[^>]+>")

def _visible_len(text: str) -> int:
    """Length with HTML tags stripped -- a sub/superscript addition is
    always wrapped in <sup>...</sup>/<sub>...</sub> (confirmed on real
    data: "ε)/ε" -> "ε)/ε<sup>3</sup>" adds 12 raw characters for a
    1-character payload), so the "is this short enough to trust"
    threshold has to measure the actual added content, not its markup."""
    return len(_TAG_RE.sub("", text))

def _classify_diff_span(tag: str, ocr_span: str, vlm_span: str) -> Literal["merge", "flag"]:
    """Decide whether one diff span is safe to auto-apply ("merge") or
    should stay report-only ("flag"). ocr_span/vlm_span are the rendered
    "(nothing)"-or-text strings, same as in the diff report."""
    if _EQUATION_PLACEHOLDER_MARKER in ocr_span or _EQUATION_PLACEHOLDER_MARKER in vlm_span:
        # The equation-placeholder marker text leaks into the raw OCR
        # side of the diff (placeholder substitution runs before the
        # per-page OCR text is captured -- see extract_abstract_region).
        # These spans compare a placeholder marker against real content
        # and must never be touched.
        return "flag"
    if tag == "delete":
        # OCR has content the VLM doesn't transcribe -- never
        # auto-remove OCR content, only ever add or correct it.
        return "flag"
    if tag == "insert":
        return "merge" if _visible_len(vlm_span) <= _MERGE_MAX_ADDITION_CHARS else "flag"
    if tag == "replace":
        if vlm_span.lower() == ocr_span.lower():
            return "merge"  # pure case fix
        if (vlm_span.lower().startswith(ocr_span.lower())
                and _visible_len(vlm_span) - _visible_len(ocr_span) <= _MERGE_MAX_ADDITION_CHARS):
            return "merge"  # short trailing addition, e.g. a dropped superscript
    return "flag"

def _wrap_vlm_merge(text: str, ocr_span: str, vlm_span: str) -> str:
    """Visibly mark auto-applied text -- same "always show your work,
    never blend silently" precedent as the equation placeholders and the
    VLM recovery blocks. Greppable, trivially strippable once this pass
    is trusted enough to blend in."""
    title = html.escape(f'OCR: "{ocr_span}" | VLM: "{vlm_span}"', quote=True)
    return f'<span class="vlm-merge" title="{title}">{text}</span>'

def _normalize_for_primary_text(text: str) -> str:
    """Approximate how raw OCR/VLM text ends up rendered in the final
    draft, for the narrow purpose of locating (or inserting) it there --
    reuses the same Greek/math-symbol and non-ASCII-to-entity conversions
    the real per-paragraph fixup chain applies (replace_greek_math,
    convert_remaining_non_ascii). Confirmed necessary on real data: raw
    OCR/VLM text has a literal "ε", the final draft already has
    "&epsilon;" -- without this, searching for the raw character always
    comes up empty and a genuinely mergeable span (ε)/ε -> ε)/ε<sup>3</sup>)
    falls back to [FLAGGED] for no real reason. Deliberately doesn't
    attempt the rest of the fixup chain (casing reference, stray-period
    detection, etc.) -- those are context-dependent on the whole
    paragraph, not meaningful to apply to a short span in isolation."""
    return convert_remaining_non_ascii(replace_greek_math(text))

def _apply_vlm_diff_merges(draft_text: str, diff_blocks: List[Tuple[int, str, str]]) -> Tuple[str, str]:
    """Apply "merge"-classified spans directly into draft_text's primary
    paragraph content, page by page in order, and render the same report
    _build_vlm_diff_report_html would -- annotated per line with whether
    it was actually merged, and left [FLAGGED] (with the rest, unchanged)
    when a "merge"-classified span couldn't be safely located.

    Returns (updated_draft_text, report_str). draft_text is never
    mutated past its primary paragraph prefix -- both existing append
    points (the VLM recovery block, this same diff report) begin with
    "\\n\\n<!--", so anything already appended by an earlier phase in
    this run is split off first and reattached untouched.
    """
    marker = "\n\n<!--"
    split_at = draft_text.find(marker)
    primary, rest = (draft_text, "") if split_at == -1 else (draft_text[:split_at], draft_text[split_at:])

    sections = []
    for page_idx, ocr_text, vlm_text in diff_blocks:
        opcodes = _diff_opcodes(ocr_text, vlm_text)
        report_lines = []
        for idx, (tag, ocr_words, vlm_words) in enumerate(opcodes):
            if tag == "equal":
                continue
            ocr_span = " ".join(ocr_words) or "(nothing)"
            vlm_span = " ".join(vlm_words) or "(nothing)"
            merged = False
            if _classify_diff_span(tag, ocr_span, vlm_span) == "merge":
                vlm_normalized = _normalize_for_primary_text(vlm_span)
                if tag == "insert":
                    anchor_words = (opcodes[idx - 1][1][-_MERGE_ANCHOR_WORDS:]
                                     if idx > 0 and opcodes[idx - 1][0] == "equal" else [])
                    anchor = _normalize_for_primary_text(" ".join(anchor_words))
                    if anchor:
                        matches = list(re.finditer(re.escape(anchor), primary, re.IGNORECASE))
                        if len(matches) == 1:
                            pos = matches[0].end()
                            addition = _wrap_vlm_merge(vlm_normalized, ocr_span, vlm_span)
                            primary = primary[:pos] + " " + addition + primary[pos:]
                            merged = True
                elif tag == "replace":
                    ocr_normalized = _normalize_for_primary_text(ocr_span)
                    matches = list(re.finditer(re.escape(ocr_normalized), primary, re.IGNORECASE))
                    if len(matches) == 1:
                        m = matches[0]
                        replacement = _wrap_vlm_merge(vlm_normalized, ocr_span, vlm_span)
                        primary = primary[:m.start()] + replacement + primary[m.end():]
                        merged = True
            label = "[MERGED]" if merged else "[FLAGGED]"
            report_lines.append(f'  {label} OCR: "{ocr_span}"  |  VLM: "{vlm_span}"')
        if not report_lines:
            continue
        sections.append(
            f"<!-- OCR/VLM DIFF REPORT -- page {page_idx+1}, [MERGED] lines were "
            f"applied inline above (see <span class=\"vlm-merge\">), [FLAGGED] "
            f"lines were not -->\n" + "\n".join(report_lines)
        )
    return primary + rest, "\n\n".join(sections)

# ---- Equation-region placeholder
# The OCR pipeline has no reliable way to detect stacked fractions or other
# multi-line equations, so we insert a placeholder in the paragraph text for
# each detected equation region.

ENABLE_EQUATION_PLACEHOLDERS = True

_WORDLIKE_TOKEN_RE = re.compile(r"[A-Za-z]{2,}")

def _line_is_equation_like(line: List[OCRWord], median_height: float, median_width: float) -> bool:
    stats = _line_stats(line)
    width = stats["x2"] - stats["x1"]
    if width > 0.5 * median_width:
        return False
    texts = [(w.text or "").strip() for w in line]
    if any(_WORDLIKE_TOKEN_RE.search(t) for t in texts):
        return False
    height = stats["bottom"] - stats["top"]
    tall = height >= 1.5 * median_height
    has_stray_symbol = any("$" in t for t in texts)
    multi_fragment = len(texts) >= 2
    return tall or has_stray_symbol or multi_fragment

def _find_equation_line_groups(lines: List[List[OCRWord]]) -> List[List[List[OCRWord]]]:
    """Group consecutive equation-like lines (see _line_is_equation_like)
    into equation regions. Usually one line per region in practice -- a
    stacked fraction typically collapses into a single OCR-grouped line --
    but adjacent equation-like lines are merged so a region formatted
    differently isn't split into several back-to-back placeholders."""
    real_lines = [ln for ln in lines if ln]
    if not real_lines:
        return []
    heights = [_line_stats(ln)["bottom"] - _line_stats(ln)["top"] for ln in real_lines]
    widths = [_line_stats(ln)["x2"] - _line_stats(ln)["x1"] for ln in real_lines]
    median_height = max(1.0, median(heights))
    median_width = max(1.0, median(widths))

    groups: List[List[List[OCRWord]]] = []
    current: List[List[OCRWord]] = []
    for ln in real_lines:
        if _line_is_equation_like(ln, median_height, median_width):
            current.append(ln)
        else:
            if current:
                groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups

def _replace_equation_groups_with_placeholders(page_words: List[OCRWord],
                                                groups: List[List[List[OCRWord]]],
                                                page_index: int) -> List[OCRWord]:
    """Remove each equation region's words and splice in one synthetic
    placeholder OCRWord per region, positioned at the region's bounding
    box. Downstream code (mark_sup_sub_lines, paragraph reconstruction)
    only needs page/bbox/text on an OCRWord and re-sorts by position, so
    the placeholder needs no special-casing anywhere else in the pipeline.
    """
    if not groups:
        return page_words
    remove_ids = set()
    placeholders: List[OCRWord] = []
    for i, group in enumerate(groups, 1):
        group_words = [w for ln in group for w in ln]
        if not group_words:
            continue
        remove_ids.update(id(w) for w in group_words)
        x1 = min(w.bbox[0] for w in group_words)
        y1 = min(w.bbox[1] for w in group_words)
        x2 = max(w.bbox[2] for w in group_words)
        y2 = max(w.bbox[3] for w in group_words)
        suffix = f" {i}" if len(groups) > 1 else ""
        label = f"[EQUATION{suffix} - VERIFY MANUALLY, PAGE {page_index+1}]"
        placeholders.append(OCRWord(page=page_index, text=label, conf=1.0, bbox=(x1, y1, x2, y2)))
    kept = [w for w in page_words if id(w) not in remove_ids]
    return kept + placeholders

def find_abstract_page(doc, max_first_pages=15) -> Optional[int]:
   """Find the abstract page using fast PDF text extraction only."""
   page_count = min(max_first_pages, doc.page_count)
   print(f"  Searching for abstract heading in first {page_count} pages...")

   # Try PDF text extraction first, then OCR fallback if MuPDF chokes on structure trees.
   page_lines = []
   for i in range(page_count):
       text = safe_page_text(doc, i)
       if not text.strip():
           page_lines.append([])
           continue  # Skip blank pages
       page_lines.append([ln.strip() for ln in text.splitlines() if ln.strip()])

   # Honor heading priority order (first match wins).
   for heading in VALID_ABSTRACT_HEADINGS:
       heading_re = _heading_regex(heading)
       for i, lines in enumerate(page_lines):
           for ln in lines[:100]:  # Check more lines for flexibility
               if heading_re.match(ln):
                   print(f"  ✓ Found {heading} on page {i+1}")
                   return i

   # Fallback to fuzzy matches only if no exact hits.
   for heading in VALID_ABSTRACT_HEADINGS:
       for i, lines in enumerate(page_lines):
           for ln in lines[:100]:  # Check more lines for flexibility
               if _fuzzy_heading_match(ln, heading):
                   print(f"  ✓ Found near-match for {heading} on page {i+1}")
                   return i

   # If not found via text extraction then the pdf doesn't have ocr
   print("  ⚠ Could not find abstract heading via text extraction.")
   print("     Specify page manually if needed.")
   return None


def safe_page_text(doc, page_index: int, confidence_threshold: float = 0.60) -> str:
    """Safely extract text from a page, falling back to OCR if MuPDF text extraction fails."""
    try:
        return doc.load_page(page_index).get_text("text")
    except Exception as exc:
        print(f"  ⚠ Falling back to OCR for page {page_index+1} text extraction")

    try:
        words = ocr_page_isolated(Path(doc.name), page_index, dpi=220, confidence_threshold=confidence_threshold)
        return " ".join(w.text for w in words if w.text).strip()
    except Exception as exc:
        print(f"  ⚠ OCR fallback failed on page {page_index+1}: {exc}")
        return ""

def avg(vals): return sum(vals)/len(vals) if vals else 0.0
def w_center_y(w: OCRWord): return (w.bbox[1]+w.bbox[3])/2

def group_words_into_lines(words: List[OCRWord], y_tol_ratio=0.03): 
    # Simple y-banding by average height
    if not words: return []
    heights = [(w.bbox[3]-w.bbox[1]) for w in words]
    avg_h = max(1, int(avg(heights)))
    y_tol = max(4, int(avg_h * (1.0 + y_tol_ratio)))
    words_sorted = sorted(words, key=lambda w: (w.bbox[1], w.bbox[0]))
    lines = []
    current = []
    cur_y = None
    for w in words_sorted:
        y = (w.bbox[1]+w.bbox[3])//2
        if cur_y is None or abs(y-cur_y) <= y_tol:
            current.append(w); cur_y = y if cur_y is None else (cur_y + y)//2
        else:
            lines.append(sorted(current, key=lambda ww: ww.bbox[0]))
            current = [w]; cur_y = y
    if current:
        lines.append(sorted(current, key=lambda ww: ww.bbox[0]))
    return lines

def extract_abstract_region(doc, start_page: int, max_pages=4, confidence_threshold: float = 0.60, pdf_path: Optional[Path] = None,
                             vlm_review_mode: Optional[str] = None, vlm_trigger_threshold: int = DEFAULT_VLM_TRIGGER_THRESHOLD,
                             vlm_diff_review: bool = False):
    """Return raw OCR words (with boxes) from start_page until a stop heading or max_pages.

    Args:
        doc: PDF document
        start_page: Starting page index (0-based)
        max_pages: Maximum pages to extract
        confidence_threshold: Filter out OCR detections below this confidence (0.0-1.0).
                             Increase to filter more noise, decrease to catch more text.
        pdf_path: Path to the PDF on disk, needed to re-open it in the isolated
                  per-page OCR subprocess (see ocr_page_isolated).
        vlm_review_mode: None (off, default), "targeted" (only pages that look
                  suspicious -- see _page_is_suspicious), or "always" (every page).
                  NOTE: no VLM call is made here -- this only *flags* pages
                  (a cheap, local, no-network decision). See the "Optional VLM
                  review pass" section for why the actual calls are deferred
                  to main(), strictly after all PaddleOCR work in the batch.
        vlm_trigger_threshold: see _page_is_suspicious. Only used when
                  vlm_review_mode == "targeted".

    Returns (collected_words, end_page, vlm_flagged_pages, vlm_diff_pages,
    page_ocr_texts). vlm_flagged_pages is a list of (page_index, reason)
    pairs -- reason is "low_confidence" or "equation" (see
    ENABLE_EQUATION_PLACEHOLDERS). Equation-region detection/placeholder
    substitution always runs (independent of vlm_review_mode); pages only
    get added to vlm_flagged_pages when vlm_review_mode is set.

    vlm_diff_pages/page_ocr_texts are for the separate --vlm-diff-review
    pass (see that section): when vlm_diff_review is True, EVERY page in
    range is added to vlm_diff_pages unconditionally (no suspicion check --
    this is a full second pass, not a targeted one), and page_ocr_texts
    captures that page's own OCR text for the deferred diff against the
    VLM's independent transcription of the same page. Kept architecturally
    separate from vlm_flagged_pages -- these are two independent systems.
    """
    collected = []
    end_page = start_page
    src_path = pdf_path if pdf_path is not None else Path(doc.name)

    vlm_flagged_pages: dict = {}  # page_index -> reason ("low_confidence" or "equation")
    vlm_diff_pages: List[int] = []
    page_ocr_texts: dict = {}  # page_index -> plain OCR text, for --vlm-diff-review
    prior_segment_counts: List[int] = []

    print(f"  Extracting text from pages {start_page+1} to {min(start_page+max_pages, doc.page_count)}...")
    print(f"  (Confidence threshold: {confidence_threshold*100:.0f}%, filtering low-confidence detections)")

    for p in range(start_page, min(start_page+max_pages, doc.page_count)):
        print(f"    Processing page {p+1} with PaddleOCR...", end='', flush=True)
        page_words = ocr_page_isolated(src_path, p, dpi=350, confidence_threshold=confidence_threshold)
        print(f" done ({len(page_words)} text segments found)")

        if vlm_review_mode:
            suspicious = vlm_review_mode == "always" or _page_is_suspicious(len(page_words), prior_segment_counts, vlm_trigger_threshold)
            if suspicious:
                print(f"    Page {p+1} flagged for deferred VLM review ({len(page_words)} segments)")
                vlm_flagged_pages[p] = "low_confidence"
            prior_segment_counts.append(len(page_words))

        # mark page index
        for w in page_words: w.page = p

        # Content bounds for this page, used to spot isolated header/footer
        # page-number words (Arabic or roman numerals) in the top/bottom margin.
        if page_words:
            page_y_min = min(w.bbox[1] for w in page_words)
            page_y_max = max(w.bbox[3] for w in page_words)
        else:
            page_y_min, page_y_max = 0.0, 1.0
        page_span = max(1.0, page_y_max - page_y_min)

        # Drop margin page-number LINES (not individual words) before further
        # line-grouping. Checking isolation at the line level -- rather than
        # relying on _is_margin_page_number_word's text-pattern + margin-band
        # check alone -- matters when an abstract's last paragraph ends near
        # the bottom of the page: isotope numbers ("241AmO2"), exponents
        # ("10-5"), and "M," all match the bare page-number pattern and sit
        # in that same bottom margin band, but they share their line with
        # real body text, so they must never be dropped just for that.
        pre_lines = group_words_into_lines(page_words)
        dropped_ids = set()
        for ln in pre_lines:
            if not ln or not _line_is_isolated_page_number(ln):
                continue
            if _is_margin_page_number_word(ln[0], page_y_min, page_span):
                dropped_ids.update(id(w) for w in ln)
        page_words = [w for w in page_words if id(w) not in dropped_ids]

        # See "Equation-region placeholder" section: replace anything that
        # looks like a stacked-fraction equation with an explicit
        # placeholder before it ever reaches the normal text pipeline.
        if ENABLE_EQUATION_PLACEHOLDERS:
            eq_groups = _find_equation_line_groups(group_words_into_lines(page_words))
            if eq_groups:
                print(f"    Page {p+1}: {len(eq_groups)} equation region(s) replaced with placeholder")
                page_words = _replace_equation_groups_with_placeholders(page_words, eq_groups, p)
                if vlm_review_mode:
                    vlm_flagged_pages[p] = "equation"

        lines = group_words_into_lines(page_words)

        # If we hit a stop heading, terminate
        stop = False
        filtered_words = []
        pre_stop_lines = []
        for ln in lines:
            text = " ".join(w.text for w in ln).strip()
            if STOP_HEADINGS_RE.match(text):
                stop = True
                break
            filtered_words.extend(ln)
            pre_stop_lines.append(ln)
        collected.extend(filtered_words)

        # See "Full OCR/VLM diff pass" section: capture only the part of
        # this page that's actually abstract content -- i.e. up to the same
        # stop-heading boundary the primary text already respects (a hard
        # content-boundary fact, not the "suspicion" heuristic this pass is
        # meant to bypass). A page entirely past that boundary (e.g. a
        # Table of Contents page) contributes nothing and isn't reviewed --
        # confirmed necessary: without this, Thesis76.K82's TOC page got
        # diff-reviewed and produced a page of page-number/dot-leader noise.
        if vlm_diff_review and pre_stop_lines:
            page_ocr_texts[p] = " ".join(" ".join(w.text for w in ln) for ln in pre_stop_lines)
            vlm_diff_pages.append(p)
        end_page = p
        if stop:
            break

    return collected, end_page, list(vlm_flagged_pages.items()), vlm_diff_pages, page_ocr_texts

# ---- Superscript/Subscript detection by baseline ----
def baseline_clusters(words: List[OCRWord], band_px=8):
    """Group by approximate y-baseline."""
    lines = group_words_into_lines(words)
    clusters = []
    for ln in lines:
        # compute baseline y as median of bottom edges
        bottoms = [w.bbox[3] for w in ln]
        baseline = median(bottoms)
        clusters.append((ln, baseline))
    return clusters

def median(lst):
    s = sorted(lst)
    n = len(s)
    return (s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2) if n else 0

def _word_height(w: OCRWord) -> float:
    return max(1.0, float(w.bbox[3] - w.bbox[1]))


def _group_words_for_supsub(words: List[OCRWord]) -> List[List[OCRWord]]:
    """Group words into text lines while keeping shifted script tokens in-line."""
    if not words:
        return []

    pages = {}
    for w in words:
        pages.setdefault(w.page, []).append(w)

    out_lines: List[List[OCRWord]] = []
    for page_num in sorted(pages.keys()):
        page_words = pages[page_num]
        page_heights = [_word_height(w) for w in page_words]
        page_med_h = max(1.0, median(page_heights))
        y_tol = max(5.0, 0.65 * page_med_h)

        page_words_sorted = sorted(page_words, key=lambda w: (w_center_y(w), w.bbox[0]))
        line_bins: List[List[OCRWord]] = []
        line_centers: List[float] = []

        for w in page_words_sorted:
            cy = w_center_y(w)
            best_idx = None
            best_dist = None
            for idx, line_center in enumerate(line_centers):
                dist = abs(cy - line_center)
                if dist <= y_tol and (best_dist is None or dist < best_dist):
                    best_idx = idx
                    best_dist = dist

            if best_idx is None:
                line_bins.append([w])
                line_centers.append(cy)
            else:
                line_bins[best_idx].append(w)
                line_centers[best_idx] = avg([w_center_y(item) for item in line_bins[best_idx]])

        normalized = [sorted(ln, key=lambda item: item.bbox[0]) for ln in line_bins if ln]
        normalized.sort(key=lambda ln: (median([w_center_y(w) for w in ln]), ln[0].bbox[0]))
        out_lines.extend(normalized)

    return out_lines


def _likely_script_token(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 4:
        return False
    if re.fullmatch(r"[^\w]+", t):
        return False
    return any(ch.isalnum() for ch in t)


def _looks_like_page_number(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    compact = re.sub(r"[^A-Za-z0-9]+", "", t)
    if not compact:
        return False
    if compact.isdigit() and len(compact) <= 4:
        return True
    if re.fullmatch(r"[ivxlcdm]+", compact, re.IGNORECASE):
        return True
    # Allow "1" to be misrecognized as "i" in roman numerals, but only for short sequences
    if len(compact) <= 5:
        normalized = compact.replace("1", "i")
        if re.fullmatch(r"[ivxlcdm]+", normalized, re.IGNORECASE):
            return True
    return False


def _is_margin_page_number_word(w: OCRWord, page_y_min: float, page_span: float,
                                 margin_ratio: float = 0.08) -> bool:
    """True if `w` looks like a running-header/footer page number (Arabic
    digits like "5" or roman numerals like "iv"/"V") and sits in the page's
    top or bottom margin band, rather than actual abstract body text.

    This only checks the text pattern and vertical position -- it does NOT
    by itself confirm the word is isolated from real body text. Callers must
    pair this with a line-level isolation check (see
    `_line_is_isolated_page_number`) before dropping anything: when an
    abstract's last paragraph ends near the bottom of a page, isotope mass
    numbers ("241" in "241AmO2"), exponents ("-5" in "10-5"), and even bare
    "M" (a valid roman numeral for 1000) all match this pattern and sit in
    that same margin band, even though they are plainly part of a real
    sentence, not a standalone footer number.
    """
    if page_span <= 0:
        return False
    text = (w.text or "").strip()
    if not _looks_like_page_number(text):
        return False
    center_y = w_center_y(w)
    rel = (center_y - page_y_min) / page_span
    return rel <= margin_ratio or rel >= (1.0 - margin_ratio)


def _line_is_isolated_page_number(line_words: List[OCRWord]) -> bool:
    """True if every word on this line looks like a bare page number, i.e.
    the line has no real body-text content sharing it.

    A genuine running header/footer page number sits alone on its own line
    ("iii", "5"). A superscript/subscript that merely happens to match the
    same digit/roman-numeral pattern (isotope numbers, exponents, "M" for
    molarity) instead shares its line with ordinary body words -- so this
    line-level check, not just the word's own text, is what actually
    distinguishes the two cases.
    """
    if not line_words:
        return False
    return all(_looks_like_page_number((w.text or "").strip()) for w in line_words)


def _line_stats(line: List[OCRWord]) -> dict:
    xs1 = [w.bbox[0] for w in line]
    ys1 = [w.bbox[1] for w in line]
    xs2 = [w.bbox[2] for w in line]
    ys2 = [w.bbox[3] for w in line]
    heights = [_word_height(w) for w in line]
    return {
        "line": sorted(line, key=lambda w: w.bbox[0]),
        "center_y": median([w_center_y(w) for w in line]),
        "top": median(ys1),
        "bottom": median(ys2),
        "height": max(1.0, median(heights)),
        "x1": min(xs1),
        "x2": max(xs2),
    }


def _nearest_line_stats(target: dict, candidates: List[dict]) -> Optional[dict]:
    best = None
    best_score = None
    for candidate in candidates:
        if candidate is target:
            continue
        y_dist = abs(candidate["center_y"] - target["center_y"])
        x_gap = 0.0
        if target["x2"] < candidate["x1"]:
            x_gap = candidate["x1"] - target["x2"]
        elif candidate["x2"] < target["x1"]:
            x_gap = target["x1"] - candidate["x2"]
        overlap = max(0.0, min(target["x2"], candidate["x2"]) - max(target["x1"], candidate["x1"]))
        score = y_dist + (x_gap * 0.35) - (overlap * 0.1)
        if best_score is None or score < best_score:
            best = candidate
            best_score = score
    return best


def _is_formula_script_candidate(text: str) -> bool:
    """Conservative candidate gate for chemistry-like script fragments.

    We intentionally bias toward short numeric fragments to avoid tagging regular words.
    """
    t = (text or "").strip()
    if not t:
        return False
    compact = re.sub(r"\s+", "", t)
    if re.fullmatch(r"\d{1,3}", compact):
        return True
    if re.fullmatch(r"\d{1,2}[+-]", compact):
        return True
    if compact in {"+", "-"}:
        return True
    return False

def _has_alpha(text: str) -> bool:
    return any(ch.isalpha() for ch in (text or ""))

def mark_sup_sub_lines(words: List[OCRWord], position_only: bool = True) -> List[List[Tuple[OCRWord, Optional[str], bool]]]:
    """Simple, conservative detector for obvious scripts.

    Rules:
    - Only very short numeric/charge-like tokens can be tagged.
    - Token must be visually attached to an alphabetic neighbor.
    - Must show clear vertical offset plus smaller height.
    """
    lines = _group_words_for_supsub(words)
    if not lines:
        return []

    # Per-page bounds to suppress footer/header page-number artifacts.
    page_bounds: dict[int, Tuple[float, float]] = {}
    for w in words:
        p = w.page
        if p not in page_bounds:
            page_bounds[p] = (float(w.bbox[1]), float(w.bbox[3]))
        else:
            y_min, y_max = page_bounds[p]
            page_bounds[p] = (min(y_min, w.bbox[1]), max(y_max, w.bbox[3]))

    marked_lines: List[List[Tuple[OCRWord, Optional[str], bool]]] = []
    for ln in lines:
        ln_sorted = sorted(ln, key=lambda w: w.bbox[0])
        if not ln_sorted:
            continue

        page = ln_sorted[0].page
        y_min, y_max = page_bounds.get(page, (0.0, 1.0))
        page_h = max(1.0, y_max - y_min)

        heights = [_word_height(w) for w in ln_sorted]
        median_h = max(1.0, median(heights))
        center_y = median([w_center_y(w) for w in ln_sorted])

        gaps = []
        for i in range(len(ln_sorted) - 1):
            g = ln_sorted[i + 1].bbox[0] - ln_sorted[i].bbox[2]
            if g >= 0:
                gaps.append(g)
        attach_gap = max(2.0, min(12.0, 1.4 * (median(gaps) if gaps else 4.0)))

        line_marked: List[Tuple[OCRWord, Optional[str], bool]] = []
        glue_next = False
        for i, w in enumerate(ln_sorted):
            txt = (w.text or "").strip()
            tag = None
            glue_before = glue_next
            glue_next = False

            # Note: intentionally NOT excluding tokens that "look like a page
            # number" here. Sub/superscript candidates are short digit runs
            # (e.g. "2" in "R2"), which is exactly what _looks_like_page_number
            # also matches -- excluding them here would suppress virtually
            # every numeric script. Running-header/footer page numbers are
            # filtered separately (by margin position, before line-grouping,
            # in _is_margin_page_number_word) and by the header/footer band
            # check just below, and a real page number won't sit attached to
            # an alphabetic neighbor the way a formula script does.
            if txt and _is_formula_script_candidate(txt):
                cy = w_center_y(w)
                h = _word_height(w)

                # Skip likely headers/footers.
                if (cy - y_min) / page_h > 0.10 and (cy - y_min) / page_h < 0.90:
                    left = ln_sorted[i - 1] if i > 0 else None
                    right = ln_sorted[i + 1] if i + 1 < len(ln_sorted) else None

                    left_ok = False
                    right_ok = False
                    left_gap = None
                    right_gap = None
                    if left is not None and _has_alpha(left.text):
                        left_gap = w.bbox[0] - left.bbox[2]
                        left_ok = left_gap <= attach_gap
                    if right is not None and _has_alpha(right.text):
                        right_gap = right.bbox[0] - w.bbox[2]
                        right_ok = right_gap <= attach_gap

                    attached_to_alpha = left_ok or right_ok
                    size_reduced = h <= 0.92 * median_h
                    vertical_delta = cy - center_y
                    obvious_sub = vertical_delta >= 0.16 * median_h
                    obvious_sup = vertical_delta <= -0.16 * median_h

                    if attached_to_alpha and size_reduced:
                        if obvious_sub:
                            tag = "sub"
                        elif obvious_sup:
                            tag = "sup"

                        if tag:
                            # Glue the tagged token to whichever neighbor it's
                            # actually attached to, so "R" + "<sub>2</sub>"
                            # renders as "R<sub>2</sub>" instead of "R <sub>2</sub>".
                            if left_ok and (not right_ok or (left_gap or 0) <= (right_gap or 0)):
                                glue_before = True
                            elif right_ok:
                                glue_next = True

            line_marked.append((w, tag, glue_before))

        marked_lines.append(line_marked)

    return marked_lines


def mark_sup_sub(words: List[OCRWord], position_only: bool = True):
    """Backwards-compatible flattened output wrapper."""
    marked_lines = mark_sup_sub_lines(words, position_only=position_only)
    return [item for line in marked_lines for item in line]

# ---- HTML Conversion ----
def escape_user_content(text: str) -> str:
    """Escape user content while preserving our intentional HTML tags and entities."""
    # First, temporarily protect our intentional tags and entities
    import re
    
    # Protect HTML entities (like &alpha;, &beta;, &#xNNNN;)
    entity_pattern = r'&[#A-Za-z0-9]+;'
    entities = re.findall(entity_pattern, text)
    entity_placeholders = {}
    for i, entity in enumerate(set(entities)):
        placeholder = f"___ENTITY_{i}___"
        entity_placeholders[placeholder] = entity
        text = text.replace(entity, placeholder)
    
    # Protect our sup/sub/italic tags
    text = text.replace("<sup>", "___SUP_OPEN___")
    text = text.replace("</sup>", "___SUP_CLOSE___")
    text = text.replace("<sub>", "___SUB_OPEN___")
    text = text.replace("</sub>", "___SUB_CLOSE___")
    text = text.replace("<i>", "___I_OPEN___")
    text = text.replace("</i>", "___I_CLOSE___")

    # NOW escape any remaining < and > (which are user content)
    text = text.replace("<", "&lt;").replace(">", "&gt;")

    # Restore our protected tags
    text = text.replace("___SUP_OPEN___", "<sup>")
    text = text.replace("___SUP_CLOSE___", "</sup>")
    text = text.replace("___SUB_OPEN___", "<sub>")
    text = text.replace("___SUB_CLOSE___", "</sub>")
    text = text.replace("___I_OPEN___", "<i>")
    text = text.replace("___I_CLOSE___", "</i>")
    
    # Restore entities
    for placeholder, entity in entity_placeholders.items():
        text = text.replace(placeholder, entity)
    
    return text

def escape_angle(text: str) -> str:
    return escape_user_content(text)

def escape_ampersands_not_entities(text: str) -> str:
    return re.sub(r"&(?!#?[A-Za-z0-9]+;)", "&amp;", text)

def replace_greek_math(text: str) -> str:
    out = []
    for ch in text:
        if ch in GREEK_MAP: out.append(GREEK_MAP[ch])
        elif ch in MATH_MAP: out.append(MATH_MAP[ch])
        else: out.append(ch)
    return "".join(out)

def convert_remaining_non_ascii(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if code > 126:
            out.append(f"&#x{code:X};")
        else:
            out.append(ch)
    return "".join(out)

def _marked_line_to_record(marked_line: List[Tuple[OCRWord, Optional[str], bool]]) -> Optional[dict]:
    """Convert one OCR line into text + layout metadata."""
    if not marked_line:
        return None

    text = ""
    words = []
    for w, tag, glue_before in marked_line:
        t = (w.text or "").strip()
        if not t:
            continue
        if tag == "sup":
            t = f"<sup>{t}</sup>"
        elif tag == "sub":
            t = f"<sub>{t}</sub>"
        if text and glue_before:
            text += t
        elif text:
            text += " " + t
        else:
            text = t
        words.append(w)

    if not text or not words:
        return None

    return {
        "text": text,
        "page": words[0].page,
        "x1": float(min(w.bbox[0] for w in words)),
        "top": float(min(w.bbox[1] for w in words)),
        "bottom": float(max(w.bbox[3] for w in words)),
        "height": max(1.0, float(max(w.bbox[3] for w in words) - min(w.bbox[1] for w in words))),
    }


def _join_wrapped_lines(line_texts: List[str]) -> str:
    """Flatten OCR hard-wraps into a single clean paragraph line."""
    merged = ""
    for line in line_texts:
        piece = re.sub(r"\s+", " ", (line or "").strip())
        if not piece:
            continue
        if not merged:
            merged = piece
            continue

        # Rejoin split words like "inter-" + "national".
        if merged.endswith("-") and piece[0].islower():
            merged = merged[:-1] + piece
        else:
            merged = f"{merged} {piece}"

    return re.sub(r"\s+", " ", merged).strip()


def _paragraphize_marked_lines(marked_lines: List[List[Tuple[OCRWord, Optional[str], bool]]]) -> List[str]:
    """Split OCR lines into paragraphs using vertical gaps and indentation."""
    line_records = []
    for marked_line in marked_lines:
        rec = _marked_line_to_record(marked_line)
        if rec:
            line_records.append(rec)

    if not line_records:
        return []

    page_metrics = {}
    by_page = {}
    for rec in line_records:
        by_page.setdefault(rec["page"], []).append(rec)

    for page, recs in by_page.items():
        sorted_recs = sorted(recs, key=lambda r: (r["top"], r["x1"]))
        gaps = []
        for i in range(len(sorted_recs) - 1):
            gaps.append(max(0.0, sorted_recs[i + 1]["top"] - sorted_recs[i]["bottom"]))

        positive_gaps = [g for g in gaps if g > 0.0]
        heights = [r["height"] for r in sorted_recs]
        x1s = [r["x1"] for r in sorted_recs]

        page_metrics[page] = {
            "typical_gap": median(positive_gaps) if positive_gaps else 0.0,
            "median_height": max(1.0, median(heights)) if heights else 1.0,
            "left_margin": min(x1s) if x1s else 0.0,
        }

    paragraphs = []
    current_lines = [line_records[0]["text"]]

    for idx in range(len(line_records) - 1):
        current = line_records[idx]
        nxt = line_records[idx + 1]

        split_here = False
        if current["page"] != nxt["page"]:
            # New page, so always split. But check if the next line is indented enough to be a new paragraph.
            metrics = page_metrics.get(nxt["page"], {"typical_gap": 0.0, "median_height": 1.0, "left_margin": 0.0})
            indent = nxt["x1"] - metrics["left_margin"]
            indent_threshold = max(18.0, metrics["median_height"] * 0.9)
            if indent >= indent_threshold and re.search(r"[.!?;:]\s*$", current["text"]):
                split_here = True
        else:
            metrics = page_metrics.get(current["page"], {"typical_gap": 0.0, "median_height": 1.0, "left_margin": 0.0})
            gap = max(0.0, nxt["top"] - current["bottom"])

            gap_threshold = max(
                8.0,
                metrics["typical_gap"] * 1.9 if metrics["typical_gap"] > 0 else metrics["median_height"] * 1.5,
            )
            if gap > gap_threshold:
                split_here = True
            else:
                indent = nxt["x1"] - metrics["left_margin"]
                indent_threshold = max(18.0, metrics["median_height"] * 0.9)
                if indent >= indent_threshold and re.search(r"[.!?;:]\s*$", current["text"]):
                    split_here = True

        if split_here:
            paragraph = _join_wrapped_lines(current_lines)
            if paragraph:
                paragraphs.append(paragraph)
            current_lines = []

        current_lines.append(nxt["text"])

    trailing = _join_wrapped_lines(current_lines)
    if trailing:
        paragraphs.append(trailing)

    return paragraphs


def to_html_paragraphs(paragraphs: List[str]) -> str:
    return "\n".join(f"<p>{p}</p>" for p in paragraphs if p.strip())

# ---- Main processing ----
def process_pdf(pdf_path: Path, out_dir: Path, overrides: dict, max_first_pages: int, confidence_threshold: float = 0.60, force_single_paragraph: bool = False,
                 vlm_review_mode: Optional[str] = None, vlm_trigger_threshold: int = DEFAULT_VLM_TRIGGER_THRESHOLD,
                 vlm_diff_review: bool = False):
    """Process a PDF and extract abstract text.

    Args:
        pdf_path: Path to PDF file
        out_dir: Output directory
        overrides: Override page numbers from CSV (optional, will auto-detect if empty)
        max_first_pages: Max pages to search for abstract
        confidence_threshold: Filter out OCR detections below this confidence (0.0-1.0).
                             Default 0.60. Increase to filter more noise.
        force_single_paragraph: Merge all extracted text into one paragraph.
        vlm_review_mode, vlm_trigger_threshold: see extract_abstract_region.
                             vlm_review_mode is None (off) by default -- this is an
                             opt-in feature. No VLM call happens in this function or
                             its subprocess -- flagged pages are written to a sidecar
                             manifest for main()'s deferred VLM phase to pick up after
                             every PDF in the batch has finished its PaddleOCR work
                             (see the "Optional VLM review pass" section for why).
        vlm_diff_review: opt-in, off by default. Independent of vlm_review_mode --
                             see the "Full OCR/VLM diff pass" section. Also writes its
                             own separate sidecar for main()'s deferred phase; never
                             calls the VLM from here either, same reasoning as above.
    """

    print(f"\n{'='*70}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*70}")

    with fitz.open(pdf_path) as doc:
        print(f"  PDF loaded: {doc.page_count} pages")

        # Learn correct casing for mixed-case abbreviations (dNMP, mRNA, ...)
        # from this thesis's own embedded text layer, so PaddleOCR's
        # case-mangling can be corrected without a hand-maintained term list.
        casing_reference = build_casing_reference(doc)

        start_p = None
        end_p = None
        
        # PRIMARY: Check overrides first
        if pdf_path.name in overrides:
            s, e = overrides[pdf_path.name]
            start_p = max(0, s - 1)
            end_p = max(start_p, e - 1)
            print(f"  ✓ Using pages from override-csv: {s}-{e}")
        else:
            # If no override, try to auto-detect abstract using find_abstract_page
            print(f"  No override found. Attempting to auto-detect abstract...")
            abstract_page = find_abstract_page(doc, max_first_pages=max_first_pages)
            if abstract_page is not None:
                start_p = abstract_page
                # For end page, assume abstract is typically 3-5 pages long
                end_p = min(abstract_page + 4, doc.page_count - 1)
                print(f"  ✓ Auto-detected abstract start at page {start_p+1}, using pages {start_p+1}-{end_p+1}")
            else:
                print(f"  ⊘ Skipped - could not find abstract (not in overrides and auto-detection failed)")
                return


        print(f"\n  Starting OCR extraction (this may take a moment)...")
        words, end_idx, vlm_flagged_pages, vlm_diff_pages, page_ocr_texts = extract_abstract_region(
            doc, start_p, max_pages=(end_p-start_p+1), confidence_threshold=confidence_threshold, pdf_path=pdf_path,
            vlm_review_mode=vlm_review_mode, vlm_trigger_threshold=vlm_trigger_threshold, vlm_diff_review=vlm_diff_review,
        )
        if not words:
            print(f"  ⚠ No text extracted")
            return
        print(f"  ✓ Extracted {len(words)} text segments")


        # Superscript/subscript tagging from baselines
        print(f"  Analyzing text layout (superscripts/subscripts)...")
        marked_lines = mark_sup_sub_lines(words, position_only=True)

        paragraphs = _paragraphize_marked_lines(marked_lines)
        if not paragraphs:
            print(f"  ⚠ No paragraph text reconstructed")
            return

        if force_single_paragraph:
            merged = _join_wrapped_lines(paragraphs)
            paragraphs = [merged] if merged else []
            if not paragraphs:
                print(f"  ⚠ No paragraph text reconstructed")
                return

        html_paragraphs = []
        for paragraph in paragraphs:
            t = fix_zero_o_confusion(paragraph)
            t = fix_double_periods(t)
            t = fix_stray_period_before_lowercase(t)
            t = fix_degree_celsius(t)
            t = fix_scientific_exponent_notation(t)
            t = fix_known_term_casing(t, casing_reference)
            t = apply_known_science_notation(t)
            t = replace_greek_math(t)
            t = escape_user_content(t)
            t = escape_ampersands_not_entities(t)
            t = convert_remaining_non_ascii(t)
            html_paragraphs.append(t)

        html_text = to_html_paragraphs(html_paragraphs)
        html_text = html_text.replace("\n", "").replace("\r", "")

        # Write HTML output
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{pdf_path.stem} draft.txt").write_text(html_text, encoding="utf-8")

        # Pages flagged for VLM review (see extract_abstract_region) are NOT
        # reviewed here -- leave a sidecar manifest for main()'s deferred VLM
        # phase, which runs strictly after every PDF in the batch has finished
        # its PaddleOCR work (see the "Optional VLM review pass" section for
        # why: an Ollama model resident on the GPU has been observed to break
        # PaddleOCR's own init for any call made while it's loaded).
        if vlm_review_mode and vlm_flagged_pages:
            sidecar = out_dir / f"{pdf_path.stem} draft.vlm_pending.json"
            # vlm_flagged_pages is a list of [page_index, reason] pairs.
            sidecar.write_text(json.dumps({"pdf_path": str(pdf_path), "flagged_pages": vlm_flagged_pages}), encoding="utf-8")

        # Separate, independent sidecar for the --vlm-diff-review pass (see
        # the "Full OCR/VLM diff pass" section) -- own file, own manifest
        # shape, own deferred-phase handler. Not merged with the sidecar
        # above: these are two parallel systems, not one extending the other.
        if vlm_diff_review and vlm_diff_pages:
            diff_sidecar = out_dir / f"{pdf_path.stem} draft.diff_pending.json"
            diff_sidecar.write_text(json.dumps({
                "pdf_path": str(pdf_path),
                "pages": vlm_diff_pages,
                "page_ocr_texts": {str(p): page_ocr_texts.get(p, "") for p in vlm_diff_pages},
            }), encoding="utf-8")

        print(f"  ✓ SUCCESS")
        print(f"  Pages used: {start_p+1} to {end_idx+1}")

def load_overrides(path: Optional[Path]) -> dict:
    """Load legacy format: filename,pages (e.g., 'doc.pdf,5-8')"""
    if not path or not path.exists(): return {}
    m = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            fn = (row.get("filename") or "").strip()
            pages = (row.get("pages") or "").strip()
            if not fn or not pages: continue
            if "-" in pages:
                a,b = pages.split("-",1); m[fn]=(int(a),int(b))
            else:
                p=int(pages); m[fn]=(p,p)
    return m

def main():
    import argparse

    # Internal/hidden: used when this script re-invokes itself as a child
    # process for exactly one PDF (see the crash-isolation loop below).
    if "--ocr-page" in sys.argv:
        worker_parser = argparse.ArgumentParser()
        worker_parser.add_argument("--ocr-page", action="store_true")
        worker_parser.add_argument("--pdf", required=True)
        worker_parser.add_argument("--page", type=int, required=True)
        worker_parser.add_argument("--dpi", type=int, required=True)
        worker_parser.add_argument("--confidence-threshold", type=float, required=True)
        worker_parser.add_argument("--out-json", required=True)
        wargs = worker_parser.parse_args()
        run_ocr_page_worker(Path(wargs.pdf), wargs.page, wargs.dpi, wargs.confidence_threshold, Path(wargs.out_json))
        return

    parser = argparse.ArgumentParser(description="Extract abstracts from PDFs using Paddle OCR.")
    parser.add_argument("--input", required=True, help="Directory with PDFs")
    parser.add_argument("--out", required=True, help="Output directory for HTML files")
    parser.add_argument("--override-csv", default=None, help="Optional CSV file with filename,pages (format: filename,start-end). If not provided, auto-detects abstract boundaries.")
    parser.add_argument("--confidence-threshold", type=float, default=0.60, help="Filter OCR below this confidence (0.0-1.0). Default: 0.60")
    parser.add_argument("--force-single-paragraph", action="store_true", help="Merge extracted abstract text into one <p> block.")
    # Internal/hidden: used when this script re-invokes itself as a child
    # process for exactly one PDF (see the crash-isolation loop below).
    # Not meant to be passed by hand.
    parser.add_argument("--single-pdf", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--vlm-review", action="store_true",
                         help="Opt-in: use a local Ollama vision model as a second-pass review for pages "
                              "that look suspicious (or all pages, with --vlm-review-mode always). Gracefully "
                              "skipped if Ollama/the model isn't available. Output lands as a separate, "
                              "clearly-labeled block per page for manual review, not spliced into the draft.")
    parser.add_argument("--vlm-review-mode", choices=["targeted", "always"], default="targeted",
                         help="targeted (default): only review pages with a suspiciously low OCR segment "
                              "count. always: review every abstract page. Only applies with --vlm-review.")
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL, help=f"Ollama model tag to use. Default: {DEFAULT_VLM_MODEL}")
    parser.add_argument("--vlm-trigger-threshold", type=int, default=DEFAULT_VLM_TRIGGER_THRESHOLD,
                         help=f"Segment-count floor below which a page is flagged for VLM review in targeted mode. Default: {DEFAULT_VLM_TRIGGER_THRESHOLD}")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help=f"Ollama server URL. Default: {DEFAULT_OLLAMA_URL}")
    parser.add_argument("--vlm-diff-review", action="store_true",
                         help="Opt-in, independent of --vlm-review: run the VLM on EVERY abstract page "
                              "(no suspicion check) and print a word-level diff against OCR's own text "
                              "at the bottom of the draft. Never modifies the primary draft text -- "
                              "purely a report for catching cases OCR got wrong without looking sparse "
                              "(e.g. a confidently-misread character, a dropped superscript) that "
                              "--vlm-review's segment-count trigger can't see. Requires Ollama/the model "
                              "the same way --vlm-review does.")
    parser.add_argument("--vlm-diff-merge", action="store_true",
                         help="Experimental, opt-in: like --vlm-diff-review (implies it -- no need to pass "
                              "both), but also applies the subset of diff spans judged safe directly into "
                              "the primary draft text (marked inline, never blended in silently). Only two "
                              "shapes are trusted -- a short pure insertion, and a short trailing addition "
                              "on text OCR already matched -- everything else (long insertions, character-"
                              "confusion substitutions, substitutions with no shared prefix) stays report-"
                              "only, same as --vlm-diff-review alone. See the diff-merge section for the "
                              "real-data reasoning behind the split.")
    args = parser.parse_args()

    vlm_review_mode = args.vlm_review_mode if args.vlm_review else None
    # --vlm-diff-merge builds on --vlm-diff-review's data collection --
    # either flag alone is enough to turn it on, so the user never needs
    # to pass both for the merge behavior to work.
    vlm_diff_review_enabled = args.vlm_diff_review or args.vlm_diff_merge

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load overrides if provided
    overrides = load_overrides(Path(args.override_csv)) if args.override_csv else {}

    if args.single_pdf:
        # If --single-pdf is provided, process just that one PDF and exit.
        # No VLM call happens here -- see process_pdf's docstring.
        process_pdf(
            Path(args.single_pdf),
            out_dir,
            overrides=overrides,
            max_first_pages=15,
            confidence_threshold=args.confidence_threshold,
            force_single_paragraph=args.force_single_paragraph,
            vlm_review_mode=vlm_review_mode,
            vlm_trigger_threshold=args.vlm_trigger_threshold,
            vlm_diff_review=vlm_diff_review_enabled,
        )
        return

    pdfs = sorted([p for p in in_dir.glob("**/*.pdf")])

    print(f"\n" + "="*70)
    print(f"PaddleOCR Abstract Extractor")
    print(f"="*70)
    print(f"Found {len(pdfs)} PDF(s) in {in_dir}")
    if args.override_csv:
        print(f"Override entries loaded: {len(overrides)}")
        print(f"Mode: Using override-csv for page detection")
    else:
        print(f"Mode: Using auto-detection to find abstract boundaries")
    print(f"Output directory: {out_dir}")
    print(f"Confidence threshold: {args.confidence_threshold*100:.0f}%")
    print(f"Force single paragraph: {'yes' if args.force_single_paragraph else 'no'}")
    print(f"="*70)

    # Process each PDF in isolation, retrying once if a crash occurs.
    failures = []
    for idx, pdf in enumerate(pdfs, 1):
        print(f"\n[{idx}/{len(pdfs)}] ", end="", flush=True)
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--input", str(in_dir),
            "--out", str(out_dir),
            "--single-pdf", str(pdf),
            "--confidence-threshold", str(args.confidence_threshold),
        ]
        if args.override_csv:
            cmd += ["--override-csv", args.override_csv]
        if args.force_single_paragraph:
            cmd += ["--force-single-paragraph"]
        if args.vlm_review:
            # Only what extract_abstract_region needs to *flag* pages (a local,
            # no-network decision) -- --vlm-model/--ollama-url aren't needed by
            # the child, since no VLM call is ever made from inside it (see the
            # "Optional VLM review pass" section for why).
            cmd += ["--vlm-review", "--vlm-review-mode", args.vlm_review_mode,
                    "--vlm-trigger-threshold", str(args.vlm_trigger_threshold)]
        if vlm_diff_review_enabled:
            # The child only needs to *collect* per-page OCR text (a local,
            # no-network decision, same as --vlm-review above) -- actual
            # merging happens in the deferred phase below, in this parent
            # process, so the child never needs to know about
            # --vlm-diff-merge specifically.
            cmd += ["--vlm-diff-review"]

        success = False
        for attempt in (1, 2):
            result = subprocess.run(cmd)
            if result.returncode == 0:
                success = True
                break
            print(f"  ⚠ Crashed (exit code {result.returncode}) processing {pdf.name}"
                  + (", retrying once..." if attempt == 1 else ", giving up after retry."))
        if not success:
            failures.append(pdf.name)

    print(f"\n\nProcessed {len(pdfs)} PDFs. Text files saved to {out_dir}")
    if failures:
        print(f"⚠ {len(failures)} PDF(s) failed even after retry: {', '.join(failures)}")

    # Deferred VLM review phase -- runs only now, strictly after every PDF's
    # PaddleOCR work (across the whole batch) is fully done. Never move this
    # earlier / interleave it with the loop above: see the "CRITICAL" note in
    # the "Optional VLM review pass" section for why that breaks PaddleOCR.
    if args.vlm_review:
        pending = sorted(out_dir.glob("*.vlm_pending.json"))
        if pending:
            print(f"\n{'='*70}")
            print(f"VLM review pass ({len(pending)} document(s) with flagged pages)")
            print(f"{'='*70}")
            vlm_ok = _vlm_available(args.vlm_model, args.ollama_url)
            if not vlm_ok:
                print(f"ℹ VLM review requested but unavailable (Ollama/{args.vlm_model} not reachable at {args.ollama_url}) -- skipping, drafts left as OCR-only output")
            for sidecar in pending:
                if vlm_ok:
                    try:
                        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
                        vlm_pdf_path = Path(manifest["pdf_path"])
                        vlm_flagged = manifest["flagged_pages"]  # [[page_index, reason], ...]
                        print(f"  {vlm_pdf_path.name}: reviewing {len(vlm_flagged)} flagged page(s)...", end="", flush=True)
                        with fitz.open(vlm_pdf_path) as vlm_doc:
                            vlm_casing_reference = build_casing_reference(vlm_doc)
                        vlm_blocks = []
                        for vlm_page, vlm_reason in vlm_flagged:
                            vlm_text = vlm_transcribe_page(vlm_pdf_path, vlm_page, model=args.vlm_model, ollama_url=args.ollama_url)
                            if vlm_text:
                                vlm_blocks.append((vlm_page, vlm_reason, vlm_text))
                        if vlm_blocks:
                            supplementary = _build_vlm_supplementary_html(vlm_blocks, vlm_casing_reference, args.vlm_model)
                            draft_path = out_dir / f"{vlm_pdf_path.stem} draft.txt"
                            if draft_path.exists():
                                existing = draft_path.read_text(encoding="utf-8")
                                draft_path.write_text(existing + "\n\n" + supplementary, encoding="utf-8")
                            print(f" done ({len(vlm_blocks)} recovered)")
                        else:
                            print(" done (none recovered)")
                    except Exception as e:
                        print(f" ⚠ failed: {e}")
                sidecar.unlink(missing_ok=True)

    # Deferred OCR/VLM diff phase -- see "Full OCR/VLM diff pass" section.
    # Independent of the --vlm-review phase above: own sidecar file, own
    # manifest shape, own output section. Same non-negotiable timing rule
    # applies (runs only after every PDF's PaddleOCR work is fully done).
    if vlm_diff_review_enabled:
        diff_pending = sorted(out_dir.glob("*.diff_pending.json"))
        if diff_pending:
            print(f"\n{'='*70}")
            print(f"OCR/VLM diff pass ({len(diff_pending)} document(s))")
            print(f"{'='*70}")
            vlm_ok = _vlm_available(args.vlm_model, args.ollama_url)
            if not vlm_ok:
                print(f"ℹ OCR/VLM diff pass requested but unavailable (Ollama/{args.vlm_model} not reachable at {args.ollama_url}) -- skipping, drafts left as OCR-only output")
            for sidecar in diff_pending:
                if vlm_ok:
                    try:
                        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
                        diff_pdf_path = Path(manifest["pdf_path"])
                        diff_pages = manifest["pages"]
                        diff_ocr_texts = manifest["page_ocr_texts"]  # {"page_index_str": ocr_text}
                        print(f"  {diff_pdf_path.name}: diffing {len(diff_pages)} page(s) against the VLM...", end="", flush=True)
                        diff_blocks = []
                        for diff_page in diff_pages:
                            vlm_text = vlm_transcribe_page(diff_pdf_path, diff_page, model=args.vlm_model, ollama_url=args.ollama_url)
                            if vlm_text:
                                diff_blocks.append((diff_page, diff_ocr_texts.get(str(diff_page), ""), vlm_text))
                        draft_path = out_dir / f"{diff_pdf_path.stem} draft.txt"
                        existing = draft_path.read_text(encoding="utf-8") if draft_path.exists() else ""
                        if args.vlm_diff_merge:
                            updated, report = _apply_vlm_diff_merges(existing, diff_blocks) if existing else ("", "")
                            # Count real per-span report lines only -- the section
                            # header's own explanatory text ("...[MERGED] lines were
                            # applied...") also contains the literal substring
                            # "[MERGED]" and would otherwise inflate this.
                            merged_count = sum(1 for line in report.splitlines() if line.strip().startswith("[MERGED]"))
                        else:
                            updated, report = existing, _build_vlm_diff_report_html(diff_blocks)
                            merged_count = 0
                        if report:
                            draft_path.write_text(updated + "\n\n" + report, encoding="utf-8")
                            print(f" done (differences found, {merged_count} merged inline)" if args.vlm_diff_merge
                                  else " done (differences found)")
                        else:
                            print(" done (no differences)")
                    except Exception as e:
                        print(f" ⚠ failed: {e}")
                sidecar.unlink(missing_ok=True)

    # See _unload_vlm_model: release the model now so it can't poison a
    # later run's PaddleOCR calls. Only relevant if a VLM phase actually
    # ran above (an unloaded model unload is a harmless no-op either way).
    if args.vlm_review or vlm_diff_review_enabled:
        _unload_vlm_model(args.vlm_model, args.ollama_url)

if __name__ == "__main__":
    main()
