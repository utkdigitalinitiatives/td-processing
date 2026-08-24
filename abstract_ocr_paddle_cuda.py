import warnings
# Suppress harmless dependency version warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*urllib3.*')
warnings.filterwarnings('ignore', message='.*chardet.*')

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
import numpy as np
import tqdm
from PIL import Image
import fitz  # PyMuPDF
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
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
        pix = render_page_image(doc, page_index, dpi=220)
        words = paddle_ocr_page(pix, confidence_threshold=confidence_threshold)
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

def extract_abstract_region(doc, start_page: int, max_pages=4, confidence_threshold: float = 0.60):
    """Return raw OCR words (with boxes) from start_page until a stop heading or max_pages.
    
    Args:
        doc: PDF document
        start_page: Starting page index (0-based)
        max_pages: Maximum pages to extract
        confidence_threshold: Filter out OCR detections below this confidence (0.0-1.0).
                             Increase to filter more noise, decrease to catch more text.
    """
    collected = []
    end_page = start_page
    
    print(f"  Extracting text from pages {start_page+1} to {min(start_page+max_pages, doc.page_count)}...")
    print(f"  (Confidence threshold: {confidence_threshold*100:.0f}%, filtering low-confidence detections)")

    for p in range(start_page, min(start_page+max_pages, doc.page_count)):
        print(f"    Processing page {p+1} with PaddleOCR...", end='', flush=True)
        pix = render_page_image(doc, p, dpi=350)
        h = pix.height
        page_words = paddle_ocr_page(pix, confidence_threshold=confidence_threshold)
        print(f" done ({len(page_words)} text segments found)")
        
        # mark page index
        for w in page_words: w.page = p
        lines = group_words_into_lines(page_words)

        # If we hit a stop heading, terminate
        stop = False
        filtered_words = []
        for ln in lines:
            text = " ".join(w.text for w in ln).strip()
            if STOP_HEADINGS_RE.match(text):
                stop = True
                break
            filtered_words.extend(ln)
        collected.extend(filtered_words)
        end_page = p
        if stop:
            break

    return collected, end_page

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
    return False


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

def mark_sup_sub_lines(words: List[OCRWord], position_only: bool = True) -> List[List[Tuple[OCRWord, Optional[str]]]]:
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

    marked_lines: List[List[Tuple[OCRWord, Optional[str]]]] = []
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

        line_marked: List[Tuple[OCRWord, Optional[str]]] = []
        for i, w in enumerate(ln_sorted):
            txt = (w.text or "").strip()
            tag = None

            if txt and _is_formula_script_candidate(txt) and not _looks_like_page_number(txt):
                cy = w_center_y(w)
                h = _word_height(w)
                
                # Skip likely headers/footers.
                if (cy - y_min) / page_h > 0.10 and (cy - y_min) / page_h < 0.90:
                    left = ln_sorted[i - 1] if i > 0 else None
                    right = ln_sorted[i + 1] if i + 1 < len(ln_sorted) else None

                    left_ok = False
                    right_ok = False
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

            line_marked.append((w, tag))

            if tag:
                line_marked.append((w, tag))



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
    
    # Protect our sup/sub tags
    text = text.replace("<sup>", "___SUP_OPEN___")
    text = text.replace("</sup>", "___SUP_CLOSE___")
    text = text.replace("<sub>", "___SUB_OPEN___")
    text = text.replace("</sub>", "___SUB_CLOSE___")
    
    # NOW escape any remaining < and > (which are user content)
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    
    # Restore our protected tags
    text = text.replace("___SUP_OPEN___", "<sup>")
    text = text.replace("___SUP_CLOSE___", "</sup>")
    text = text.replace("___SUB_OPEN___", "<sub>")
    text = text.replace("___SUB_CLOSE___", "</sub>")
    
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

def _marked_line_to_record(marked_line: List[Tuple[OCRWord, Optional[str]]]) -> Optional[dict]:
    """Convert one OCR line into text + layout metadata."""
    if not marked_line:
        return None

    text_parts = []
    words = []
    for w, tag in marked_line:
        t = (w.text or "").strip()
        if not t:
            continue
        if tag == "sup":
            t = f"<sup>{t}</sup>"
        elif tag == "sub":
            t = f"<sub>{t}</sub>"
        text_parts.append(t)
        words.append(w)

    if not text_parts or not words:
        return None

    return {
        "text": " ".join(text_parts),
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


def _paragraphize_marked_lines(marked_lines: List[List[Tuple[OCRWord, Optional[str]]]]) -> List[str]:
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
def process_pdf(pdf_path: Path, out_dir: Path, overrides: dict, max_first_pages: int, confidence_threshold: float = 0.60, force_single_paragraph: bool = False):
    """Process a PDF and extract abstract text.
    
    Args:
        pdf_path: Path to PDF file
        out_dir: Output directory
        overrides: Override page numbers from CSV (optional, will auto-detect if empty)
        max_first_pages: Max pages to search for abstract
        confidence_threshold: Filter out OCR detections below this confidence (0.0-1.0).
                             Default 0.60. Increase to filter more noise.
        force_single_paragraph: Merge all extracted text into one paragraph.
    """

    print(f"\n{'='*70}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*70}")

    with fitz.open(pdf_path) as doc:
        print(f"  PDF loaded: {doc.page_count} pages")
        
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
        words, end_idx = extract_abstract_region(doc, start_p, max_pages=(end_p-start_p+1), confidence_threshold=confidence_threshold)
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
            t = replace_greek_math(paragraph)
            t = escape_user_content(t)
            t = escape_ampersands_not_entities(t)
            t = convert_remaining_non_ascii(t)
            html_paragraphs.append(t)

        html_text = to_html_paragraphs(html_paragraphs)
        html_text = html_text.replace("\n", "").replace("\r", "")

        # Write HTML output
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{pdf_path.stem} draft.txt").write_text(html_text, encoding="utf-8")
        
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
    parser = argparse.ArgumentParser(description="Extract abstracts from PDFs using Paddle OCR.")
    parser.add_argument("--input", required=True, help="Directory with PDFs")
    parser.add_argument("--out", required=True, help="Output directory for HTML files")
    parser.add_argument("--override-csv", default=None, help="Optional CSV file with filename,pages (format: filename,start-end). If not provided, auto-detects abstract boundaries.")
    parser.add_argument("--confidence-threshold", type=float, default=0.60, help="Filter OCR below this confidence (0.0-1.0). Default: 0.60")
    parser.add_argument("--force-single-paragraph", action="store_true", help="Merge extracted abstract text into one <p> block.")
    args = parser.parse_args()
    
    init_paddle_ocr()

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load overrides if provided
    overrides = load_overrides(Path(args.override_csv)) if args.override_csv else {}

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
    
    for idx, pdf in enumerate(pdfs, 1):
        print(f"\n[{idx}/{len(pdfs)}] ", end="")
        process_pdf(
            pdf,
            out_dir,
            overrides=overrides,
            max_first_pages=15,
            confidence_threshold=args.confidence_threshold,
            force_single_paragraph=args.force_single_paragraph,
        )
    
    print(f"\n\nProcessed {len(pdfs)} PDFs. Text files saved to {out_dir}")

if __name__ == "__main__":
    main()
