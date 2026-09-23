import argparse
from pathlib import Path
import re
import fitz  # PyMuPDF
from rapidfuzz import fuzz

# Report only (prints to console without generating comparison files)
# python detect_and_extract_duplicates.py "/path/to/pdfs"

# Report and export paired comparison PDFs into a designated review folder
# python detect_and_extract_duplicates.py "/path/to/pdfs" -o "/path/to/review_folder"

# Matches standard Roman numerals (1 to 3999)
ROMAN_REGEX = (
    r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)


def extract_printed_folio(page) -> str | None:
    """Inspects top and bottom text lines to extract Arabic or Roman page numbers."""
    text_lines = [
        line.strip()
        for line in page.get_text("text").splitlines()
        if line.strip()
    ]
    if not text_lines:
        return None

    candidate_lines = text_lines[:2] + text_lines[-2:]

    for line in candidate_lines:
        clean = line.strip().lower()

        # 1. Pure Arabic numeral
        digits = re.findall(r"\b\d+\b", clean)
        if len(digits) == 1 and len(clean.split()) <= 3:
            return digits[0]

        # 2. Pure Roman numeral
        cleaned_alpha = re.sub(r"[^\w]", "", clean)
        if cleaned_alpha and re.match(
            ROMAN_REGEX, cleaned_alpha, re.IGNORECASE
        ):
            return cleaned_alpha.lower()

    return None


def extract_normalized_body(page) -> str:
    """Extract embedded text, strip punctuation, and collapse all whitespace."""
    raw_text = page.get_text("text")
    clean = re.sub(r"[^\w\s]", " ", raw_text.lower())
    return " ".join(clean.split())


def export_duplicate_pairs(
    pdf_path: Path, duplicates: list[dict], output_dir: Path
):
    """Extracts paired pages (original and duplicate) into a standalone PDF."""
    source_doc = fitz.open(pdf_path)
    review_doc = fitz.open()

    # Collect 0-indexed page indices while preserving pair order
    pages_to_extract = []
    for d in duplicates:
        pages_to_extract.append(d["orig_idx"])
        pages_to_extract.append(d["dupe_idx"])

    # Remove duplicates if multiple pages matched the same base page, preserving sequence
    unique_pages = list(dict.fromkeys(pages_to_extract))

    for p_idx in unique_pages:
        # insert_pdf takes 0-indexed from_page and to_page (inclusive)
        review_doc.insert_pdf(
            source_doc, from_page=p_idx, to_page=p_idx
        )

    out_file = output_dir / f"DUPLICATES_{pdf_path.name}"
    review_doc.save(out_file, garbage=4, deflate=True)

    review_doc.close()
    source_doc.close()
    print(f"  -> Exported comparison PDF to: {out_file.name}")


def find_and_export_duplicates(
    pdf_path: Path,
    output_dir: Path | None = None,
    similarity_threshold: float = 88.0,
    min_chars: int = 40,
    max_length_ratio_diff: float = 0.20,
):
    doc = fitz.open(pdf_path)
    processed_pages = []
    duplicates = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = extract_normalized_body(page)
        char_len = len(text)

        if char_len < min_chars:
            continue

        folio = extract_printed_folio(page)
        matched = False

        for prev_num, prev_text, prev_len, prev_folio in processed_pages:
            # 1. Length Guard
            len_ratio = min(char_len, prev_len) / max(char_len, prev_len)
            if (1.0 - len_ratio) > max_length_ratio_diff:
                continue

            # 2. Folio Gate
            if (
                folio is not None
                and prev_folio is not None
                and folio != prev_folio
            ):
                continue

            # 3. Sequence Similarity
            score = fuzz.ratio(text, prev_text)
            required_threshold = (
                similarity_threshold
                if (folio and prev_folio and folio == prev_folio)
                else (similarity_threshold + 4.0)
            )

            if score >= required_threshold:
                duplicates.append(
                    {
                        "dupe_idx": page_num,
                        "orig_idx": prev_num,
                        "duplicate_page": page_num + 1,
                        "original_page": prev_num + 1,
                        "score": round(score, 1),
                        "folio": folio or "N/A",
                    }
                )
                matched = True
                break

        if not matched:
            processed_pages.append((page_num, text, char_len, folio))

    doc.close()

    # If duplicates were detected and an export directory is supplied, generate side-by-side PDF
    if duplicates and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        export_duplicate_pairs(pdf_path, duplicates, output_dir)

    return duplicates


def scan_directory(
    directory_path: str,
    output_dir: str | None = None,
    recursive: bool = False,
):
    dir_path = Path(directory_path)
    out_path = Path(output_dir) if output_dir else None

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(list(dir_path.glob(pattern)))

    if not pdf_files:
        print(f"No PDF files found in '{dir_path.resolve()}'.")
        return

    print(f"Scanning {len(pdf_files)} PDF(s) with Duplicate Page Extraction...\n")

    for pdf in pdf_files:
        # Avoid scanning generated comparison files if run in the same directory
        if pdf.name.startswith("DUPLICATES_"):
            continue

        try:
            dupes = find_and_export_duplicates(pdf, output_dir=out_path)
            if dupes:
                print(f"[{pdf.name}] - Found {len(dupes)} duplicate(s):")
                for d in dupes:
                    print(
                        f"  * Page {d['duplicate_page']} matches Page {d['original_page']} "
                        f"(Score: {d['score']}%, Folio: '{d['folio']}')"
                    )
            else:
                print(f"[{pdf.name}] - Clean")
        except Exception as e:
            print(f"[{pdf.name}] - Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect PDF duplicate pages and export matched pairs to a review PDF."
    )
    parser.add_argument("directory", help="Path to input PDF directory")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to save combined duplicate-page PDFs",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Scan subdirectories recursively",
    )
    args = parser.parse_args()

    scan_directory(
        args.directory, output_dir=args.output_dir, recursive=args.recursive
    )