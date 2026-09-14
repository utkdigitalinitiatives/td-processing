import argparse
from pathlib import Path
import shutil
import fitz  # PyMuPDF

# Finds pages whose table/figure was printed sideways and sets the page's
# /Rotate so a viewer displays the page landscape with the content upright.
#
# This is a lossless change. /Rotate is one key in the page dictionary -- the
# scanned image bytes are never re-rendered or re-compressed.

# Report only (prints to console, changes nothing)
# python fix_rotation.py "/path/to/pdfs" -r

# Report and export review PDFs containing just the affected pages, rotated
# python fix_rotation.py "/path/to/pdfs" -r -o "/path/to/review_folder"

# Write corrected copies into an output folder, leaving the originals alone
# python fix_rotation.py "/path/to/pdfs" -r --apply -o "/path/to/rotated"

# Overwrite the originals (a .bak is written alongside each one first)
# python fix_rotation.py "/path/to/pdfs" -r --in-place

# Two things this deliberately does not do:
#   * It does not repair the text layer. On scans whose OCR was rotation-blind
#     the invisible text stays garbage, and after rotating it sits sideways
#     relative to the image. Fixing that needs a re-OCR of those pages.
#   * /Rotate is whole-page, so a page mixing upright body text with one
#     sideways table cannot be fixed this way. Those normally classify as 0 and
#     are left alone, which is the right outcome.
#   * It does not flip upside-down pages on its own. A 180 that only the image
#     classifier believes is always left for a human -- see decide_rotation().

# PaddleOCR's document-orientation classifier. Already cached under
# ~/.paddlex/official_models/ by the OCR script, so this costs no new download.
ORIENTATION_MODEL = "PP-LCNet_x1_0_doc_ori"

# A line's writing direction is a unit vector. Anything with |dx| >= this is
# being read left-to-right; below it the line runs up or down the page.
HORIZONTAL_DIR_CUTOFF = 0.7

# Below this many text lines a page is "near-blank": the image classifier gets
# unreliable there (it is what produced the only false positives seen in
# testing), so such pages are sent to review rather than rotated.
MIN_LINES_FOR_CONFIDENCE = 5

# Share of lines that must agree before the text layer alone is trusted.
TEXTLAYER_MAJORITY = 0.8

# How far to look either side of an uncertain page for confident neighbours
# turning the same way, and how many of them it takes to settle the matter.
NEIGHBOUR_WINDOW = 2
NEIGHBOUR_VOTES = 2

# How sure the model has to be before its disagreement overrules a rotation
# that something else settled. Pages it simply cannot read -- a scatter of dots,
# a dense block of figures -- come back disagreeing at around 0.6 to 0.7, while
# a page genuinely left sideways comes back at 0.88 and up. Only the latter is
# evidence; the former is the model declining to judge.
CONTRADICTION_SCORE = 0.80


def textlayer_rotation(page) -> tuple[int | None, int, int]:
    """Reads the /Rotate a page needs from its OCR text layer's line directions.

    Returns (rotation still needed, lines counted, rotation the raw text shows).
    The last two differ once a page carries a /Rotate that already cancels the
    raw direction, which is how an already-corrected page is recognised.
    """
    votes: dict[int, int] = {}

    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            dx, dy = line["dir"]
            if abs(dx) >= HORIZONTAL_DIR_CUTOFF:
                # (1, 0) is normal; (-1, 0) is upside down.
                rotation = 0 if dx > 0 else 180
            else:
                # (0, -1) reads bottom-to-top, (0, 1) reads top-to-bottom.
                rotation = 90 if dy < 0 else 270
            votes[rotation] = votes.get(rotation, 0) + 1

    if not votes:
        return None, 0, 0

    total = sum(votes.values())
    winner, count = max(votes.items(), key=lambda kv: kv[1])

    # A mixed page (a rotated caption beside upright body text) is not something
    # a whole-page rotation can fix, so only report a clear majority.
    if count / total < TEXTLAYER_MAJORITY:
        return None, total, 0

    if winner == 0:
        # Horizontal text is not evidence of anything, because OCR that was
        # blind to rotation reads a sideways table as horizontal garbage. It
        # especially must not reach the subtraction below, which would turn a
        # page this script had already corrected into a confident demand to
        # rotate it straight back.
        return None, total, 0

    # These vectors are in unrotated page space and ignore any /Rotate the page
    # already carries, so discount it -- otherwise an already-corrected file
    # reports itself as still needing the same rotation.
    return (winner - page.rotation) % 360, total, winner


def classify_at(page, classifier, extra_rotation: int, dpi: int = 100) -> tuple[int, float]:
    """Renders the page as if turned by extra_rotation and reads its orientation."""
    import numpy as np

    original = page.rotation
    page.set_rotation((original + extra_rotation) % 360)
    try:
        pixmap = page.get_pixmap(dpi=dpi)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        # The classifier wants 3 channels; drop alpha if the page rendered with one.
        if pixmap.n == 4:
            image = image[:, :, :3]

        result = list(classifier.predict(image))[0]
        label = int(result["label_names"][0])
        score = float(result["scores"][0])
    finally:
        page.set_rotation(original)

    # The label is the degrees counter-clockwise needed to upright the image,
    # while /Rotate turns the page clockwise, so the two are complements.
    return (360 - label) % 360, score


def image_rotation(page, classifier, dpi: int = 100) -> tuple[int, float, bool]:
    """Asks the orientation model what /Rotate a page needs, then checks its work.

    Returns (rotation, score, verified). The model is good at spotting that a
    page is sideways but can pick the wrong way round, landing 180 degrees out,
    so its answer is only trusted once the page turned that far reads back as
    upright. Where the re-read says it is exactly 180 out, that is a usable
    answer in itself and the correction is tried and re-checked in turn.
    """
    rotation, score = classify_at(page, classifier, 0, dpi=dpi)
    if rotation == 0:
        return 0, score, True

    check, check_score = classify_at(page, classifier, rotation, dpi=dpi)
    if check == 0:
        return rotation, score, True

    if check == 180:
        corrected = (rotation + 180) % 360
        recheck, _ = classify_at(page, classifier, corrected, dpi=dpi)
        if recheck == 0:
            # The weaker of the two readings, since this took two goes.
            return corrected, min(score, check_score), True

    return rotation, score, False


def decide_rotation(
    tl_rotation: int | None,
    line_count: int,
    tl_raw: int,
    img_rotation: int,
    score: float,
    img_verified: bool = True,
    min_score: float = 0.70,
) -> tuple[int, str, str]:
    """Weighs the two detectors against each other and returns (rotation, confidence, why)."""
    # The two detectors are not equally strong, and the asymmetry matters:
    # vertical text is measured geometry that nothing produces by accident,
    # while horizontal text proves little, because OCR that was blind to
    # rotation reads a sideways table as horizontal garbage. So a vertical text
    # layer outranks the image classifier, and a horizontal one does not.
    if tl_rotation:
        if tl_rotation == img_rotation:
            # Corroboration beats either detector's own score; in testing this
            # held up on classifier scores as low as 0.48.
            return tl_rotation, "high", "text layer + image agree"
        # An uncorroborated 180 is refused here for the same reason it is
        # refused further down when the image is the one proposing it: a bound
        # thesis is scanned one way round, so a genuinely upside-down page is
        # rare, while a wrong 180 is what a patch of bad direction vectors
        # looks like. Checked by eye across two documents, every one of these
        # was a page that was already the right way up.
        if tl_rotation == 180:
            return 180, "review", "only the text layer says 180"
        return tl_rotation, "high", "text layer majority"

    # Nothing left to do, but the raw text ran vertically -- so the page already
    # carries a /Rotate that cancels it. It is upright whatever the image says,
    # and the image often does disagree here, because a wide table displayed
    # landscape is exactly what the classifier is weakest on.
    if tl_rotation == 0 and tl_raw:
        return 0, "clean", "already rotated, text layer agrees"

    if img_rotation == 0:
        return 0, "clean", "image says upright"

    # Down to the image classifier alone, so it has to clear the score bar and
    # the page has to have enough on it to judge.
    if line_count < MIN_LINES_FOR_CONFIDENCE:
        return img_rotation, "review", f"image only, near-blank page, score {score:.2f}"

    # The page did not read as upright once turned this far, so the model has
    # spotted that it is sideways without settling which way round. Applying a
    # guess here is how a table ends up upside down instead of merely rotated.
    if not img_verified:
        return img_rotation, "review", f"image unsure which way round, score {score:.2f}"

    # An uncorroborated 180 never gets applied, whatever it scores. A bound
    # thesis is scanned one way round, so a genuinely upside-down page is rare,
    # while a wide table turned 90 degrees is routine -- and in testing every
    # image-only 180 on a page with real content was a page already upright.
    # A 180 the text layer agrees with is handled above and still applies.
    if img_rotation == 180:
        return 180, "review", f"image only, 180 needs a human, score {score:.2f}"

    if score >= min_score:
        return img_rotation, "medium", f"image, score {score:.2f}"

    return img_rotation, "review", f"image, low score {score:.2f}"


def demote_by_neighbours(decisions: list[dict]):
    """Drops pages whose neighbours proved the text layer unreliable there.

    Where OCR wrote bad direction vectors it did so across a stretch of pages,
    not one, so a page kept only by the text layer while its neighbours were
    caught reading sideways is almost certainly the same corruption -- just
    with too little on it for the model to say so outright. Those pages are the
    ones no single signal can settle, so the run settles them.
    """
    caught = {d["page"] for d in decisions if d.get("contradicted")}
    if not caught:
        return

    for d in decisions:
        if d["confidence"] not in ("high", "medium") or not d["rotation"]:
            continue
        # A page the image independently backed is standing on its own evidence,
        # not on the text layer, so a bad patch of direction vectors says
        # nothing about it.
        if d.get("img_agreed"):
            continue
        near = sum(
            1
            for offset in range(-NEIGHBOUR_WINDOW, NEIGHBOUR_WINDOW + 1)
            if offset and (d["page"] + offset) in caught
        )
        if near >= NEIGHBOUR_VOTES:
            d["confidence"] = "review"
            d["contradicted"] = True
            d["why"] = f"{d['why']}, and {near} neighbours read sideways turned that way"


def promote_by_neighbours(decisions: list[dict]):
    """Settles uncertain pages that sit inside a run of confident ones, in place.

    Sideways material arrives in runs -- an appendix of computer output, a block
    of wide tables -- and the model reads those pages worst, because a sparse
    monospace listing lacks the dense line texture it was trained on. Scores
    dip below the bar on scattered pages in the middle of a run its neighbours
    were confident about. A page surrounded by confident pages all turning the
    same way is that same run, so the neighbours settle it.

    Only pages already believed sideways are eligible, and only in the
    direction the neighbours agree on, so this widens what gets applied without
    inventing a rotation for a page nothing else suspected.
    """
    confident = {
        d["page"]: d["rotation"]
        for d in decisions
        if d["confidence"] in ("high", "medium") and d["rotation"]
    }

    for d in decisions:
        if d["confidence"] != "review" or not d["rotation"]:
            continue
        # Neighbours settle a page nothing could read. They do not overrule the
        # page itself having been read and found still sideways -- that is
        # direct evidence about this page, and it wins.
        if d.get("contradicted"):
            continue
        votes = sum(
            1
            for offset in range(-NEIGHBOUR_WINDOW, NEIGHBOUR_WINDOW + 1)
            if offset and confident.get(d["page"] + offset) == d["rotation"]
        )
        if votes >= NEIGHBOUR_VOTES:
            d["confidence"] = "medium"
            d["why"] = f"{d['why']}, but {votes} neighbours agree"


def export_rotated_pages(pdf_path: Path, decisions: list[dict], output_dir: Path):
    """Extracts just the affected pages, already rotated, into a review PDF."""
    source_doc = fitz.open(pdf_path)
    review_doc = fitz.open()

    for d in decisions:
        review_doc.insert_pdf(
            source_doc, from_page=d["page_idx"], to_page=d["page_idx"]
        )
        new_page = review_doc[-1]
        new_page.set_rotation((new_page.rotation + d["rotation"]) % 360)

    out_file = output_dir / f"ROTATED_{pdf_path.name}"
    review_doc.save(out_file, garbage=4, deflate=True)

    review_doc.close()
    source_doc.close()
    print(f"  -> Exported review PDF to: {out_file.name}")


def find_and_fix_rotations(
    pdf_path: Path,
    classifier,
    output_dir: Path | None = None,
    apply: bool = False,
    in_place: bool = False,
    dpi: int = 100,
    min_score: float = 0.70,
):
    doc = fitz.open(pdf_path)
    decisions = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        contradicted = False
        tl_rotation, line_count, tl_raw = textlayer_rotation(page)
        img_rotation, score, img_verified = image_rotation(page, classifier, dpi=dpi)
        rotation, confidence, why = decide_rotation(
            tl_rotation,
            line_count,
            tl_raw,
            img_rotation,
            score,
            img_verified=img_verified,
            min_score=min_score,
        )

        # Whatever settled it, a page about to be turned has to read as upright
        # once turned. The image check above already established exactly that
        # for its own answer, so only a rotation it did not verify is re-read
        # here -- which is how the text layer gets checked. Being measured
        # geometry does not make it right: where the OCR wrote garbled
        # direction vectors it is confidently and consistently wrong, and
        # nothing else was looking.
        # Two independent signals already pointing the same way is the strongest
        # evidence available, so a rotation resting on both is left alone. Only
        # one resting on a single signal gets re-read -- which is the text layer
        # on its own, or the image on its own where its check confirmed nothing.
        independently_backed = bool(rotation) and rotation == img_rotation and (
            img_verified or tl_rotation == rotation
        )
        if confidence in ("high", "medium") and rotation and not independently_backed:
            check, check_score = classify_at(page, classifier, rotation, dpi=dpi)
            if check != 0 and check_score >= CONTRADICTION_SCORE:
                confidence = "review"
                contradicted = True
                why = f"{why}, but reads sideways turned that way ({check_score:.2f})"

        if rotation == 0 and confidence != "review":
            continue

        decisions.append(
            {
                "page_idx": page_num,
                "page": page_num + 1,
                "rotation": rotation,
                "confidence": confidence,
                "why": why,
                "score": round(score, 2),
                "contradicted": contradicted,
                "img_agreed": independently_backed,
            }
        )

    demote_by_neighbours(decisions)
    promote_by_neighbours(decisions)

    to_rotate = [d for d in decisions if d["confidence"] in ("high", "medium")]

    if to_rotate and output_dir and not apply:
        output_dir.mkdir(parents=True, exist_ok=True)
        export_rotated_pages(pdf_path, to_rotate, output_dir)

    if to_rotate and (apply or in_place):
        for d in to_rotate:
            page = doc[d["page_idx"]]
            # Added, not assigned: some PDFs already carry a /Rotate.
            page.set_rotation((page.rotation + d["rotation"]) % 360)

        if in_place:
            backup = pdf_path.with_suffix(pdf_path.suffix + ".bak")
            if not backup.exists():
                shutil.copy2(pdf_path, backup)
            # incremental=True appends the changed page dictionaries rather than
            # rewriting the file, so the scan bytes are physically untouched.
            doc.save(pdf_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
            print(f"  -> Rotated in place ({backup.name} kept as backup)")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_file = output_dir / pdf_path.name
            # No garbage/deflate: only the page dictionary should differ from
            # the original, which is the whole point of doing it this way.
            doc.save(out_file)
            print(f"  -> Wrote corrected copy to: {out_file.name}")

    doc.close()
    return decisions


def scan_directory(
    directory_path: str,
    output_dir: str | None = None,
    recursive: bool = False,
    apply: bool = False,
    in_place: bool = False,
    dpi: int = 100,
    min_score: float = 0.70,
):
    dir_path = Path(directory_path)
    out_path = Path(output_dir) if output_dir else None

    if dir_path.is_file() and dir_path.suffix.lower() == ".pdf":
        pdf_files = [dir_path]
    else:
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(list(dir_path.glob(pattern)))

    if not pdf_files:
        print(f"No PDF files found in '{dir_path.resolve()}'.")
        return

    if (apply or in_place) and not in_place and out_path is None:
        print("--apply needs an output directory (-o). Refusing to guess one.")
        return

    # --apply promises the originals are left alone, so an -o that resolves to a
    # folder the PDFs are already in is a contradiction rather than a shortcut.
    # Caught here so it fails immediately instead of part-way through a batch.
    if apply and out_path is not None:
        clashing = [p for p in pdf_files if p.parent.resolve() == out_path.resolve()]
        if clashing:
            print(
                f"--apply would overwrite the originals: -o points at the folder "
                f"{clashing[0].parent} that the PDFs are already in.\n"
                f"Point -o somewhere else, or use --in-place to rewrite them on "
                f"purpose (which keeps a .bak of each)."
            )
            return

    # Loading the model takes about a second, so it is done once for the batch.
    from paddleocr import DocImgOrientationClassification

    classifier = DocImgOrientationClassification(model_name=ORIENTATION_MODEL)

    print(f"Scanning {len(pdf_files)} PDF(s) for sideways pages...\n")

    for pdf in pdf_files:
        # Avoid re-scanning our own review output if it lands in the same folder
        if pdf.name.startswith("ROTATED_"):
            continue

        try:
            decisions = find_and_fix_rotations(
                pdf,
                classifier,
                output_dir=out_path,
                apply=apply,
                in_place=in_place,
                dpi=dpi,
                min_score=min_score,
            )

            rotated = [d for d in decisions if d["confidence"] in ("high", "medium")]
            review = [d for d in decisions if d["confidence"] == "review"]

            if not decisions:
                print(f"[{pdf.name}] - Clean")
                continue

            summary = f"Found {len(rotated)} sideways page(s)"
            if review:
                summary += f", {len(review)} for review"
            print(f"[{pdf.name}] - {summary}:")

            for d in sorted(decisions, key=lambda x: x["page"]):
                marker = "?" if d["confidence"] == "review" else "*"
                print(
                    f"  {marker} Page {d['page']} -> /Rotate {d['rotation']} "
                    f"({d['confidence']}: {d['why']})"
                )
        except Exception as e:
            print(f"[{pdf.name}] - Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect sideways PDF pages and set /Rotate so they display upright."
    )
    parser.add_argument("directory", help="Path to a PDF file or a directory of them")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory for review PDFs, or for corrected copies when --apply is used",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Scan subdirectories recursively",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write corrected copies into the output directory",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the original PDFs, keeping a .bak of each",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Render resolution for orientation detection (default: 100)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.70,
        help="Score the image classifier must reach when it is the only detector",
    )
    args = parser.parse_args()

    scan_directory(
        args.directory,
        output_dir=args.output_dir,
        recursive=args.recursive,
        apply=args.apply,
        in_place=args.in_place,
        dpi=args.dpi,
        min_score=args.min_score,
    )
