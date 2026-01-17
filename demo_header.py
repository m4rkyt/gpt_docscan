#!/usr/bin/env python3
from __future__ import annotations

"""
demo_header.py

Minimal ROI detector (NO OCR):
- Ingest a PDF via PyMuPDF (fitz)
- Render selected pages to BGR images (OpenCV) at configurable DPI
- Extract word boxes via page.get_text("words") and map to IMAGE pixel coords
  (handles /Rotate via page.rotation_matrix so coords match rendered pixels)

Detect per page:
1) Bottom footer "Page N" ROI (used as an additional end-of-table cue)
2) Main transaction HEADER ROI (tight bbox around header tokens only)
3) Transaction TABLE ROI (header x-range extended downward using a multi-signal end detector)
4) Statement YEAR ROI (tight bbox around year near statement date/period; may be above, beside,
   or on adjacent lines near the header/table)

Table-end detector (design-change resistant):
- Signal C (baseline): "table-like" row run ends -> N consecutive non-table lines OR large whitespace band
- Signal A (clamp): footer page band cutoff_y (if present) clamps table end upward
- Signal B (clamp): single/short marker lines like "continued", "summary" near the table bottom clamp end

Outputs:
- out/page_XXX_roi.png overlays:
  * FOOTER PAGE ROI (green)
  * HEADER ROI (blue)
  * TABLE ROI (red)
  * STATEMENT YEAR ROI (magenta)
- out/roi_results.txt summary per page

Deps:
  pip install pymupdf opencv-python numpy
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import subprocess
import numpy as np
from page_number_roi import detect_page_number_roi
import cv2
import fitz
import json

def dump_extracted_table_json(out_dir: str, pdf_path: str, header: list[str], rows: list[list[str]], statement_year: int | None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    jpath = os.path.join(out_dir, f"{base}.extracted_table.json")
    obj = {
        "pdf_path": pdf_path,
        "out_dir": out_dir,
        "header": header,
        "rows": rows,
        "statement_year": statement_year,
    }
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return jpath



# -----------------------------
# Data model + helpers
# -----------------------------
@dataclass(frozen=True)
class WordBox:
    x0: int
    y0: int
    x1: int
    y1: int
    text: str


def _center_y(w: WordBox) -> float:
    return 0.5 * (w.y0 + w.y1)


def _center_x(w: WordBox) -> float:
    return 0.5 * (w.x0 + w.x1)


# -----------------------------
# ROI exclusion helpers (enforce non-overlap & detection order)
# -----------------------------

from typing import Iterable


def _inside_bbox_xy(x: float, y: float, bb: Tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = map(int, bb)
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def filter_words_excluding_rois(words: List[WordBox], rois: List[Tuple[int, int, int, int]]) -> List[WordBox]:
    """Remove words whose CENTER lies inside any ROI in rois."""
    if not rois:
        return words
    out: List[WordBox] = []
    for w in words:
        cx = _center_x(w)
        cy = _center_y(w)
        if any(_inside_bbox_xy(cx, cy, bb) for bb in rois):
            continue
        out.append(w)
    return out


def mask_image_regions(img_bgr: np.ndarray, rois: List[Tuple[int, int, int, int]], pad: int = 2) -> np.ndarray:
    """Paint earlier ROIs white so pixel-based detectors don't see them."""
    if not rois:
        return img_bgr
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    for (x0, y0, x1, y1) in rois:
        x0 = max(0, int(x0) - pad); y0 = max(0, int(y0) - pad)
        x1 = min(w - 1, int(x1) + pad); y1 = min(h - 1, int(y1) + pad)
        out[y0:y1, x0:x1] = 255
    return out


def clip_bbox_to_avoid_rois(
    bbox: Tuple[int, int, int, int],
    avoid: List[Tuple[int, int, int, int]],
    *,
    img_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, int, int]:
    """
    Enforce: earlier ROIs never overlap later ones.

    Since ROIs are rectangles, we resolve overlaps by clipping vertically away from
    the overlapping earlier ROI, choosing the direction by comparing centers.

    This works well for the typical page layout where ROIs form vertical bands
    (date/header/table/footer).
    """
    x0, y0, x1, y1 = map(int, bbox)
    for (ax0, ay0, ax1, ay1) in avoid:
        ax0, ay0, ax1, ay1 = map(int, (ax0, ay0, ax1, ay1))
        overlap = not (x1 < ax0 or ax1 < x0 or y1 < ay0 or ay1 < y0)
        if not overlap:
            continue

        cy = 0.5 * (y0 + y1)
        acy = 0.5 * (ay0 + ay1)
        if cy < acy:
            # bbox is above earlier ROI -> clip bottom up
            y1 = min(y1, ay0 - 1)
        else:
            # bbox is below earlier ROI -> clip top down
            y0 = max(y0, ay1 + 1)

    if img_shape is not None:
        ih, iw = img_shape
        x0 = max(0, min(x0, iw - 1))
        x1 = max(0, min(x1, iw - 1))
        y0 = max(0, min(y0, ih - 1))
        y1 = max(0, min(y1, ih - 1))

    if x1 < x0:
        x1 = x0
    if y1 < y0:
        y1 = y0
    return (x0, y0, x1, y1)


def _norm_token(s: str) -> str:
    return "".join(ch.lower() for ch in (s or "").strip() if ch.isalnum() or ch in {"&"})


def _split_norm_tokens(text: str) -> List[str]:
    parts = re.split(r"[^A-Za-z0-9&]+", (text or "").strip())
    out: List[str] = []
    for p in parts:
        nt = _norm_token(p)
        if nt:
            out.append(nt)
    if not out:
        nt = _norm_token(text or "")
        if nt:
            out.append(nt)
    return out


def group_words_into_lines(words: List[WordBox], y_tol: int = 8) -> List[List[WordBox]]:
    """Cluster word boxes into approximate text lines."""
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w.y0, w.x0))
    lines: List[List[WordBox]] = []
    cur: List[WordBox] = []
    cur_y: Optional[float] = None

    for w in words_sorted:
        cy = _center_y(w)
        if cur_y is None:
            cur = [w]
            cur_y = cy
            continue
        if abs(cy - cur_y) <= y_tol:
            cur.append(w)
            cur_y = (cur_y * (len(cur) - 1) + cy) / len(cur)
        else:
            lines.append(sorted(cur, key=lambda z: z.x0))
            cur = [w]
            cur_y = cy

    if cur:
        lines.append(sorted(cur, key=lambda z: z.x0))
    return lines


def _line_text_lower(line: List[WordBox]) -> str:
    return " ".join((w.text or "").strip().lower() for w in line if (w.text or "").strip())


def _is_amount_token(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    t2 = t.replace(",", "").replace("£", "").replace("$", "").replace("€", "")
    if t2.startswith("(") and t2.endswith(")"):
        t2 = t2[1:-1]
    t2 = re.sub(r"(CR|DR)$", "", t2, flags=re.IGNORECASE).strip()
    try:
        float(t2)
        return True
    except Exception:
        return False


def _is_date_token(s: str) -> bool:
    """Light heuristic: dd Mon, dd/mm(/yy), dd-mm(/yy)."""
    t = (s or "").strip()
    if not t:
        return False
    parts = t.replace(".", "").split()
    if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) in (1, 2):
        if parts[1].isalpha() and 3 <= len(parts[1]) <= 9:
            return True
    for sep in ("/", "-"):
        if sep in t:
            p = t.split(sep)
            if 2 <= len(p) <= 3 and all(x.isdigit() for x in p[:2]):
                return True
    return False


# -----------------------------
# Bottom footer "Page N" detection
# -----------------------------
from footer_roi import detect_footer_page_roi


# -----------------------------
# Header detection
# -----------------------------
from header_roi import detect_header_band


# -----------------------------
# Table end detector (multi-signal)
# -----------------------------
MARKER_WORDS = {
    "continued", "continue", "contd", "cont.", "summary", "summarised", "totals", "total",
    "end", "endofstatement", "endofpage", "broughtforward", "carriedforward",
}


def _detect_whitespace_gap(
    img_bgr: np.ndarray,
    rx0: int,
    rx1: int,
    *,
    search_from: int,
    min_gap_px: int,
) -> Optional[Tuple[int, int]]:
    """Return first (gap_y0,gap_y1) that is a mostly-white band across [rx0,rx1]."""
    h, w = img_bgr.shape[:2]
    x0 = max(0, min(rx0, w - 1))
    x1 = max(0, min(rx1, w))
    if x1 <= x0 + 10:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    roi = gray[:, x0:x1]
    white = roi > 245
    white_ratio = white.mean(axis=1)

    y = max(0, min(search_from, h - 1))
    in_run = False
    run_start = 0
    for yy in range(y, h):
        is_white_row = white_ratio[yy] >= 0.985
        if is_white_row and not in_run:
            in_run = True
            run_start = yy
        elif (not is_white_row) and in_run:
            run_end = yy - 1
            in_run = False
            if (run_end - run_start + 1) >= min_gap_px:
                return (run_start, run_end)
    if in_run:
        run_end = h - 1
        if (run_end - run_start + 1) >= min_gap_px:
            return (run_start, run_end)
    return None


def _quantize_centers(xs: List[float], q: int) -> List[int]:
    return [int(round(x / q)) for x in xs]


def _build_column_signature(lines: List[List[WordBox]], rx0: int, rx1: int, *, q: int = 25, max_rows: int = 12) -> Dict[int, int]:
    """
    Build a simple "column signature": quantized x-center bins -> frequency
    from the first few table-like rows. This is deliberately lightweight and robust.
    """
    freq: Dict[int, int] = {}
    rows_used = 0
    for ln in lines:
        if rows_used >= max_rows:
            break
        xs = []
        for w in ln:
            cx = _center_x(w)
            if rx0 <= cx <= rx1 and (w.text or "").strip():
                xs.append(cx)
        if len(xs) < 2 or len(xs) > 12:
            continue
        for b in _quantize_centers(xs, q):
            freq[b] = freq.get(b, 0) + 1
        rows_used += 1
    return freq


def _alignment_overlap(line: List[WordBox], sig: Dict[int, int], rx0: int, rx1: int, *, q: int = 25) -> float:
    if not sig:
        return 0.0
    xs = []
    for w in line:
        cx = _center_x(w)
        if rx0 <= cx <= rx1 and (w.text or "").strip():
            xs.append(cx)
    if not xs:
        return 0.0
    bins = set(_quantize_centers(xs, q))
    sig_bins = set(sig.keys())
    return len(bins & sig_bins) / max(1, len(bins))


def _is_table_like_line(
    line: List[WordBox],
    rx0: int,
    rx1: int,
    *,
    col_sig: Dict[int, int],
    q: int = 25,
) -> bool:
    toks = [w for w in line if (w.text or "").strip() and rx0 <= _center_x(w) <= rx1]
    n = len(toks)
    if n == 0:
        return False
    if n > 14:  # paragraph/text box
        return False

    has_num = any(_is_amount_token(w.text) or any(ch.isdigit() for ch in (w.text or "")) for w in toks)
    has_date = any(_is_date_token(w.text) for w in toks)
    overlap = _alignment_overlap(toks, col_sig, rx0, rx1, q=q)

    # table-like if it's compact and has some structure
    features = 0
    if has_num:
        features += 1
    if has_date:
        features += 1
    if overlap >= 0.40:
        features += 1
    if 2 <= n <= 12:
        features += 1

    return features >= 2


def detect_table_end_y(
    words: List[WordBox],
    img_bgr: np.ndarray,
    rx0: int,
    rx1: int,
    header_y1: int,
    *,
    y_tol: int = 8,
    min_rows: int = 4,
    max_consec_non_table: int = 8,
) -> Dict[str, Any]:
    """
    Baseline Signal C: find end of table by scanning lines under header.
    Stops when:
      - N consecutive non-table lines after seeing enough table rows, OR
      - a large whitespace gap appears after last table row.

    Returns dict with:
      last_table_y1, y_end_transition, whitespace_gap (optional)
    """
    h, _w = img_bgr.shape[:2]
    # only consider words within x-span and below header
    scoped = [w for w in words if (rx0 <= _center_x(w) <= rx1 and w.y0 >= header_y1 - 2)]
    lines = group_words_into_lines(scoped, y_tol=y_tol)
    lines = sorted(lines, key=lambda ln: min(z.y0 for z in ln))

    # Build a lightweight column signature using early rows
    col_sig = _build_column_signature(lines, rx0, rx1, q=25, max_rows=12)

    last_table_y1 = header_y1
    seen_table = 0
    consec_non = 0

    for ln in lines:
        y0 = min(w.y0 for w in ln)
        y1 = max(w.y1 for w in ln)
        if y0 < header_y1 - 2:
            continue

        table_like = _is_table_like_line(ln, rx0, rx1, col_sig=col_sig, q=25)
        if table_like:
            last_table_y1 = max(last_table_y1, y1)
            seen_table += 1
            consec_non = 0
        else:
            if seen_table > 0:
                consec_non += 1
                if seen_table >= min_rows and consec_non >= max_consec_non_table:
                    break

    y_end_transition = min(h - 1, last_table_y1 + 8)

    # Whitespace gap check after last table row
    # Require at least ~2 line heights of white to count as a gap
    heights = [(w.y1 - w.y0) for w in scoped if (w.y1 - w.y0) > 0]
    line_h = int(np.median(heights)) if heights else 14
    min_gap_px = max(12, 2 * line_h)
    gap = _detect_whitespace_gap(img_bgr, rx0, rx1, search_from=max(header_y1, last_table_y1 + 1), min_gap_px=min_gap_px)

    y_end_whitespace = None
    if gap is not None:
        gap_y0, gap_y1 = gap
        # only accept if it starts after header and reasonably after last table row
        if gap_y0 > header_y1 + 5 and gap_y0 >= last_table_y1 + 2:
            y_end_whitespace = max(header_y1 + 5, gap_y0 - 2)

    return {
        "last_table_y1": int(last_table_y1),
        "y_end_transition": int(y_end_transition),
        "y_end_whitespace": int(y_end_whitespace) if y_end_whitespace is not None else None,
        "whitespace_gap": gap,
        "seen_table_rows": int(seen_table),
    }


def detect_marker_end_y(
    words: List[WordBox],
    rx0: int,
    rx1: int,
    *,
    after_y: int,
    before_y: int,
    y_tol: int = 8,
    max_lines_down: int = 15,
) -> Optional[int]:
    """
    Signal B: find short marker lines like 'continued' / 'summary' below the table.
    Looks between after_y and before_y (clamps), within x-span.
    """
    scoped = [w for w in words if (rx0 <= _center_x(w) <= rx1 and after_y <= w.y0 <= before_y)]
    lines = group_words_into_lines(scoped, y_tol=y_tol)
    lines = sorted(lines, key=lambda ln: min(z.y0 for z in ln))

    count = 0
    for ln in lines:
        if count >= max_lines_down:
            break
        count += 1
        toks = [nt for wb in ln for nt in _split_norm_tokens(wb.text)]
        toks = [t for t in toks if t]
        if not toks:
            continue
        # short line preference
        if len(toks) > 4:
            continue
        if any(t in MARKER_WORDS for t in toks) or any("continued" in t for t in toks):
            return int(min(w.y0 for w in ln))
    return None


# -----------------------------
# Page-level ROI detection (footer -> header -> table -> year)
# -----------------------------


def detect_table_rois_for_page(
    words: List[WordBox],
    img_bgr: np.ndarray,
    *,
    y_tol: int = 8,
    pad_px: int = 8,
) -> Dict[str, Any]:
    """
    Enforced detection order:
      1) Page number ROI
      2) Footer ROI
      3) Table header ROI
      4) Table data ROI
      5) Statement year/date ROI (ONLY if a table is detected)

    Contract:
      - Later detections never overlap earlier ROIs.
      - Earlier ROIs are excluded from later word- and pixel-based detection.
    """
    img_h, img_w = img_bgr.shape[:2]

    occupied: List[Tuple[int, int, int, int]] = []
    year_roi: Optional[Dict[str, Any]] = None

    # 1) PAGE number ROI
    page_n = detect_page_number_roi(words, (img_h, img_w), y_tol=y_tol, bottom_lines_to_search=3)
    page_n_bbox = page_n.get("bbox") if page_n else None
    if page_n_bbox is not None:
        page_n_bbox = tuple(map(int, page_n_bbox))
        occupied.append(page_n_bbox)

    # 2) FOOTER ROI (pixel/word-based) — exclude page ROI
    words_for_footer = filter_words_excluding_rois(words, occupied)
    img_for_footer = mask_image_regions(img_bgr, occupied, pad=2)
    #footer_page = detect_footer_page_roi(words_for_footer, img_for_footer, y_tol=y_tol)
    footer_page = detect_footer_page_roi(words_for_footer, img_for_footer, y_tol=y_tol, exclude_roi=page_n_bbox)
    footer_bbox = footer_page.get("bbox") if footer_page else None
    
    if footer_bbox is not None:
        footer_bbox = clip_bbox_to_avoid_rois(tuple(map(int, footer_bbox)), occupied, img_shape=(img_h, img_w))
        footer_page["bbox"] = footer_bbox
        footer_page["cutoff_y"] = int(footer_bbox[1])
        occupied.append(footer_bbox)

    # 3) HEADER ROI — exclude page+footer
    words_for_header = filter_words_excluding_rois(words, occupied)
    hits, _header_line_bbox, header_table_bounds = detect_header_band(words_for_header, img_w, img_h, y_tol=y_tol)

    if header_table_bounds is None:
        # Fallback band (still must avoid earlier ROIs)
        rx0, rx1 = int(0.10 * img_w), int(0.90 * img_w)
        header_y0, header_y1 = int(0.15 * img_h), int(0.18 * img_h)

        header_roi = clip_bbox_to_avoid_rois((rx0, header_y0, rx1, header_y1), occupied, img_shape=(img_h, img_w))
        occupied.append(header_roi)

        # 4) TABLE ROI — exclude page+footer+header
        words_for_table = filter_words_excluding_rois(words, occupied)
        img_for_table = mask_image_regions(img_bgr, occupied, pad=2)

        baseline = detect_table_end_y(words_for_table, img_for_table, rx0, rx1, header_roi[3], y_tol=y_tol)
        #y_end = baseline["y_end_whitespace"] if baseline["y_end_whitespace"] is not None else baseline["y_end_transition"]
        y_end = (end_info.get("last_row_y1") or end_info.get("y_end_transition") or y_end) + 3

        footer_used = False
        if footer_page and footer_page.get("cutoff_y") is not None:
            cutoff_y = int(footer_page["cutoff_y"])
            if header_roi[3] + 20 < cutoff_y < y_end - 10:
                y_end = cutoff_y - 2
                footer_used = True

        table_roi = (
            max(0, rx0 - pad_px),
            max(0, header_roi[1]),
            min(img_w - 1, rx1 + pad_px),
            min(img_h - 1, y_end + pad_px),
        )
        table_roi = clip_bbox_to_avoid_rois(tuple(map(int, table_roi)), occupied, img_shape=(img_h, img_w))

        # 5) STATEMENT YEAR ROI — only if table detected
        if baseline.get("seen_table_rows", 0) > 0:
            occupied_for_year = occupied + [table_roi]
            words_for_year = filter_words_excluding_rois(words, occupied_for_year)
            year_roi = detect_statement_year_roi(
                words_for_year,
                (img_h, img_w),
                table_roi=table_roi,
                header_roi=header_roi,
                y_tol=y_tol,
            )
            if year_roi is not None and year_roi.get("bbox") is not None:
                yb = clip_bbox_to_avoid_rois(tuple(map(int, year_roi["bbox"])), occupied_for_year, img_shape=(img_h, img_w))
                year_roi["bbox"] = yb

        return {
            "header_hits": hits,
            "page_n_roi": page_n_bbox,
            "footer_page_roi": footer_page["bbox"] if footer_page else None,
            "footer_page_used": footer_used,
            "marker_used": False,
            "year_roi": year_roi,
            "header_roi": header_roi,
            "table_roi": table_roi,
            "table_end_debug": baseline,
        }

    # Header ROI bounds (tight)
    rx0 = max(0, header_table_bounds[0] - 2)
    rx1 = min(img_w - 1, header_table_bounds[2] + 2)
    header_y0 = max(0, header_table_bounds[1] - pad_px)
    header_y1 = min(img_h - 1, header_table_bounds[3] + pad_px)

    header_roi = (rx0, header_y0, rx1, header_y1)
    header_roi = clip_bbox_to_avoid_rois(tuple(map(int, header_roi)), occupied, img_shape=(img_h, img_w))
    occupied.append(header_roi)

    # 4) TABLE ROI — exclude page+footer+header
    words_for_table = filter_words_excluding_rois(words, occupied)
    img_for_table = mask_image_regions(img_bgr, occupied, pad=2)

    baseline = detect_table_end_y(words_for_table, img_for_table, rx0, rx1, header_roi[3], y_tol=y_tol)
    y_end = baseline["y_end_whitespace"] if baseline["y_end_whitespace"] is not None else baseline["y_end_transition"]

    # Marker clamp (Signal B)
    before_y = img_h - 1
    if footer_page and footer_page.get("cutoff_y") is not None:
        before_y = min(before_y, int(footer_page["cutoff_y"]))

    marker_y = detect_marker_end_y(
        words_for_table,
        rx0,
        rx1,
        after_y=int(baseline["last_table_y1"]) + 1,
        before_y=before_y,
        y_tol=y_tol,
        max_lines_down=18,
    )

    marker_used = False
    if marker_y is not None and header_roi[3] + 20 < marker_y < y_end - 5:
        y_end = marker_y - 2
        marker_used = True

    # Footer clamp (Signal A)
    footer_used = False
    if footer_page and footer_page.get("cutoff_y") is not None:
        cutoff_y = int(footer_page["cutoff_y"])
        if header_roi[3] + 20 < cutoff_y < y_end - 10:
            y_end = cutoff_y - 2
            footer_used = True

    table_roi = (
        max(0, rx0 - pad_px),
        max(0, header_roi[1]),
        min(img_w - 1, rx1 + pad_px),
        min(img_h - 1, y_end + pad_px),
    )
    table_roi = clip_bbox_to_avoid_rois(tuple(map(int, table_roi)), occupied, img_shape=(img_h, img_w))

    # 5) STATEMENT YEAR ROI — only if table detected
    if baseline.get("seen_table_rows", 0) > 0:
        occupied_for_year = occupied + [table_roi]
        words_for_year = filter_words_excluding_rois(words, occupied_for_year)
        year_roi = detect_statement_year_roi(
            words_for_year,
            (img_h, img_w),
            table_roi=table_roi,
            header_roi=header_roi,
            y_tol=y_tol,
        )
        if year_roi is not None and year_roi.get("bbox") is not None:
            yb = clip_bbox_to_avoid_rois(tuple(map(int, year_roi["bbox"])), occupied_for_year, img_shape=(img_h, img_w))
            year_roi["bbox"] = yb

    return {
        "header_hits": hits,
        "page_n_roi": page_n_bbox,
        "footer_page_roi": footer_page["bbox"] if footer_page else None,
        "footer_page_used": footer_used,
        "marker_used": marker_used,
        "year_roi": year_roi,
        "header_roi": header_roi,
        "table_roi": table_roi,
        "table_end_debug": baseline,
    }


# -----------------------------
# Statement YEAR ROI
# -----------------------------
# Statement YEAR ROI
# -----------------------------
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MONTH_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_DATE_WITH_MONTH_YEAR_RE = re.compile(
    r"\b(\d{1,2})\s+"
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:tember)?|sept|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+"
    r"(\d{2}|\d{4})\b",
    re.IGNORECASE,
)
_DATE_WITH_NUM_YEAR_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](\d{2}|\d{4})\b")

STATEMENT_YEAR_KEYWORDS = {
    "statement date",
    "statement dated",
    "date of statement",
    "period ending",
    "statement period",
    "year ended",
    "for the year",
    "tax year",
    "financial year",
    "period from",
    "period to",
}


def detect_statement_year_roi(
    words: List[WordBox],
    img_shape: Tuple[int, int],
    *,
    table_roi: Optional[Tuple[int, int, int, int]] = None,
    header_roi: Optional[Tuple[int, int, int, int]] = None,
    y_tol: int = 8,
    look_ahead: int = 3,
) -> Optional[Dict[str, Any]]:
    img_h, img_w = img_shape
    if not words:
        return None

    lines = group_words_into_lines(words, y_tol=y_tol)
    lines_sorted = sorted(lines, key=lambda ln: (min(w.y0 for w in ln), min(w.x0 for w in ln)))

    def _inside_bbox(w: WordBox, bb: Tuple[int, int, int, int]) -> bool:
        x0, y0, x1, y1 = map(int, bb)
        cx = _center_x(w)
        cy = _center_y(w)
        return (x0 <= cx <= x1) and (y0 <= cy <= y1)

    def _inside_table(w: WordBox) -> bool:
        if table_roi is None:
            return False
        return _inside_bbox(w, table_roi)

    def _filter_ln(ln: List[WordBox]) -> List[WordBox]:
        if table_roi is None:
            return ln
        return [w for w in ln if not _inside_table(w)]

    table_top = int(table_roi[1]) if table_roi is not None else None
    table_left = int(table_roi[0]) if table_roi is not None else None
    table_right = int(table_roi[2]) if table_roi is not None else None

    heights = [(w.y1 - w.y0) for w in words if (w.y1 - w.y0) > 0]
    line_h = int(np.median(heights)) if heights else 14
    near_band_px = max(80, 6 * line_h)

    if table_top is not None:
        near_y0 = max(0, table_top - near_band_px)
        near_y1 = min(img_h - 1, table_top + near_band_px)
    else:
        near_y0, near_y1 = 0, img_h - 1

    side_margin = 8

    def _is_side_adjacent(w: WordBox) -> bool:
        if table_left is None or table_right is None:
            return False
        cx = _center_x(w)
        return (cx < (table_left - side_margin)) or (cx > (table_right + side_margin))

    candidates: List[Dict[str, Any]] = []

    for i, ln_raw in enumerate(lines_sorted):
        ln = _filter_ln(ln_raw)
        if not ln:
            continue

        y0 = min(w.y0 for w in ln)
        keep = True
        if table_top is not None:
            keep = (y0 < table_top) or (near_y0 <= y0 <= near_y1)

        if header_roi is not None:
            hy1 = int(header_roi[3])
            keep = keep or (hy1 <= y0 <= min(img_h - 1, hy1 + near_band_px))

        if not keep:
            continue

        text_l = _line_text_lower(ln)
        if not text_l:
            continue

        years: List[int] = []
        year_words: List[WordBox] = []

        full_line = " ".join((w.text or "").strip() for w in ln if (w.text or "").strip())

        mdate = _DATE_WITH_MONTH_YEAR_RE.search(full_line)
        if mdate:
            yraw = mdate.group(3)
            y = int(yraw)
            if len(yraw) == 2:
                y = (2000 + y) if y <= 50 else (1900 + y)
            years.append(y)
            for w in ln:
                if _MONTH_RE.search(w.text or "") or re.search(r"\b\d{2,4}\b", (w.text or "").strip()):
                    year_words.append(w)

        if not years:
            mnum = _DATE_WITH_NUM_YEAR_RE.search(full_line)
            if mnum:
                yraw = mnum.group(1)
                y = int(yraw)
                if len(yraw) == 2:
                    y = (2000 + y) if y <= 50 else (1900 + y)
                years.append(y)
                for w in ln:
                    if _DATE_WITH_NUM_YEAR_RE.search((w.text or "").strip()):
                        year_words.append(w)

        if not years:
            for w in ln:
                for m in _YEAR_RE.finditer((w.text or "").strip()):
                    years.append(int(m.group(0)))
                    year_words.append(w)

        if not years:
            continue

        score = 5
        hit_kw = None
        found_kw_nearby = False
        norm_text = re.sub(r"[^a-z0-9\s]", " ", text_l)
        norm_text = re.sub(r"\s+", " ", norm_text).strip()

        for kw in STATEMENT_YEAR_KEYWORDS:
            if kw in norm_text:
                hit_kw = kw
                score += 8
                break

        # Guardrail: far from table and no keyword -> skip
        if table_top is not None:
            dist_to_top = abs(int(y0) - table_top)
            in_near_band = dist_to_top <= near_band_px
            if (not in_near_band) and (hit_kw is None):
                continue

        if table_top is not None:
            dist = abs(int(y0) - table_top)
            if dist <= near_band_px:
                score += 10
                score += max(0, int((near_band_px - dist) / max(1, line_h)))
            if y0 >= table_top and dist <= near_band_px:
                score += 2

        if table_roi is not None and any(_is_side_adjacent(w) for w in year_words):
            score += 6

        if hit_kw is None:
            for j in range(max(0, i - look_ahead), min(len(lines_sorted), i + look_ahead + 1)):
                if j == i:
                    continue
                ln2 = _filter_ln(lines_sorted[j])
                if not ln2:
                    continue
                y02 = min(w.y0 for w in ln2)
                if table_top is not None and not (y02 < table_top or (near_y0 <= y02 <= near_y1)):
                    continue
                s2 = _line_text_lower(ln2)
                s2 = re.sub(r"[^a-z0-9\s]", " ", s2)
                s2 = re.sub(r"\s+", " ", s2).strip()
                if any(kw in s2 for kw in STATEMENT_YEAR_KEYWORDS):
                    score += 5
                    found_kw_nearby = True
                    break
        # Context gate: avoid false positives (e.g., sort-code-like 20-12-25) by requiring
        # at least one statement/date keyword on this line or a nearby line.
        if hit_kw is None and not found_kw_nearby:
            continue


        year = years[0]

        evidence: List[WordBox] = list(year_words)
        if hit_kw:
            kw_parts = [p for p in hit_kw.split() if p.strip()]
            kw_norm = set(_norm_token(p) for p in kw_parts)
            for w in ln:
                if _norm_token(w.text) in kw_norm:
                    evidence.append(w)

        seen = set()
        uniq: List[WordBox] = []
        for w in evidence:
            key = (w.x0, w.y0, w.x1, w.y1, _norm_token(w.text))
            if key not in seen:
                seen.add(key)
                uniq.append(w)
        if not uniq:
            continue

        pad = 1
        bx0 = max(0, int(min(w.x0 for w in uniq)) - pad)
        by0 = max(0, int(min(w.y0 for w in uniq)) - pad)
        bx1 = min(img_w - 1, int(max(w.x1 for w in uniq)) + pad)
        by1 = min(img_h - 1, int(max(w.y1 for w in uniq)) + pad)

        candidates.append({
            "bbox": (bx0, by0, bx1, by1),
            "year": year,
            "score": int(score),
            "line_idx": int(i),
            "text": norm_text,
        })

    if not candidates:
        return None

    if table_top is not None:
        candidates.sort(key=lambda c: (c["score"], -abs(int(c["bbox"][1]) - table_top)), reverse=True)
    else:
        candidates.sort(key=lambda c: (c["score"], -c["bbox"][1]), reverse=True)
    return candidates[0]


# -----------------------------
# Rendering + PDF parsing
# -----------------------------
def render_page_to_bgr(page: fitz.Page, dpi: int) -> Tuple[np.ndarray, float]:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, zoom


def extract_words_image_coords(page: fitz.Page, scale: float) -> List[WordBox]:
    """Map word boxes into IMAGE pixel coords, applying rotation matrix if present."""
    out: List[WordBox] = []
    rot = getattr(page, "rotation_matrix", None)
    for w in page.get_text("words"):
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        if rot is not None:
            r = fitz.Rect(x0, y0, x1, y1) * rot
            x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
        out.append(WordBox(
            int(round(x0 * scale)),
            int(round(y0 * scale)),
            int(round(x1 * scale)),
            int(round(y1 * scale)),
            str(text),
        ))
    return out


def parse_pages(pages_str: str, n_pages: int) -> List[int]:
    if not pages_str:
        return list(range(n_pages))
    s = pages_str.replace(" ", "")
    out: List[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            a_i = max(1, int(a))
            b_i = min(n_pages, int(b))
            out.extend([i - 1 for i in range(a_i, b_i + 1)])
        else:
            i = int(part)
            if 1 <= i <= n_pages:
                out.append(i - 1)
    return sorted(set(out))


# -----------------------------
# Drawing
# -----------------------------
def draw_bbox(img: np.ndarray, bbox: Tuple[int, int, int, int], label: str, color: Tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = map(int, bbox)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    cv2.putText(img, label, (x0 + 2, max(10, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ROI detector: footer-page ROI, header ROI, table ROI, statement year ROI (no OCR).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("pdf", help="Path to input PDF")
    p.add_argument("--dpi", type=int, default=200, help="Render DPI for debug images")
    p.add_argument("--pages", default="", help="Optional page selection: e.g. '1,2,5' or '3-7' (1-based). Blank=all")
    p.add_argument("--out_dir", default="out", help="Output folder for debug images/results")
    p.add_argument("--no_images", action="store_true", help="Do not write debug images")
    p.add_argument("--debug", action="store_true", help="Verbose logging")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.pdf):
        print(f"ERROR: file not found: {args.pdf}", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)

    doc = fitz.open(args.pdf)
    selected = parse_pages(args.pages, doc.page_count)

    report_lines: List[str] = []

    for page_idx in selected:
        page = doc.load_page(page_idx)
        img, scale = render_page_to_bgr(page, args.dpi)
        words = extract_words_image_coords(page, scale)

        rois = detect_table_rois_for_page(words, img, y_tol=8, pad_px=8)
        table_roi = rois.get("table_roi")
        header_roi = rois.get("header_roi")
        footer_page_roi = rois.get("footer_page_roi")
        footer_page_used = rois.get("footer_page_used", False)
        marker_used = rois.get("marker_used", False)
        year_roi = rois.get("year_roi")

        if not args.no_images:
            vis = img.copy()
            if footer_page_roi is not None:
                draw_bbox(vis, footer_page_roi, "FOOTER PAGE ROI", (0, 255, 0))  # green
            if rois.get("page_n_roi") is not None:
                draw_bbox(vis, rois["page_n_roi"], "PAGE N ROI", (0, 255, 255))  # yellow
            if header_roi is not None:
                draw_bbox(vis, header_roi, "HEADER ROI", (255, 0, 0))  # blue (BGR)
            if table_roi is not None:
                draw_bbox(vis, table_roi, "TABLE ROI", (0, 0, 255))    # red (BGR)
            if year_roi is not None and year_roi.get("bbox") is not None:
                draw_bbox(vis, year_roi["bbox"], f"STATEMENT YEAR {year_roi.get('year','')}".strip(), (255, 0, 255))
            out_img = os.path.join(args.out_dir, f"page_{page_idx+1:03d}_roi.png")
            cv2.imwrite(out_img, vis)

        dbg = rois.get("table_end_debug") or {}
        report_lines.append(
            f"PAGE {page_idx+1:03d} | header_hits={rois.get('header_hits')} | "
            f"footer_page_used={footer_page_used} marker_used={marker_used} | "
            f"header_roi={header_roi} | table_roi={table_roi} | "
            f"seen_rows={dbg.get('seen_table_rows')} last_row_y1={dbg.get('last_table_y1')} "
            f"y_end_transition={dbg.get('y_end_transition')} y_end_whitespace={dbg.get('y_end_whitespace')} | "
            f"year={year_roi.get('year') if year_roi else None} year_roi={(year_roi.get('bbox') if year_roi else None)}"
        )

        if args.debug:
            print(report_lines[-1])

    report_path = os.path.join(args.out_dir, "roi_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Run external extractor to print table text to console
    extract_script = os.path.join(os.path.dirname(__file__), "extract_table_text.py")
    subprocess.run(
       [sys.executable, extract_script, args.pdf, "--out_dir", args.out_dir, "--dpi", str(args.dpi)],
    check=True
    )
    
    print(f"Done. Results written to: {args.out_dir}")
    print(f"- ROI report: {report_path}")
    if not args.no_images:
        print("- Debug images: page_XXX_roi.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
