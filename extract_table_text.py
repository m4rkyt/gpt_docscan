#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import numpy as np
import cv2

from page_number_roi import detect_page_number_roi
from footer_roi import detect_footer_page_roi
from header_roi import detect_header_band
from statement_year_roi import detect_statement_year_roi

# Reuse the proven table-end + line grouping logic from your main script (no modification)
import demo_header  # provides WordBox, group_words_into_lines, detect_table_end_y


# ----------------------------
# Date formatting (dd mmm yy)
# ----------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _parse_dd_mmm_optional_year(s: str) -> Optional[Tuple[int, int, Optional[int]]]:
    """
    Accepts:
      '1 Jan' / '01 Jan' / '01 Jan 24' / '01 Jan 2024'
      '01Jan' / '01Jan24' / '01Jan2024'
    Returns (dd, mm, yyyy_or_None) or None.
    """
    s = _norm_spaces(s).lower()
    s = s.replace(".", "").replace(",", "")
    m = re.match(r"^(\d{1,2})\s*([a-z]{3,4})\s*(\d{2,4})?$", s)
    if not m:
        return None

    dd = int(m.group(1))
    mon = m.group(2)
    if mon not in _MONTHS:
        return None
    mm = _MONTHS[mon]

    yy_raw = m.group(3)
    if yy_raw is None:
        yyyy = None
    else:
        yyyy = 2000 + int(yy_raw) if len(yy_raw) == 2 else int(yy_raw)

    return dd, mm, yyyy

def format_date_dd_mmm_yy(date_str: str, statement_year: Optional[int]) -> str:
    """
    Outputs 'dd mmm yy' (e.g., '01 Jan 24').
    If date has no year, uses statement_year.
    """
    raw = _norm_spaces(date_str)
    parsed = _parse_dd_mmm_optional_year(raw)
    if not parsed:
        return raw

    dd, mm, yyyy = parsed
    if yyyy is None:
        if not statement_year:
            return raw
        yyyy = int(statement_year)

    try:
        dt = datetime(yyyy, mm, dd)
    except ValueError:
        return raw

    return dt.strftime("%d %b %y")


# ----------------------------
# Fixed-width output (spaces)
# ----------------------------

def _render_fixed_width(header: List[str], rows: List[List[str]], sep: str = "  ") -> List[str]:
    """
    Renders a fixed-width table (space padded) for consistent console alignment.
    """
    ncols = len(header)
    widths = [len(h) for h in header]

    for r in rows:
        for i in range(ncols):
            cell = r[i] if i < len(r) else ""
            widths[i] = max(widths[i], len(cell))

    lines: List[str] = []
    lines.append(sep.join(header[i].ljust(widths[i]) for i in range(ncols)))

    for r in rows:
        rr = (r + [""] * ncols)[:ncols]
        lines.append(sep.join(rr[i].ljust(widths[i]) for i in range(ncols)))

    return lines


# ----------------------------
# PDF -> image + words in pixel coords
# ----------------------------

def render_page_bgr(page: fitz.Page, dpi: int) -> np.ndarray:
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return bgr

def words_in_image_coords(page: fitz.Page, dpi: int) -> List[demo_header.WordBox]:
    """
    Word boxes mapped into rendered image pixel coords.
    """
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0) * page.rotation_matrix

    out: List[demo_header.WordBox] = []
    for w in page.get_text("words"):  # (x0,y0,x1,y1,"text",block,line,word)
        x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], w[4]
        if not (txt or "").strip():
            continue

        p0 = fitz.Point(x0, y0) * mat
        p1 = fitz.Point(x1, y1) * mat

        ix0 = int(min(p0.x, p1.x))
        iy0 = int(min(p0.y, p1.y))
        ix1 = int(max(p0.x, p1.x))
        iy1 = int(max(p0.y, p1.y))

        out.append(demo_header.WordBox(ix0, iy0, ix1, iy1, txt))
    return out


# ----------------------------
# Column building from header line
# ----------------------------

@dataclass(frozen=True)
class HeaderCol:
    name: str
    x0: int
    x1: int
    cx: float

def _merge_header_words(words: List[demo_header.WordBox], gap_px: int = 24) -> List[HeaderCol]:
    """
    Merge adjacent header words into phrases like 'Money In'.
    Assumes words are on the same line already (header band).
    """
    if not words:
        return []

    ws = sorted(words, key=lambda w: w.x0)
    merged: List[List[demo_header.WordBox]] = [[ws[0]]]
    for w in ws[1:]:
        prev = merged[-1][-1]
        gap = w.x0 - prev.x1
        if gap <= gap_px:
            merged[-1].append(w)
        else:
            merged.append([w])

    cols: List[HeaderCol] = []
    for chunk in merged:
        x0 = min(w.x0 for w in chunk)
        x1 = max(w.x1 for w in chunk)
        name = _norm_spaces(" ".join(w.text.strip() for w in chunk))
        cx = 0.5 * (x0 + x1)
        if name:
            cols.append(HeaderCol(name=name, x0=x0, x1=x1, cx=cx))

    return cols

def _column_edges(cols: List[HeaderCol]) -> List[float]:
    """
    Build edges [-inf, mid(c0,c1), mid(c1,c2), ..., +inf]
    """
    if not cols:
        return [-1e9, 1e9]
    cs = sorted(cols, key=lambda c: c.cx)
    mids = [0.5 * (cs[i].cx + cs[i + 1].cx) for i in range(len(cs) - 1)]
    return [-1e9] + mids + [1e9]

def _assign_col(cx: float, edges: List[float]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= cx < edges[i + 1]:
            return i
    return max(0, len(edges) - 2)


# ----------------------------
# Extract rows from table ROI
# ----------------------------

def extract_rows_from_table_roi(
    words: List[demo_header.WordBox],
    table_roi: Tuple[int, int, int, int],
    header_y1: int,
    header_cols: List[HeaderCol],
    y_tol: int = 8,
) -> List[List[str]]:
    rx0, ry0, rx1, ry1 = map(int, table_roi)
    edges = _column_edges(header_cols)
    ncols = len(header_cols)

    # keep words within ROI and below header
    scoped = [
        w for w in words
        if (w.y0 >= header_y1 - 1)
        and (rx0 <= (0.5 * (w.x0 + w.x1)) <= rx1)
        and (ry0 <= (0.5 * (w.y0 + w.y1)) <= ry1)
        and (w.text or "").strip()
    ]

    lines = demo_header.group_words_into_lines(scoped, y_tol=y_tol)
    lines = sorted(lines, key=lambda ln: min(w.y0 for w in ln))

    rows: List[List[str]] = []
    for ln in lines:
        ln_sorted = sorted(ln, key=lambda w: w.x0)

        cols = [[] for _ in range(ncols)]
        for w in ln_sorted:
            cx = 0.5 * (w.x0 + w.x1)
            ci = _assign_col(cx, edges)
            if 0 <= ci < ncols:
                cols[ci].append(w.text.strip())

        row = [_norm_spaces(" ".join(parts)) for parts in cols]

        if any(cell for cell in row):
            rows.append(row)

    return rows


# ----------------------------
# Main extraction orchestration
# ----------------------------

def extract_pdf_to_space_aligned_txt(
    pdf_path: str,
    out_dir: str,
    dpi: int = 220,
    pages: Optional[str] = None,
    y_tol: int = 8,
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    page_indices = list(range(doc.page_count))

    # optional pages like "1,2,5-7" (1-based)
    if pages:
        wanted: List[int] = []
        for part in pages.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                a = int(a); b = int(b)
                wanted.extend(list(range(a - 1, b)))
            else:
                wanted.append(int(part) - 1)
        page_indices = [i for i in wanted if 0 <= i < doc.page_count]

    all_rows: List[List[str]] = []
    final_header: Optional[List[str]] = None
    date_idx: Optional[int] = None
    statement_year_global: Optional[int] = None

    for pi in page_indices:
        page = doc.load_page(pi)
        img = render_page_bgr(page, dpi)
        h, w = img.shape[:2]
        words = words_in_image_coords(page, dpi)

        # not strictly required for extraction, but keeps behavior consistent with your ROI logic
        _ = detect_page_number_roi(words, (h, w))
        _ = detect_footer_page_roi(words, img, y_tol=y_tol)

        # Detect header band
        _hits, _header_line_bbox, header_tokens_bbox = detect_header_band(words, w, h, y_tol=y_tol)
        if not header_tokens_bbox:
            continue

        hx0, hy0, hx1, hy1 = map(int, header_tokens_bbox)

        # Table end from demo_header logic
        end_info = demo_header.detect_table_end_y(words, img, hx0, hx1, hy1, y_tol=y_tol)
        y_end = end_info.get("y_end_whitespace") or end_info.get("y_end_transition") or min(h - 1, hy1 + 1500)

        table_roi = (hx0, hy0, hx1, int(y_end))

        # statement year (prefer first non-None)
        yr_info = detect_statement_year_roi(
            words,
            (h, w),
            table_roi=table_roi,
            header_roi=header_tokens_bbox,
            y_tol=y_tol
        )
        yr = None
        if isinstance(yr_info, dict):
            yr = yr_info.get("year")
        if yr and not statement_year_global:
            statement_year_global = int(yr)

        # Build header columns from header tokens
        header_words = [wb for wb in words if (hx0 <= wb.x0 <= hx1 and hy0 <= wb.y0 <= hy1)]
        cols = _merge_header_words(header_words)
        cols_sorted = sorted(cols, key=lambda c: c.cx)

        # lock header once
        if final_header is None:
            final_header = [c.name for c in cols_sorted]
            for i, nm in enumerate(final_header):
                if _norm_spaces(nm).lower() == "date":
                    date_idx = i
                    break

        # Extract body rows aligned to detected columns
        rows = extract_rows_from_table_roi(words, table_roi, hy1, cols_sorted, y_tol=y_tol)

        # normalize dates
        if date_idx is not None and final_header is not None:
            for r in rows:
                if 0 <= date_idx < len(r):
                    r[date_idx] = format_date_dd_mmm_yy(r[date_idx], statement_year_global)

        all_rows.extend(rows)

    # Write single output file (fixed-width with spaces)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_txt = os.path.join(out_dir, f"{base}.txt")

    with open(out_txt, "w", encoding="utf-8", newline="\n") as f:
        if final_header:
            lines = _render_fixed_width(final_header, all_rows, sep="  ")
            for ln in lines:
                f.write(ln.rstrip() + "\n")
        else:
            # fallback: write rows with 2-space separator
            for r in all_rows:
                f.write("  ".join(_norm_spaces(x) for x in r) + "\n")

    return out_txt


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract detected transaction table(s) to one space-aligned .txt per PDF.")
    ap.add_argument("pdf", help="Path to PDF")
    ap.add_argument("--out_dir", default="out", help="Output directory")
    ap.add_argument("--dpi", type=int, default=220, help="Render DPI (match demo_header DPI for best results)")
    ap.add_argument("--pages", default=None, help='Optional 1-based page selection like "1,2,5-7"')
    ap.add_argument("--y_tol", type=int, default=8, help="Line clustering tolerance in px")
    args = ap.parse_args()

    out_txt = extract_pdf_to_space_aligned_txt(args.pdf, args.out_dir, dpi=args.dpi, pages=args.pages, y_tol=args.y_tol)
    print(out_txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
