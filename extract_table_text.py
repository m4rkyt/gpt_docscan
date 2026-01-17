#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import datetime
from typing import List, Optional, Tuple

import fitz
import numpy as np
import cv2

import demo_header
from header_roi import detect_header_band
from statement_year_roi import detect_statement_year_roi


MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

def ns(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def fix_date(s: str, year: Optional[int]) -> str:
    s0 = ns(s).lower().replace(".", "").replace(",", "")
    m = re.match(r"^(\d{1,2})\s*([a-z]{3,4})(?:\s*(\d{2,4}))?$", s0)
    if not m or m.group(2) not in MONTHS:
        return ns(s)
    d = int(m.group(1))
    mth = MONTHS[m.group(2)]
    y = m.group(3)
    if y is None:
        if not year:
            return ns(s)
        y = year
    else:
        y = 2000 + int(y) if len(y) == 2 else int(y)
    try:
        return datetime(int(y), mth, d).strftime("%d %b %y")
    except Exception:
        return ns(s)

def render(page: fitz.Page, dpi: int) -> np.ndarray:
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR if img.shape[2] == 3 else cv2.COLOR_RGBA2BGR)

def words_img(page: fitz.Page, dpi: int):
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0) * page.rotation_matrix
    out = []
    for w in page.get_text("words"):
        if not (w[4] or "").strip():
            continue
        p0 = fitz.Point(w[0], w[1]) * mat
        p1 = fitz.Point(w[2], w[3]) * mat
        out.append(demo_header.WordBox(
            int(min(p0.x, p1.x)), int(min(p0.y, p1.y)),
            int(max(p0.x, p1.x)), int(max(p0.y, p1.y)),
            w[4]
        ))
    return out

def extract_table_rows(words, table_roi, header_roi, header_cols, y_tol=8):
    rx0, ry0, rx1, ry1 = map(int, table_roi)
    _, _, _, hy1 = map(int, header_roi)

    cols = sorted(header_cols, key=lambda c: c.cx)
    edges = [-1e9] + [(cols[i].cx + cols[i + 1].cx) / 2 for i in range(len(cols) - 1)] + [1e9]

    def col_index(cx):
        for i in range(len(edges) - 1):
            if edges[i] <= cx < edges[i + 1]:
                return i
        return len(edges) - 2

    scoped = [
        w for w in words
        if w.y0 >= hy1
        and rx0 <= (w.x0 + w.x1) / 2 <= rx1
        and ry0 <= (w.y0 + w.y1) / 2 <= ry1
        and w.text.strip()
    ]

    lines = demo_header.group_words_into_lines(scoped, y_tol=y_tol)
    lines = sorted(lines, key=lambda ln: min(w.y0 for w in ln))

    out_rows = []
    for ln in lines:
        ln = sorted(ln, key=lambda w: w.x0)
        row = [[] for _ in range(len(cols))]
        for w in ln:
            cx = (w.x0 + w.x1) / 2
            row[col_index(cx)].append(w.text.strip())
        out_rows.append([" ".join(c).strip() for c in row])

    return out_rows

def header_words_to_columns(header_words, gap_px: int = 24):
    ws = sorted(header_words, key=lambda w: w.x0)
    if not ws:
        return []

    groups = [[ws[0]]]
    for w in ws[1:]:
        if w.x0 - groups[-1][-1].x1 <= gap_px:
            groups[-1].append(w)
        else:
            groups.append([w])

    cols = []
    for g in groups:
        x0 = min(w.x0 for w in g)
        x1 = max(w.x1 for w in g)
        name = " ".join(w.text.strip() for w in g).strip()
        cx = 0.5 * (x0 + x1)
        cols.append(type("Col", (), {"name": name, "cx": cx})())
    return cols



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--y_tol", type=int, default=8)
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    statement_year: Optional[int] = None

    for i in range(doc.page_count):
        page = doc.load_page(i)
        img = render(page, args.dpi)
        h, w = img.shape[:2]
        words = words_img(page, args.dpi)

        hits, _, header_roi = detect_header_band(words, w, h, y_tol=args.y_tol)
        if not header_roi:
            continue

        hx0, hy0, hx1, hy1 = map(int, header_roi)

        end = demo_header.detect_table_end_y(words, img, hx0, hx1, hy1, y_tol=args.y_tol)
        y_end = end.get("y_end_whitespace") or end.get("y_end_transition") or min(h - 1, hy1 + 1500)
        table_roi = (hx0, hy0, hx1, int(y_end))

        yr = detect_statement_year_roi(words, (h, w), table_roi=table_roi, header_roi=header_roi, y_tol=args.y_tol)
        if isinstance(yr, dict) and yr.get("year") and not statement_year:
            statement_year = int(yr["year"])

        header_words = [wb for wb in words if hx0 <= wb.x0 <= hx1 and hy0 <= wb.y0 <= hy1]
        header_cols = sorted(header_words_to_columns(header_words), key=lambda c: c.cx)
     
        rows = extract_table_rows(words, table_roi, header_roi, header_cols, y_tol=args.y_tol)

        # Date fix (if we have a Date column)
        header_names = [c.name for c in sorted(header_cols, key=lambda c: c.cx)]
        date_idx = None
        for j, nm in enumerate(header_names):
            if nm.strip().lower() == "date":
                date_idx = j
                break
        if date_idx is not None:
            for r in rows:
                if date_idx < len(r):
                    r[date_idx] = fix_date(r[date_idx], statement_year)

        # Print header + rows per page (minimal)
        print("  ".join(header_names))
        for r in rows:
            print("  ".join(ns(x) for x in r))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
