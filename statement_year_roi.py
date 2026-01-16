#!/usr/bin/env python3
from __future__ import annotations

"""statement_year_roi.py

Isolated Statement YEAR ROI detector used by demo_header.py.

Purpose: keep the 'statement year' ROI logic stable and harder to accidentally regress when
editing other parts of the pipeline.

This module is intentionally self-contained: it only expects each word item to have
x0,y0,x1,y1,text attributes (WordBox-like).
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Protocol
import numpy as np


class WordBoxLike(Protocol):
    x0: int
    y0: int
    x1: int
    y1: int
    text: str


def _center_y(w: WordBoxLike) -> float:
    return 0.5 * (w.y0 + w.y1)


def _center_x(w: WordBoxLike) -> float:
    return 0.5 * (w.x0 + w.x1)


def _norm_token(s: str) -> str:
    return "".join(ch.lower() for ch in (s or "").strip() if ch.isalnum() or ch in {"&"})


def group_words_into_lines(words: List[WordBoxLike], y_tol: int = 8) -> List[List[WordBoxLike]]:
    """Cluster word boxes into approximate text lines."""
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w.y0, w.x0))
    lines: List[List[WordBoxLike]] = []
    cur: List[WordBoxLike] = []
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


def _line_text_lower(line: List[WordBoxLike]) -> str:
    return " ".join((w.text or "").strip().lower() for w in line if (w.text or "").strip())


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
    words: List[WordBoxLike],
    img_shape: Tuple[int, int],
    *,
    table_roi: Optional[Tuple[int, int, int, int]] = None,
    header_roi: Optional[Tuple[int, int, int, int]] = None,
    y_tol: int = 8,
    look_ahead: int = 3,
) -> Optional[Dict[str, Any]]:
    """Return best statement-year ROI dict or None.

    Output keys mirror what demo_header.py expects:
      - bbox: (x0,y0,x1,y1)
      - year: int
      - score: int
      - line_idx: int
      - text: normalized line text
    """
    img_h, img_w = img_shape
    if not words:
        return None

    lines = group_words_into_lines(words, y_tol=y_tol)
    lines_sorted = sorted(lines, key=lambda ln: (min(w.y0 for w in ln), min(w.x0 for w in ln)))

    def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], pad: int = 0) -> bool:
        """Axis-aligned bbox overlap test with optional padding."""
        ax0, ay0, ax1, ay1 = map(int, a)
        bx0, by0, bx1, by1 = map(int, b)
        ax0 -= pad
        ay0 -= pad
        ax1 += pad
        ay1 += pad
        bx0 -= pad
        by0 -= pad
        bx1 += pad
        by1 += pad
        return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

    def _inside_bbox(w: WordBoxLike, bb: Tuple[int, int, int, int]) -> bool:
        """Treat a word as 'inside' a bbox if it overlaps (not just center-point inclusion).

        This is more robust when narrow columns or clipping cause the word center to fall
        just outside the ROI while the glyphs still visually overlap it.
        """
        wb = (int(w.x0), int(w.y0), int(w.x1), int(w.y1))
        return _rects_overlap(wb, bb, pad=1)

    def _inside_table(w: WordBoxLike) -> bool:
        if table_roi is None:
            return False
        return _inside_bbox(w, table_roi)

    def _filter_ln(ln: List[WordBoxLike]) -> List[WordBoxLike]:
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

    def _is_side_adjacent(w: WordBoxLike) -> bool:
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
        year_words: List[WordBoxLike] = []

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
        norm_text = re.sub(r"[^a-z0-9\s]", " ", text_l)
        norm_text = re.sub(r"\s+", " ", norm_text).strip()

        for kw in STATEMENT_YEAR_KEYWORDS:
            if kw in norm_text:
                score += 8
                hit_kw = kw
                break

        if table_top is not None:
            dist = abs(int(min(w.y0 for w in ln)) - table_top)
            score += max(0, 6 - int(dist / max(1, line_h * 2)))

        if any(_is_side_adjacent(w) for w in ln):
            score += 2

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
                    break

        year = years[0]

        evidence: List[WordBoxLike] = list(year_words)
        if hit_kw:
            kw_parts = [p for p in hit_kw.split() if p.strip()]
            kw_norm = set(_norm_token(p) for p in kw_parts)
            for w in ln:
                if _norm_token(w.text) in kw_norm:
                    evidence.append(w)

        seen = set()
        uniq: List[WordBoxLike] = []
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

        # --- Fix #1: Statement year ROI must NEVER overlap the detected table ROI ---
        # In multi-column layouts it's common for a "statement date" in a side panel
        # to share the same y-band as transaction rows. If we accidentally absorb any
        # table-adjacent numerics, the bbox can expand into the table.
        if table_roi is not None:
            cand_bb = (bx0, by0, bx1, by1)
            if _rects_overlap(cand_bb, table_roi, pad=0):
                tx0, ty0, tx1, ty1 = map(int, table_roi)
                ccx = 0.5 * (bx0 + bx1)
                # If the candidate lives to the LEFT of the table, clamp its right edge.
                if ccx < tx0:
                    bx1 = min(bx1, max(0, tx0 - 2))
                # If the candidate lives to the RIGHT of the table, clamp its left edge.
                elif ccx > tx1:
                    bx0 = max(bx0, min(img_w - 1, tx1 + 2))
                else:
                    # Candidate straddles the table: treat as invalid.
                    continue

                # After clamping, re-check overlap; if still overlapping, reject.
                cand_bb2 = (bx0, by0, bx1, by1)
                if _rects_overlap(cand_bb2, table_roi, pad=0) or bx1 <= bx0:
                    continue

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
