#!/usr/bin/env python3
from __future__ import annotations

"""footer_roi.py

Simplified footer ROI detection.

Goal
----
Detect the *footer text block* at the bottom of a page by scanning upward from
the bottom of the rendered page image, then (optionally) using PDF-extracted
word boxes to refine the ROI and to expand upward using a font/spacing rule.

Key behavior
------------
- We intentionally *reject* tiny / single-token bands (typically the page
  number) as the seed block. Otherwise the detector locks onto the page number
  and never grows to include the real centered footer text.

Public API
----------
- detect_footer_text_block_roi(words, img_bgr, y_tol=8, ...) -> dict|None
- detect_footer_page_roi(...) is kept as a backwards-compatible alias wrapper.

Expected `words`
----------------
A list of objects (or dicts) with fields/attrs: x0,y0,x1,y1,text.
Coordinates are pixel coordinates in the rendered image space.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2


# -----------------------------
# Types / helpers
# -----------------------------

@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def h(self) -> float:
        return float(self.y1 - self.y0)


def _to_word(obj: Any) -> Word:
    """Accepts dict-like or object-like entries."""
    if isinstance(obj, Word):
        return obj
    if isinstance(obj, dict):
        return Word(
            text=str(obj.get("text", "")),
            x0=float(obj["x0"]),
            y0=float(obj["y0"]),
            x1=float(obj["x1"]),
            y1=float(obj["y1"]),
        )
    return Word(
        text=str(getattr(obj, "text", "")),
        x0=float(getattr(obj, "x0")),
        y0=float(getattr(obj, "y0")),
        x1=float(getattr(obj, "x1")),
        y1=float(getattr(obj, "y1")),
    )


def _clip_bbox(b: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = b
    x0 = max(0, min(int(x0), w - 1))
    x1 = max(0, min(int(x1), w))
    y0 = max(0, min(int(y0), h - 1))
    y1 = max(0, min(int(y1), h))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return x0, y0, x1, y1


def _bbox_overlaps(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    *,
    pad: float = 0.0,
) -> bool:
    """Axis-aligned overlap test with optional padding (applied to `a`)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ax0 -= pad
    ay0 -= pad
    ax1 += pad
    ay1 += pad
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _mask_roi_white(img_bgr: np.ndarray, roi: Tuple[int, int, int, int], *, pad: int = 2) -> np.ndarray:
    """Return a copy of img with roi painted white (for pixel-only scanners)."""
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = roi
    x0 = max(0, int(x0) - pad)
    y0 = max(0, int(y0) - pad)
    x1 = min(w - 1, int(x1) + pad)
    y1 = min(h - 1, int(y1) + pad)
    out = img_bgr.copy()
    out[y0:y1, x0:x1] = 255
    return out


def _intersects(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float], pad: float = 0.0) -> bool:
    x0a, y0a, x1a, y1a = b1
    x0b, y0b, x1b, y1b = b2
    return not (
        (x1a + pad) < (x0b - pad)
        or (x1b + pad) < (x0a - pad)
        or (y1a + pad) < (y0b - pad)
        or (y1b + pad) < (y0a - pad)
    )


def _group_words_into_lines(words: List[Word], y_tol: float) -> List[List[Word]]:
    """Cluster word boxes into approximate lines using vertical centers."""
    if not words:
        return []

    ws = sorted(words, key=lambda w: (w.y0 + w.y1) / 2.0)
    lines: List[List[Word]] = []

    for w in ws:
        yc = (w.y0 + w.y1) / 2.0
        placed = False
        for line in lines:
            ycs = [(ww.y0 + ww.y1) / 2.0 for ww in line]
            line_yc = float(np.median(ycs))
            if abs(yc - line_yc) <= y_tol:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    for line in lines:
        line.sort(key=lambda w: w.x0)

    lines.sort(key=lambda line: float(np.median([(w.y0 + w.y1) / 2.0 for w in line])))
    return lines


def _line_bbox(line: List[Word]) -> Tuple[float, float, float, float]:
    return (
        min(w.x0 for w in line),
        min(w.y0 for w in line),
        max(w.x1 for w in line),
        max(w.y1 for w in line),
    )


def _median_font_height(line: List[Word]) -> float:
    hs = [w.h for w in line if w.h > 0]
    return float(np.median(hs)) if hs else 0.0


# -----------------------------
# Pixel-first detection
# -----------------------------

def _make_ink_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Binary mask: 1 where ink likely exists."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )

    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k1, iterations=1)

    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k2, iterations=1)

    return (thr > 0).astype(np.uint8)


def _band_ink_ratio(ink: np.ndarray, y0: int, y1: int) -> float:
    h, w = ink.shape[:2]
    y0 = max(0, int(y0))
    y1 = min(h, int(y1))
    if y1 <= y0:
        return 0.0
    area = (y1 - y0) * w
    return float(ink[y0:y1, :].sum()) / float(area)


def _scan_up_for_first_text_band(
    ink: np.ndarray,
    *,
    start_from_bottom_px: int = 0,
    band_h: int = 18,
    min_ink_ratio: float = 0.0025,
    max_scan_frac: float = 0.45,
) -> Optional[Tuple[int, int]]:
    """Scan upward from bottom for the first band containing enough ink."""
    h, _w = ink.shape[:2]
    y = h - 1 - int(start_from_bottom_px)
    scan_limit_y = int(h * (1.0 - max_scan_frac))

    while y > scan_limit_y:
        y0 = max(0, y - band_h)
        y1 = y
        if _band_ink_ratio(ink, y0, y1) >= min_ink_ratio:
            return (y0, y1)
        y -= band_h

    return None


def _expand_band_to_block(
    ink: np.ndarray,
    seed_band: Tuple[int, int],
    *,
    band_h: int = 18,
    min_ink_ratio: float = 0.0020,
    gap_allow_bands: int = 1,
) -> Tuple[int, int]:
    """Expand a seed band into a contiguous-ish block of ink bands."""
    h, _w = ink.shape[:2]
    seed_y0, seed_y1 = seed_band

    # expand up
    y0 = int(seed_y0)
    gaps = 0
    while y0 > 0:
        cand0 = max(0, y0 - band_h)
        if _band_ink_ratio(ink, cand0, y0) >= min_ink_ratio:
            y0 = cand0
            gaps = 0
        else:
            gaps += 1
            if gaps > gap_allow_bands:
                break
            y0 = cand0

    # expand down
    y1 = int(seed_y1)
    gaps = 0
    while y1 < h:
        cand1 = min(h, y1 + band_h)
        if _band_ink_ratio(ink, y1, cand1) >= min_ink_ratio:
            y1 = cand1
            gaps = 0
        else:
            gaps += 1
            if gaps > gap_allow_bands:
                break
            y1 = cand1

    return (max(0, int(y0)), min(h, int(y1)))


def _tight_bbox_from_ink(ink: np.ndarray, y0: int, y1: int, *, pad: int = 6) -> Optional[Tuple[int, int, int, int]]:
    """Tight bbox around ink pixels within [y0:y1]."""
    h, w = ink.shape[:2]
    y0 = max(0, int(y0))
    y1 = min(h, int(y1))
    if y1 <= y0:
        return None

    region = ink[y0:y1, :]
    ys, xs = np.where(region > 0)
    if len(xs) == 0:
        return None

    rx0 = int(xs.min())
    rx1 = int(xs.max() + 1)
    ry0 = int(ys.min() + y0)
    ry1 = int(ys.max() + 1 + y0)

    bbox = (rx0 - pad, ry0 - pad, rx1 + pad, ry1 + pad)
    return _clip_bbox(bbox, w, h)


# -----------------------------
# Public detector
# -----------------------------

def detect_footer_text_block_roi(
    words: List[Any],
    img_bgr: np.ndarray,
    *,
    exclude_roi: Optional[Tuple[int, int, int, int]] = None,
    y_tol: float = 8.0,
    # pixel scan
    scan_band_h: int = 16,
    scan_min_ink_ratio: float = 0.0015,
    scan_max_scan_frac: float = 0.45,
    # seed rejection guards (fix for PAGE N being below centered footer)
    min_seed_height_px: int = 14,
    min_seed_words: int = 3,
    # block sanity guards (prevent grabbing a huge table/content block)
    min_block_top_frac: float = 0.55,
    max_block_height_frac: float = 0.25,
    min_block_width_frac: float = 0.20,
    # font expansion
    font_height_tol: float = 0.25,
    max_line_gap_mult: float = 1.6,
    # bbox padding
    pad_px: int = 6,
) -> Optional[Dict[str, Any]]:
    """Detect the footer text block at the bottom of the page.

    Returns dict:
      {
        'bbox': (x0,y0,x1,y1),
        'block_bbox_px': (x0,y0,x1,y1),
        'font_h': float,
        'lines_included': int,
        'cutoff_y': int
      }

    Notes
    -----
    - The seed band is found via pixel scan from the bottom.
    - We reject seed bands that look like page-number artifacts:
        * too thin (height < min_seed_height_px)
        * intersect too few word boxes (count < min_seed_words)
      and continue scanning upward.
    """
    if img_bgr is None:
        return None

    # Optionally exclude an already-detected ROI (typically the page number ROI).
    # This prevents the pixel scan from locking onto "Page N" artifacts and also
    # prevents word refinement from re-introducing them.
    if exclude_roi is not None:
        img_bgr = _mask_roi_white(img_bgr, tuple(map(int, exclude_roi)), pad=2)

    h, w = img_bgr.shape[:2]
    ink = _make_ink_mask(img_bgr)

    words_norm: List[Word] = [_to_word(x) for x in (words or [])]
    if exclude_roi is not None and words_norm:
        ex = tuple(float(v) for v in exclude_roi)
        words_norm = [
            ww
            for ww in words_norm
            if not _bbox_overlaps((ww.x0, ww.y0, ww.x1, ww.y1), ex, pad=2.0)
        ]

    # Find a seed band, expand to a block, and validate it.
    # If the candidate block is implausible for a footer (too high / too tall / too narrow)
    # we keep scanning upward rather than accidentally selecting the main page content.
    start_from_bottom_px = 0
    seed: Optional[Tuple[int, int]] = None
    block_bbox_px: Optional[Tuple[int, int, int, int]] = None

    while True:
        seed = _scan_up_for_first_text_band(
            ink,
            start_from_bottom_px=start_from_bottom_px,
            band_h=scan_band_h,
            min_ink_ratio=scan_min_ink_ratio,
            max_scan_frac=scan_max_scan_frac,
        )
        if seed is None:
            return None

        sy0, sy1 = seed

        # Guard 1: too thin -> likely page number / noise
        if (sy1 - sy0) < int(min_seed_height_px):
            start_from_bottom_px = int(h - sy0)
            continue

        # Guard 2: intersects too few words -> likely PAGE / small artifact
        if words_norm:
            band_box = (0.0, float(sy0), float(w), float(sy1))
            seed_words = [
                ww
                for ww in words_norm
                if _intersects((ww.x0, ww.y0, ww.x1, ww.y1), band_box, pad=2.0)
                and (ww.text or "").strip()
            ]
            if len(seed_words) < int(min_seed_words):
                start_from_bottom_px = int(h - sy0)
                continue

        # Expand seed band into a block
        block_y0, block_y1 = _expand_band_to_block(
            ink,
            seed_band=seed,
            band_h=scan_band_h,
            min_ink_ratio=max(0.0015, scan_min_ink_ratio * 0.8),
            gap_allow_bands=1,
        )

        block_bbox_px = _tight_bbox_from_ink(ink, block_y0, block_y1, pad=pad_px)
        if block_bbox_px is None:
            # move start above the seed and continue
            start_from_bottom_px = int(h - sy0)
            continue

        # Block sanity checks (avoid selecting main content when footer text is sparse)
        bx0, by0, bx1, by1 = block_bbox_px
        block_h = float(by1 - by0)
        block_w = float(bx1 - bx0)

        if by0 < int(h * float(min_block_top_frac)):
            # We've reached a substantial text region above the bottom band
            # without finding a plausible footer. Bail rather than selecting main content.
            return None

        if block_h > float(max_block_height_frac) * float(h):
            # Too tall to be a footer; treat as "no footer found".
            return None

        if block_w < float(min_block_width_frac) * float(w):
            # too narrow -> often side artifacts (page number, tiny stamps)
            start_from_bottom_px = int(h - by0)
            continue

        # Passed all checks
        break

    if block_bbox_px is None:
        return None

    # Pixel-only fallback when no words
    if not words_norm:
        x0, y0, x1, y1 = block_bbox_px
        return {
            "bbox": (x0, y0, x1, y1),
            "block_bbox_px": block_bbox_px,
            "font_h": 0.0,
            "lines_included": 0,
            "cutoff_y": int(y0),
        }

    # Words that intersect the pixel block bbox
    px_box_f = tuple(float(v) for v in block_bbox_px)
    cand_words = [
        ww
        for ww in words_norm
        if _intersects((ww.x0, ww.y0, ww.x1, ww.y1), px_box_f, pad=3.0)
        and (ww.text or "").strip()
    ]

    if not cand_words:
        x0, y0, x1, y1 = block_bbox_px
        return {
            "bbox": (x0, y0, x1, y1),
            "block_bbox_px": block_bbox_px,
            "font_h": 0.0,
            "lines_included": 0,
            "cutoff_y": int(y0),
        }

    lines = _group_words_into_lines(cand_words, y_tol=y_tol)
    if not lines:
        x0, y0, x1, y1 = block_bbox_px
        return {
            "bbox": (x0, y0, x1, y1),
            "block_bbox_px": block_bbox_px,
            "font_h": 0.0,
            "lines_included": 0,
            "cutoff_y": int(y0),
        }

    # Bottom-most line defines footer font
    lines_sorted = sorted(lines, key=lambda ln: _line_bbox(ln)[1], reverse=True)
    footer_line = lines_sorted[0]
    footer_font_h = _median_font_height(footer_line)
    if footer_font_h <= 0:
        footer_font_h = float(np.median([ww.h for ww in cand_words if ww.h > 0])) if cand_words else 0.0

    included = [footer_line]
    prev_top = _line_bbox(footer_line)[1]

    # Expand upward by font+spacing similarity
    for ln in lines_sorted[1:]:
        bx0, by0, bx1, by1 = _line_bbox(ln)
        if by0 > prev_top:
            continue

        ln_font_h = _median_font_height(ln)
        if ln_font_h <= 0 or footer_font_h <= 0:
            break

        if abs(ln_font_h - footer_font_h) / max(footer_font_h, 1e-6) > font_height_tol:
            break

        gap = prev_top - by1
        if gap > (max_line_gap_mult * footer_font_h):
            break

        included.append(ln)
        prev_top = by0

    # Build tight bbox from included lines
    x0 = int(min(wd.x0 for ln in included for wd in ln)) - pad_px
    y0 = int(min(wd.y0 for ln in included for wd in ln)) - pad_px
    x1 = int(max(wd.x1 for ln in included for wd in ln)) + pad_px
    y1 = int(max(wd.y1 for ln in included for wd in ln)) + pad_px

    final_bbox = _clip_bbox((x0, y0, x1, y1), w, h)

    return {
        "bbox": final_bbox,
        "block_bbox_px": block_bbox_px,
        "font_h": float(footer_font_h),
        "lines_included": int(len(included)),
        "cutoff_y": int(final_bbox[1]),
    }


# -----------------------------
# Backwards-compatible wrapper
# -----------------------------

def detect_footer_page_roi(words, img_bgr, y_tol: int = 8, exclude_roi=None, **kwargs):
    """Legacy name used by demo_header.py. Delegates to detect_footer_text_block_roi.

    Parameters
    ----------
    exclude_roi:
        Optional bbox (x0,y0,x1,y1) to ignore while finding footer (typically the
        page-number ROI).
    """
    return detect_footer_text_block_roi(words, img_bgr, y_tol=float(y_tol), exclude_roi=exclude_roi, **kwargs)
