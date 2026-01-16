#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

def detect_page_number_roi(
    words: List[Any],
    img_shape: Tuple[int, int],
    *,
    y_tol: int = 8,
    bottom_lines_to_search: int = 3,
    y_min_frac: float = 0.70,
) -> Optional[Dict[str, Any]]:
    """
    Detect a tight ROI around the page label 'Page N' near the bottom.

    Returns bbox tightly surrounding ONLY the token(s) that form 'Page N':
      - merged: one token like 'Page 2'
      - split: two tokens 'Page' and '2' on the same line
    """
    img_h, img_w = img_shape

    y_min = int(img_h * y_min_frac)
    band = [wb for wb in words if getattr(wb, "text", "").strip() and getattr(wb, "y0", 0) >= y_min]
    if not band:
        return None

    def y_center(wb) -> float:
        return (wb.y0 + wb.y1) / 2.0

    band.sort(key=lambda wb: (y_center(wb), wb.x0))

    lines: List[List[Any]] = []
    cur: List[Any] = []
    cur_y: Optional[float] = None

    for wb in band:
        yc = y_center(wb)
        if cur_y is None or abs(yc - cur_y) <= y_tol:
            cur.append(wb)
            cur_y = yc if cur_y is None else (0.7 * cur_y + 0.3 * yc)
        else:
            cur.sort(key=lambda w: w.x0)
            lines.append(cur)
            cur = [wb]
            cur_y = yc
    if cur:
        cur.sort(key=lambda w: w.x0)
        lines.append(cur)

    lines_sorted_bottom = sorted(lines, key=lambda ln: max(w.y1 for w in ln), reverse=True)
    lines_to_check = lines_sorted_bottom[:max(1, int(bottom_lines_to_search))]

    page_only_re = re.compile(r"^page$", re.IGNORECASE)
    num_re = re.compile(r"^\d{1,4}$")
    merged_re = re.compile(r"^page\s*\d{1,4}$", re.IGNORECASE)

    best = None  # (y0, bbox, text)

    for line in lines_to_check:
        merged_tokens = [w for w in line if merged_re.match(w.text.strip())]
        if merged_tokens:
            w0 = sorted(merged_tokens, key=lambda w: (w.y0, w.x0), reverse=True)[0]
            bbox = (int(w0.x0), int(w0.y0), int(w0.x1), int(w0.y1))
            txt = w0.text.strip()
            if best is None or w0.y0 > best[0]:
                best = (w0.y0, bbox, txt)
            continue

        page_tokens = [w for w in line if page_only_re.match(w.text.strip())]
        num_tokens = [w for w in line if num_re.match(w.text.strip())]
        if not page_tokens or not num_tokens:
            continue

        best_pair = None  # (dist, p, n)
        for p in page_tokens:
            right = [n for n in num_tokens if n.x0 >= p.x1 - 2]
            cands = right if right else num_tokens
            for n in cands:
                dist = abs(n.x0 - p.x1)
                if best_pair is None or dist < best_pair[0]:
                    best_pair = (dist, p, n)

        if best_pair is None:
            continue

        _, p, n = best_pair
        x0 = min(p.x0, n.x0)
        y0 = min(p.y0, n.y0)
        x1 = max(p.x1, n.x1)
        y1 = max(p.y1, n.y1)
        bbox = (int(x0), int(y0), int(x1), int(y1))
        txt = f"{p.text.strip()} {n.text.strip()}"

        if best is None or y0 > best[0]:
            best = (y0, bbox, txt)

    if best is None:
        return None

    _, bbox, txt = best
    return {"bbox": bbox, "text": txt, "method": "page_number_tight"}
