#!/usr/bin/env python3
from __future__ import annotations

"""header_roi.py

Rollback / simplified header detector.

Public API:
  detect_header_band(words, img_w, img_h, y_tol=8)
    -> (hits, header_line_bbox, header_table_bounds)

- `words` is a list of objects (or dicts) with fields/keys: x0,y0,x1,y1,text
- Returns:
    hits: List[str]                    # normalized hit labels (title-cased)
    header_line_bbox: (x0,y0,x1,y1)    # bbox for the full detected header line
    header_table_bounds: (x0,y0,x1,y1) # bbox spanning ONLY the header-token words

This version intentionally does NOT apply:
- weak/core token weighting
- table-row validation below the header

Those changes caused regressions on some document types, so this rolls back to the
previous behavior: pick the best "header-looking" line based on header keyword hits.
"""

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


# -----------------------------
# Data model (flexible)
# -----------------------------

@dataclass(frozen=True)
class WordBox:
    x0: int
    y0: int
    x1: int
    y1: int
    text: str


def _to_word(w: Any) -> WordBox:
    if isinstance(w, WordBox):
        return w
    if isinstance(w, dict):
        return WordBox(int(w['x0']), int(w['y0']), int(w['x1']), int(w['y1']), str(w.get('text', '')))
    return WordBox(int(getattr(w, 'x0')), int(getattr(w, 'y0')), int(getattr(w, 'x1')), int(getattr(w, 'y1')), str(getattr(w, 'text', '')))


def _center_y(w: WordBox) -> float:
    return 0.5 * (w.y0 + w.y1)


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


def group_words_into_lines(words: List[Any], y_tol: int = 8) -> List[List[WordBox]]:
    """Cluster word boxes into approximate text lines."""
    ws = [_to_word(w) for w in (words or [])]
    if not ws:
        return []

    # stable sort by y then x
    words_sorted = sorted(ws, key=lambda w: (w.y0, w.x0))
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


def _line_bbox(line: List[WordBox]) -> Tuple[int, int, int, int]:
    return (
        int(min(w.x0 for w in line)),
        int(min(w.y0 for w in line)),
        int(max(w.x1 for w in line)),
        int(max(w.y1 for w in line)),
    )


# -----------------------------
# Header detection configuration
# -----------------------------

HEADER_PHRASES = [
    # core banking / statement
    "date",
    "description",
    "details",
    "amount",
    "in",
    "out",
    "total",
    "tot",
    "balance",
    "moneyin",
    "moneyout",
    "inout",
    # common variants
    "rate",
    "quantity",
    "qty",
    # brokerage / trading statements
    "transaction",
    "transactions",
    "symbol",
    "symbols",
    "cusip",
    "type",
    "price",
    "trade",
    "settlement",
    "purchased",
    "sold",
    "acct",
    "mkt",
    "value",
    "portfolio",
    "income",
    "yield",
    "annual",
    "est",
    # additional common headers
    "reference",
    "narrative",
    "debit",
    "credit",
]

HEADER_ALLOWED = set(HEADER_PHRASES)

HEADER_MULTI = {
    "money in": "moneyin",
    "money out": "moneyout",
    "in out": "inout",
}


def _is_allowed_header_token(nt: str) -> bool:
    if not nt:
        return False
    if nt in HEADER_ALLOWED:
        return True
    # basic plural handling
    if len(nt) > 3:
        if nt.endswith("ies") and (nt[:-3] + "y") in HEADER_ALLOWED:
            return True
        if nt.endswith("es") and nt[:-2] in HEADER_ALLOWED:
            return True
        if nt.endswith("s") and nt[:-1] in HEADER_ALLOWED:
            return True
    return False


def _header_hits_for_line(line: List[WordBox]) -> Tuple[List[str], List[WordBox]]:
    """Return (hit_labels, contributing_words)."""
    raw_l = " ".join((w.text or "").strip().lower() for w in line if (w.text or "").strip())

    hits_norm: List[str] = []
    contrib: List[WordBox] = []

    # multi-word phrases first
    for phrase, token in HEADER_MULTI.items():
        if phrase in raw_l:
            # add canonical hit
            if token == "moneyin":
                hits_norm.append("Money In")
            elif token == "moneyout":
                hits_norm.append("Money Out")
            elif token == "inout":
                hits_norm.append("In Out")

    # per-word tokens
    for w in line:
        toks = _split_norm_tokens(w.text)
        if not toks:
            continue

        w_hit = False
        for t in toks:
            if _is_allowed_header_token(t):
                w_hit = True
                if t == "moneyin":
                    hits_norm.append("Money In")
                elif t == "moneyout":
                    hits_norm.append("Money Out")
                elif t == "inout":
                    hits_norm.append("In Out")
                else:
                    hits_norm.append(t.title())

        if w_hit:
            contrib.append(w)

    # de-dup but preserve order
    seen = set()
    hits: List[str] = []
    for h in hits_norm:
        if h not in seen:
            seen.add(h)
            hits.append(h)

    return hits, contrib


def detect_header_band(
    words: List[Any],
    img_w: int,
    img_h: int,
    y_tol: int = 8,
) -> Tuple[List[str], Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    """Detect a likely header line and return (hits, full_line_bbox, header_token_bbox)."""

    lines = group_words_into_lines(words, y_tol=y_tol)
    if not lines:
        return [], None, None

    # Scan all lines, choose the one with the most header hits.
    best_hits: List[str] = []
    best_line_bbox: Optional[Tuple[int, int, int, int]] = None
    best_token_bbox: Optional[Tuple[int, int, int, int]] = None

    for ln in lines:
        hits, contrib = _header_hits_for_line(ln)
        if len(hits) < 2:
            continue

        # favor lines in the upper 75% of the page (most headers are not in the very bottom)
        x0, y0, x1, y1 = _line_bbox(ln)
        if y0 > int(img_h * 0.85):
            continue

        if len(hits) > len(best_hits):
            best_hits = hits
            best_line_bbox = (x0, y0, x1, y1)
            if contrib:
                best_token_bbox = (
                    int(min(w.x0 for w in contrib)),
                    int(min(w.y0 for w in contrib)),
                    int(max(w.x1 for w in contrib)),
                    int(max(w.y1 for w in contrib)),
                )
            else:
                best_token_bbox = best_line_bbox

    return best_hits, best_line_bbox, best_token_bbox
