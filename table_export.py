#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from typing import Any, Iterable, List, Optional, Sequence, Tuple


# ---------------------------
# Date parsing/formatting
# ---------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _parse_dd_mmm_optional_year(s: str) -> Optional[Tuple[int, int, Optional[int]]]:
    """
    Accepts common date variants:
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
    mon_txt = m.group(2)
    if mon_txt not in _MONTHS:
        return None
    mm = _MONTHS[mon_txt]

    yy_raw = m.group(3)
    if yy_raw is None:
        yyyy = None
    else:
        if len(yy_raw) == 2:
            yyyy = 2000 + int(yy_raw)
        else:
            yyyy = int(yy_raw)

    return dd, mm, yyyy

def format_date_dd_mmm_yy(date_str: str, statement_year: Optional[int]) -> str:
    """
    Ensures output format: 'dd mmm yy' (e.g., '01 Jan 24').
    If date has no year, uses statement_year.
    If cannot parse, returns the trimmed original.
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


# ---------------------------
# Table export
# ---------------------------

def _find_date_col_index(header: Sequence[str]) -> Optional[int]:
    for i, h in enumerate(header):
        if _norm_spaces(h).lower() == "date":
            return i
    return None

def _pad_or_truncate(row: Sequence[str], ncols: int) -> List[str]:
    r = list(row)
    if len(r) < ncols:
        r.extend([""] * (ncols - len(r)))
    elif len(r) > ncols:
        r = r[:ncols]
    return r

def write_tab_delimited_txt(
    pdf_path: str,
    out_dir: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    statement_year: Optional[int],
) -> str:
    """
    Writes ONE output .txt:
      - same basename as pdf
      - tab-delimited
      - header once at top
      - all rows appended
      - date column normalized to dd mmm yy and missing year filled from statement_year
    Returns output path.
    """
    os.makedirs(out_dir, exist_ok=True)

    clean_header = [_norm_spaces(h) for h in header]
    ncols = len(clean_header)
    date_idx = _find_date_col_index(clean_header)

    fixed_rows: List[List[str]] = []
    for r in rows:
        rr = _pad_or_truncate([_norm_spaces(x) for x in r], ncols)
        if date_idx is not None and 0 <= date_idx < ncols:
            rr[date_idx] = format_date_dd_mmm_yy(rr[date_idx], statement_year)
        fixed_rows.append(rr)

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(out_dir, f"{base}.txt")

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(clean_header) + "\n")
        for r in fixed_rows:
            f.write("\t".join(r) + "\n")

    return out_path


# ---------------------------
# Optional standalone mode
# ---------------------------

def _load_extracted_json(path: str) -> Tuple[str, str, List[str], List[List[str]], Optional[int]]:
    """
    Expected JSON shape:
    {
      "pdf_path": "...",
      "out_dir": "...",
      "header": ["Date", "Description", ...],
      "rows": [["01 Jan", "Foo", ...], ...],
      "statement_year": 2024
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    pdf_path = obj.get("pdf_path") or ""
    out_dir = obj.get("out_dir") or "."
    header = obj.get("header") or []
    rows = obj.get("rows") or []
    statement_year = obj.get("statement_year")

    if not isinstance(header, list) or not all(isinstance(x, str) for x in header):
        raise ValueError("Invalid JSON: 'header' must be a list[str]")
    if not isinstance(rows, list) or not all(isinstance(r, list) for r in rows):
        raise ValueError("Invalid JSON: 'rows' must be a list[list[str]]")
    if statement_year is not None:
        statement_year = int(statement_year)

    return pdf_path, out_dir, header, rows, statement_year

def main() -> int:
    p = argparse.ArgumentParser(description="Export extracted table rows to one tab-delimited .txt file.")
    p.add_argument("--from_json", help="Standalone mode: path to JSON containing extracted header/rows.")
    args = p.parse_args()

    if not args.from_json:
        print("Nothing to do. Use --from_json <path> or import this module from demo_header.py.")
        return 2

    pdf_path, out_dir, header, rows, statement_year = _load_extracted_json(args.from_json)
    out_path = write_tab_delimited_txt(pdf_path, out_dir, header, rows, statement_year)
    print(out_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
