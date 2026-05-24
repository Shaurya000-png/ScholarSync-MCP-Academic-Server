"""PDF text extraction helpers.

Supports PyMuPDF (primary) with PyPDF2 fallback, table extraction,
OCR fallback for scanned PDFs via pytesseract, and automatic subject
detection from filename and content keywords.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Subject keyword mapping for auto-detection ──────────────────────
SUBJECT_KEYWORDS: dict[str, list[str]] = {
    "DBMS": ["database", "dbms", "sql", "normalization", "relational", "er diagram", "transaction"],
    "OS": ["operating system", "process", "thread", "deadlock", "scheduling", "paging", "semaphore"],
    "AI": ["artificial intelligence", "machine learning", "neural network", "search algorithm", "heuristic", "deep learning"],
    "CN": ["computer network", "tcp", "udp", "osi model", "routing", "ip address", "dns", "subnet"],
    "Maths": ["calculus", "matrix", "differential", "integral", "probability", "statistics", "linear algebra"],
    "Physics": ["physics", "mechanics", "thermodynamics", "electromagnetism", "quantum", "optics", "kinematics"],
    "DSA": ["data structure", "algorithm", "sorting", "linked list", "binary tree", "graph", "stack", "queue", "hash"],
    "SE": ["software engineering", "sdlc", "agile", "scrum", "uml", "requirement", "testing"],
}


def _detect_subject(filename: str, text_snippet: str) -> str:
    """
    Attempt to detect the academic subject from filename and text content.

    Returns the best-matching subject key or an empty string.
    """
    combined = f"{filename} {text_snippet}".lower()

    best_subject = ""
    best_count = 0

    for subject, keywords in SUBJECT_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw.lower() in combined)
        if count > best_count:
            best_count = count
            best_subject = subject

    return best_subject if best_count >= 1 else ""


def _extract_tables_pymupdf(page: Any) -> list[list[list[str]]]:
    """
    Extract tables from a PyMuPDF page object.

    Returns a list of tables, each table being a list of rows,
    each row being a list of cell strings.
    """
    tables: list[list[list[str]]] = []
    try:
        # fitz page.find_tables() is available in PyMuPDF ≥ 1.23.0
        tab_finder = page.find_tables()
        for table in tab_finder.tables:
            extracted = table.extract()
            if extracted:
                # Clean None cells
                clean_table = [
                    [str(cell) if cell is not None else "" for cell in row]
                    for row in extracted
                ]
                tables.append(clean_table)
    except AttributeError:
        # find_tables not available in older PyMuPDF versions
        logger.debug("page.find_tables() not available; skipping table extraction")
    except Exception as exc:
        logger.debug("Table extraction failed for page: %s", exc)
    return tables


def _format_table_text(tables: list[list[list[str]]]) -> str:
    """Format extracted tables into readable text with [TABLE] prefix."""
    if not tables:
        return ""

    parts: list[str] = []
    for table in tables:
        rows = []
        for row in table:
            rows.append(" | ".join(row))
        parts.append("[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]")
    return "\n\n".join(parts)


def _try_ocr_page(page: Any) -> str:
    """
    Attempt OCR on a PyMuPDF page rendered as an image.

    Returns the OCR text or an empty string if pytesseract is unavailable.
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        # Render page at 200 DPI for OCR
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except ImportError:
        logger.warning(
            "pytesseract or Pillow not installed; skipping OCR. "
            "Install with: pip install pytesseract Pillow"
        )
        return ""
    except Exception as exc:
        logger.warning("OCR failed for page: %s", exc)
        return ""


def extract_pdf_pages(
    pdf_path: Path,
    max_pages: int = 30,
    enable_ocr: bool = True,
    enable_tables: bool = True,
) -> tuple[int, list[dict[str, Any]], str]:
    """
    Extract PDF pages with PyMuPDF, falling back to PyPDF2 if needed.

    Returns (total_pages, extracted_pages, extractor_name).
    Each extracted page dict has keys: page, text, tables (optional),
    ocr_used (bool).
    """
    safe_pages = max(1, min(int(max_pages), 100))

    try:
        import fitz

        extracted: list[dict[str, Any]] = []
        with fitz.open(str(pdf_path)) as document:
            total_pages = document.page_count

            # First pass: extract text and tables
            raw_texts: list[str] = []
            for index in range(min(safe_pages, total_pages)):
                page = document.load_page(index)
                text = " ".join(page.get_text("text").split())

                # Table extraction
                tables: list[list[list[str]]] = []
                table_text = ""
                if enable_tables:
                    tables = _extract_tables_pymupdf(page)
                    table_text = _format_table_text(tables)

                raw_texts.append(text)
                page_entry: dict[str, Any] = {
                    "page": index + 1,
                    "text": text,
                    "tables": tables,
                    "ocr_used": False,
                }
                if table_text:
                    page_entry["text"] = f"{text}\n\n{table_text}" if text else table_text
                extracted.append(page_entry)

            # OCR fallback: check if average text length per page is < 100
            if enable_ocr and extracted:
                avg_chars = sum(len(e.get("text", "")) for e in extracted) / len(extracted)
                if avg_chars < 100:
                    logger.info(
                        "PDF '%s' appears scanned (avg %.0f chars/page); attempting OCR",
                        pdf_path.name, avg_chars,
                    )
                    for idx, entry in enumerate(extracted):
                        if len(entry.get("text", "")) < 100:
                            page = document.load_page(idx)
                            ocr_text = _try_ocr_page(page)
                            if ocr_text:
                                entry["text"] = f"{entry['text']} {ocr_text}".strip()
                                entry["ocr_used"] = True

            # Filter out pages with no content at all
            extracted = [e for e in extracted if e.get("text", "").strip()]

        # Detect subject from filename + first 500 chars of content
        all_text = " ".join(e["text"] for e in extracted[:3])
        detected_subject = _detect_subject(pdf_path.stem, all_text[:500])

        return total_pages, extracted, "pymupdf", detected_subject

    except Exception:
        # Fallback to PyPDF2
        from PyPDF2 import PdfReader

        reader = PdfReader(str(pdf_path))
        extracted = []
        for index, page in enumerate(reader.pages[:safe_pages]):
            text = " ".join((page.extract_text() or "").split())
            if text:
                extracted.append({
                    "page": index + 1,
                    "text": text,
                    "tables": [],
                    "ocr_used": False,
                })

        all_text = " ".join(e["text"] for e in extracted[:3])
        detected_subject = _detect_subject(pdf_path.stem, all_text[:500])

        return len(reader.pages), extracted, "pypdf2", detected_subject
