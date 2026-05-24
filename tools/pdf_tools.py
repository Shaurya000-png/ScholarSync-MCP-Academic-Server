"""PDF extraction and summary-preparation tools."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from services.pdf_extraction import extract_pdf_pages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = PROJECT_ROOT / "pdfs"

logger = logging.getLogger(__name__)


def _find_pdf(file_name: str) -> Path | None:
    if not PDF_DIR.exists():
        return None

    if file_name:
        direct = PDF_DIR / file_name
        if direct.exists() and direct.suffix.lower() == ".pdf":
            return direct

        needle = file_name.lower().replace(".pdf", "")
        for path in sorted(PDF_DIR.glob("*.pdf")):
            if needle in path.stem.lower():
                return path

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    return pdfs[0] if len(pdfs) == 1 else None


def summarize_pdf(file_name: str = "", max_pages: int = 10, query: str = "") -> dict[str, Any]:
    """
    Extract study PDF text and return summary-ready content.

    Use this when the user asks Claude to summarize a stored PDF, extract key
    points, explain a PDF document, or review PDF study material.
    """
    logger.info("summarize_pdf input file_name=%r max_pages=%s query=%r", file_name, max_pages, query)
    safe_pages = max(1, min(int(max_pages), 30))
    pdf_path = _find_pdf(file_name)

    if not pdf_path:
        available = [path.name for path in PDF_DIR.glob("*.pdf")] if PDF_DIR.exists() else []
        return {
            "success": False,
            "error": "PDF not found. Put PDFs in the pdfs folder or pass a matching file_name.",
            "available_pdfs": available,
        }

    try:
        result = extract_pdf_pages(pdf_path, max_pages=safe_pages)
        # Support both old 3-tuple and new 4-tuple return format
        if len(result) == 4:
            total_pages, extracted_pages, extractor, detected_subject = result
        else:
            total_pages, extracted_pages, extractor = result
            detected_subject = ""

        combined_text = "\n\n".join(str(page["text"]) for page in extracted_pages)
        excerpt = combined_text[:8000]

        result = {
            "success": True,
            "file_name": pdf_path.name,
            "total_pages": total_pages,
            "pages_extracted": len(extracted_pages),
            "extractor": extractor,
            "detected_subject": detected_subject,
            "query": query,
            "extracted_text_excerpt": excerpt,
            "summary_instruction": "Summarize the extracted PDF text into clear headings, key points, and exam-focused takeaways.",
        }
        logger.info("summarize_pdf output file=%s pages=%s", pdf_path.name, len(extracted_pages))
        return result
    except Exception as exc:
        logger.exception("summarize_pdf failed")
        return {"success": False, "error": str(exc), "file_name": pdf_path.name}
