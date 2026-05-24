"""MCP tool wrappers for semantic RAG ingestion and search."""

from __future__ import annotations

import logging
from typing import Any

from config.settings import get_settings
from services.rag_service import (
    ingest_academic_materials,
    normalize_user_id,
    semantic_search_academic_context,
)

logger = logging.getLogger(__name__)


def _run_search(
    query: str,
    user_id: str,
    subject: str,
    top_k: int,
    min_score: float,
) -> dict[str, Any]:
    """Route to hybrid or pure-vector search based on configuration."""
    settings = get_settings()
    resolved_user_id = normalize_user_id(user_id)

    if settings.search_mode.lower() == "hybrid":
        try:
            from services.hybrid_search import hybrid_search

            matches = hybrid_search(
                query=query,
                user_id=resolved_user_id,
                subject=subject,
                top_k=top_k,
            )
            return {
                "success": True,
                "user_id": resolved_user_id,
                "query": query,
                "subject": subject,
                "search_mode": "hybrid_rrf",
                "match_count": len(matches),
                "matches": matches,
            }
        except Exception as exc:
            logger.warning("Hybrid search failed; falling back to vector: %s", exc)

    # Default: vector (or keyword fallback handled inside semantic_search)
    return semantic_search_academic_context(
        query=query,
        user_id=resolved_user_id,
        subject=subject,
        top_k=top_k,
        min_score=min_score,
    )


def ingest_academic_knowledge_base(
    user_id: str = "",
    subject: str = "",
    reset_existing: bool = False,
    include_mongodb_notes: bool = True,
    include_local_notes: bool = True,
    include_pdfs: bool = True,
    max_pdf_pages: int = 50,
) -> dict[str, Any]:
    """
    Ingest notes and PDFs into MongoDB-backed semantic chunks.

    Use this after adding notes or PDFs so ScholarSync can perform real
    semantic retrieval instead of only keyword search.
    """
    return ingest_academic_materials(
        user_id=user_id,
        subject=subject,
        reset_existing=reset_existing,
        include_mongodb_notes=include_mongodb_notes,
        include_local_notes=include_local_notes,
        include_pdfs=include_pdfs,
        max_pdf_pages=max_pdf_pages,
    )


def semantic_search_academic_knowledge(
    query: str,
    user_id: str = "",
    subject: str = "",
    top_k: int = 5,
    min_score: float = 0.0,
) -> dict[str, Any]:
    """
    Search the embedded academic knowledge base using semantic similarity.

    Use this for concept questions, fuzzy search, exam prep queries, and RAG
    retrieval across notes and PDFs.
    """
    return _run_search(
        query=query,
        user_id=user_id,
        subject=subject,
        top_k=top_k,
        min_score=min_score,
    )
