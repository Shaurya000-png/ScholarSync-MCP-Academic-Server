"""Multilingual search tool for academic content.

Allows students to query in any language (Hindi, Marathi, etc.).
The query is auto-detected and translated to English before running
semantic (or hybrid) search over the English knowledge base.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import get_settings
from services.rag_service import normalize_user_id, semantic_search_academic_context

logger = logging.getLogger(__name__)


def translate_and_search(
    query: str,
    target_language: str = "en",
    user_id: str = "",
    subject: str = "",
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Translate a query from any language to English and search academic content.

    Use this when the student asks a question in Hindi, Marathi, or any
    non-English language.  The tool detects the language, translates the
    query to English, and runs semantic (or hybrid) search on the translated
    text so that English notes and PDFs are correctly retrieved.
    """
    resolved_user_id = normalize_user_id(user_id)
    logger.info(
        "translate_and_search input user_id=%s query=%r target_language=%r subject=%r",
        resolved_user_id, query, target_language, subject,
    )

    original_query = query
    detected_language = "en"
    translated_query = query

    # ── Detect and translate ─────────────────────────────────────────
    try:
        from langdetect import detect
        from deep_translator import GoogleTranslator

        detected_language = detect(query)

        if detected_language != target_language:
            translated_query = GoogleTranslator(source=detected_language, target=target_language).translate(query)
            logger.info(
                "translate_and_search translated from %s to %s: %r -> %r",
                detected_language, target_language, query, translated_query,
            )
        else:
            logger.info("translate_and_search: query already in target language (%s)", target_language)

    except ImportError:
        logger.warning(
            "langdetect or deep-translator not installed; skipping translation. "
            "Install with: pip install langdetect deep-translator"
        )
    except Exception as exc:
        logger.warning("Translation failed; searching with original query: %s", exc)

    # ── Run search on the (translated) query ─────────────────────────
    settings = get_settings()
    try:
        if settings.search_mode.lower() == "hybrid":
            try:
                from services.hybrid_search import hybrid_search

                matches = hybrid_search(
                    query=translated_query,
                    user_id=resolved_user_id,
                    subject=subject,
                    top_k=top_k,
                )
                search_results = {
                    "success": True,
                    "search_mode": "hybrid_rrf",
                    "match_count": len(matches),
                    "matches": matches,
                }
            except Exception as exc:
                logger.warning("Hybrid search failed in translate_and_search: %s", exc)
                search_results = semantic_search_academic_context(
                    query=translated_query,
                    user_id=resolved_user_id,
                    subject=subject,
                    top_k=top_k,
                )
        else:
            search_results = semantic_search_academic_context(
                query=translated_query,
                user_id=resolved_user_id,
                subject=subject,
                top_k=top_k,
            )
    except Exception as exc:
        logger.exception("translate_and_search search failed")
        return {"success": False, "message": str(exc)}

    result: dict[str, Any] = {
        "success": search_results.get("success", False),
        "message": (
            f"Detected language: {detected_language}. "
            + (f"Translated to {target_language}. " if detected_language != target_language else "")
            + f"Found {search_results.get('match_count', 0)} result(s)."
        ),
        "data": {
            "original_query": original_query,
            "detected_language": detected_language,
            "translated_query": translated_query,
            "target_language": target_language,
            "search_results": search_results.get("matches", []),
            "match_count": search_results.get("match_count", 0),
        },
    }
    logger.info(
        "translate_and_search output detected=%s matches=%s",
        detected_language, search_results.get("match_count", 0),
    )
    return result
