"""Academic notes search and lightweight retrieval tools."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config.settings import get_settings
from database.mongo import get_collection, regex_filter, serialize_documents
from services.rag_service import normalize_user_id, semantic_search_academic_context

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTES_DIR = PROJECT_ROOT / "notes"

logger = logging.getLogger(__name__)


def _excerpt(text: str, query: str = "", limit: int = 1200) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned

    if query:
        index = cleaned.lower().find(query.lower())
        if index > 0:
            start = max(0, index - 180)
            return cleaned[start : start + limit]

    return cleaned[:limit]


def _search_local_notes(query: str, subject: str, limit: int) -> list[dict[str, Any]]:
    if not NOTES_DIR.exists():
        return []

    matches: list[dict[str, Any]] = []
    query_lower = query.lower().strip()
    subject_lower = subject.lower().strip()

    for path in sorted(NOTES_DIR.glob("*.txt")):
        content = path.read_text(encoding="utf-8", errors="ignore")
        searchable = f"{path.stem} {content}".lower()

        query_match = not query_lower or query_lower in searchable
        subject_match = not subject_lower or subject_lower in path.stem.lower() or subject_lower in searchable

        if query_match and subject_match:
            matches.append(
                {
                    "source": "local_file",
                    "file_name": path.name,
                    "subject": path.stem,
                    "content_excerpt": _excerpt(content, query),
                }
            )

        if len(matches) >= limit:
            break

    return matches


def search_notes(
    query: str,
    subject: str = "",
    limit: int = 5,
    include_local_files: bool = True,
    user_id: str = "",
) -> dict[str, Any]:
    """
    Search academic notes by topic, keyword, subject, or concept.

    Use this when the user asks to find notes, search academic material,
    revise a concept, or locate subject-specific study content.
    """
    resolved_user_id = normalize_user_id(user_id)
    logger.info("search_notes input user_id=%s query=%r subject=%r limit=%s", resolved_user_id, query, subject, limit)
    safe_limit = max(1, min(int(limit), 20))

    filters: list[dict[str, Any]] = [{"user_id": resolved_user_id}]
    text_filter = regex_filter(["subject", "topic", "title", "content", "tags"], query)
    if text_filter:
        filters.append(text_filter)
    if subject:
        filters.append({"subject": {"$regex": subject, "$options": "i"}})

    mongo_query = {"$and": filters} if filters else {}

    try:
        documents = list(get_collection("notes").find(mongo_query).limit(safe_limit))
        mongo_matches = serialize_documents(documents)
        for item in mongo_matches:
            if "content" in item:
                item["content_excerpt"] = _excerpt(str(item["content"]), query)
                item.pop("content", None)
            item["source"] = "mongodb"

        remaining = max(0, safe_limit - len(mongo_matches))
        local_matches = _search_local_notes(query, subject, remaining) if include_local_files and remaining else []

        result = {
            "success": True,
            "user_id": resolved_user_id,
            "query": query,
            "subject": subject,
            "count": len(mongo_matches) + len(local_matches),
            "matches": mongo_matches + local_matches,
        }
        logger.info("search_notes output count=%s", result["count"])
        return result
    except Exception as exc:
        logger.exception("search_notes failed")
        return {"success": False, "error": str(exc), "matches": []}


def retrieve_academic_context(question: str, subject: str = "", limit: int = 5, user_id: str = "") -> dict[str, Any]:
    """
    Retrieve academic context from notes for question answering.

    Use this before answering conceptual academic questions so Claude can ground
    its explanation in the student's notes and available study material.
    """
    resolved_user_id = normalize_user_id(user_id)
    logger.info("retrieve_academic_context input user_id=%s question=%r subject=%r", resolved_user_id, question, subject)

    settings = get_settings()
    if settings.search_mode.lower() == "hybrid":
        try:
            from services.hybrid_search import hybrid_search

            matches = hybrid_search(
                query=question,
                user_id=resolved_user_id,
                subject=subject,
                top_k=limit,
            )
            semantic_result = {
                "success": True,
                "match_count": len(matches),
                "matches": matches,
            }
        except Exception as exc:
            logger.warning("Hybrid search failed in retrieve_academic_context: %s", exc)
            semantic_result = semantic_search_academic_context(
                query=question, subject=subject, top_k=limit, user_id=resolved_user_id,
            )
    else:
        semantic_result = semantic_search_academic_context(
            query=question,
            subject=subject,
            top_k=limit,
            user_id=resolved_user_id,
        )

    if semantic_result.get("success") and semantic_result.get("match_count", 0) > 0:
        context_items = []
        for match in semantic_result.get("matches", []):
            context_items.append(
                {
                    "source": match.get("source_type"),
                    "source_id": match.get("source_id"),
                    "subject": match.get("subject"),
                    "topic": match.get("title"),
                    "page": match.get("page"),
                    "score": match.get("score"),
                    "context": match.get("text"),
                }
            )

        result = {
            "success": True,
            "user_id": resolved_user_id,
            "retrieval_mode": "semantic_vector",
            "question": question,
            "retrieved_count": len(context_items),
            "context": context_items,
            "answer_instruction": "Answer using the retrieved semantic context. Mention missing context if the matches are weak.",
        }
        logger.info("retrieve_academic_context output mode=semantic count=%s", result["retrieved_count"])
        return result

    search_result = search_notes(
        query=question,
        subject=subject,
        limit=limit,
        include_local_files=True,
        user_id=resolved_user_id,
    )

    context_items = []
    for match in search_result.get("matches", []):
        context_items.append(
            {
                "source": match.get("source"),
                "subject": match.get("subject"),
                "topic": match.get("topic") or match.get("title") or match.get("file_name"),
                "context": match.get("content_excerpt"),
            }
        )

    result = {
        "success": search_result.get("success", False),
        "user_id": resolved_user_id,
        "retrieval_mode": "keyword_fallback",
        "question": question,
        "retrieved_count": len(context_items),
        "context": context_items,
        "answer_instruction": "Use the retrieved context to answer the student clearly. If context is weak, say what is missing.",
    }
    logger.info("retrieve_academic_context output count=%s", result["retrieved_count"])
    return result
