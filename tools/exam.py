"""Exam question generation tool backed by semantic retrieval.

This module retrieves relevant academic chunks for a given subject and topic,
then returns them as context along with instructions for Claude to generate
exam-style questions.
"""

from __future__ import annotations

import logging
from typing import Any

from services.rag_service import normalize_user_id, semantic_search_academic_context

logger = logging.getLogger(__name__)


def generate_exam_questions(
    subject: str,
    topic: str = "",
    num_questions: int = 5,
    question_type: str = "mcq",
    user_id: str = "",
) -> dict[str, Any]:
    """
    Retrieve academic context and instruct Claude to generate exam questions.

    Use this when the user asks to generate practice questions, quiz questions,
    MCQs, true/false questions, or short-answer questions for a subject or topic.
    The tool retrieves relevant study material; Claude then generates the
    questions from that material.
    """
    resolved_user_id = normalize_user_id(user_id)
    logger.info(
        "generate_exam_questions input user_id=%s subject=%r topic=%r "
        "num_questions=%s question_type=%r",
        resolved_user_id, subject, topic, num_questions, question_type,
    )

    # ── Validate inputs ──────────────────────────────────────────────
    valid_types = {"mcq", "short_answer", "true_false"}
    normalized_type = question_type.strip().lower()
    if normalized_type not in valid_types:
        msg = f"question_type must be one of {sorted(valid_types)}, got '{question_type}'"
        logger.warning("generate_exam_questions rejected: %s", msg)
        return {"success": False, "message": msg}

    safe_num = max(1, min(int(num_questions), 10))
    query = f"{subject} {topic}".strip()

    # ── Retrieve relevant chunks via semantic search ─────────────────
    try:
        search_result = semantic_search_academic_context(
            query=query,
            user_id=resolved_user_id,
            subject=subject,
            top_k=5,
        )
    except Exception as exc:
        logger.exception("generate_exam_questions semantic search failed")
        return {"success": False, "message": str(exc)}

    matches = search_result.get("matches", [])
    if not matches:
        logger.info("generate_exam_questions: no chunks found for query=%r", query)
        return {
            "success": True,
            "message": (
                f"No study material found for subject='{subject}'"
                + (f", topic='{topic}'" if topic else "")
                + ". Ingest notes or PDFs first using ingest_academic_knowledge_base."
            ),
            "data": {"context_chunks": [], "num_questions": safe_num, "question_type": normalized_type},
        }

    # ── Build context and generation instructions ────────────────────
    context_chunks = []
    for match in matches:
        context_chunks.append({
            "text": match.get("text", ""),
            "source_type": match.get("source_type", ""),
            "source_id": match.get("source_id", ""),
            "subject": match.get("subject", ""),
            "page": match.get("page"),
            "score": match.get("score"),
        })

    type_label = {
        "mcq": "multiple-choice questions (4 options each, mark the correct answer)",
        "short_answer": "short-answer questions (expect 2-4 sentence answers)",
        "true_false": "true/false questions (state the correct answer and a brief explanation)",
    }

    result = {
        "success": True,
        "message": (
            f"Retrieved {len(context_chunks)} context chunks for subject='{subject}'"
            + (f", topic='{topic}'" if topic else "")
            + f". Generate {safe_num} {normalized_type} questions from this context."
        ),
        "data": {
            "subject": subject,
            "topic": topic,
            "num_questions": safe_num,
            "question_type": normalized_type,
            "context_chunks": context_chunks,
            "generation_instruction": (
                f"Using ONLY the provided context chunks, generate exactly {safe_num} "
                f"{type_label[normalized_type]} about '{subject}"
                + (f" — {topic}" if topic else "")
                + "'. Number each question. Base every question on facts present in the context."
            ),
        },
    }
    logger.info(
        "generate_exam_questions output chunks=%s questions_requested=%s type=%s",
        len(context_chunks), safe_num, normalized_type,
    )
    return result
