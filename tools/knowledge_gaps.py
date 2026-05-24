"""Knowledge gap detection tool backed by MongoDB.

This module cross-references the subjects found in assignments against
the number of ingested academic chunks per subject, flagging subjects
with little or no study material as knowledge gaps.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import get_settings
from database.mongo import get_collection
from services.rag_service import normalize_user_id

logger = logging.getLogger(__name__)

# Subjects with fewer chunks than this threshold are flagged as gaps.
CHUNK_THRESHOLD = 3


def find_knowledge_gaps(
    user_id: str = "",
) -> dict[str, Any]:
    """
    Identify subjects where the student has assignments but few or no notes.

    Use this when the user asks about weak areas, missing study material,
    knowledge gaps, or wants to know which subjects need more notes or PDFs
    ingested. The tool compares assignment subjects against ingested chunk
    counts and flags subjects that are under-represented.
    """
    resolved_user_id = normalize_user_id(user_id)
    settings = get_settings()
    logger.info("find_knowledge_gaps input user_id=%s", resolved_user_id)

    try:
        # ── Distinct subjects from notes ────────────────────────────
        note_subjects = get_collection("notes").distinct(
            "subject", {"user_id": resolved_user_id}
        )

        # ── Distinct subjects from assignments ──────────────────────
        assignment_subjects = get_collection("assignments").distinct(
            "subject", {"user_id": resolved_user_id}
        )

        # ── For each assignment subject, count chunks ───────────────
        chunk_col = get_collection(settings.chunk_collection)
        all_subjects = sorted(set(
            s for s in (note_subjects + assignment_subjects) if s
        ))

        report: list[dict[str, Any]] = []
        gaps: list[str] = []

        for subject in all_subjects:
            assignment_count = get_collection("assignments").count_documents(
                {"user_id": resolved_user_id, "subject": {"$regex": f"^{subject}$", "$options": "i"}}
            )
            chunk_count = chunk_col.count_documents(
                {"user_id": resolved_user_id, "subject": {"$regex": f"^{subject}$", "$options": "i"}}
            )

            is_gap = chunk_count < CHUNK_THRESHOLD
            entry: dict[str, Any] = {
                "subject": subject,
                "assignment_count": assignment_count,
                "chunk_count": chunk_count,
                "is_knowledge_gap": is_gap,
            }
            report.append(entry)
            if is_gap:
                gaps.append(subject)

        result: dict[str, Any] = {
            "success": True,
            "message": (
                f"Analysed {len(all_subjects)} subject(s). "
                f"Found {len(gaps)} knowledge gap(s) "
                f"(subjects with fewer than {CHUNK_THRESHOLD} chunks)."
            ),
            "data": {
                "user_id": resolved_user_id,
                "total_subjects": len(all_subjects),
                "gap_threshold": CHUNK_THRESHOLD,
                "knowledge_gaps": gaps,
                "gap_count": len(gaps),
                "per_subject_report": report,
            },
        }
        logger.info(
            "find_knowledge_gaps output subjects=%s gaps=%s",
            len(all_subjects), len(gaps),
        )
        return result

    except Exception as exc:
        logger.exception("find_knowledge_gaps failed")
        return {"success": False, "message": str(exc)}
