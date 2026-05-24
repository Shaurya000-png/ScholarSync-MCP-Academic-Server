"""Timetable and subject information tools backed by MongoDB."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from database.mongo import get_collection, serialize_document, serialize_documents
from services.rag_service import normalize_user_id

logger = logging.getLogger(__name__)

WEEKDAYS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}


def _resolve_day(day: str) -> str:
    normalized = day.strip().lower()
    today = datetime.now().date()

    if not normalized or normalized == "today":
        return today.strftime("%A")
    if normalized == "tomorrow":
        return (today + timedelta(days=1)).strftime("%A")
    if normalized in WEEKDAYS:
        return normalized.title()
    return day.strip().title()


def get_schedule(day: str = "today", subject: str = "", user_id: str = "") -> dict[str, Any]:
    """
    Get timetable entries for a day or subject.

    Use this when the user asks about today's classes, tomorrow's schedule,
    Friday timetable, next lectures, or subject-specific class timings.
    """
    resolved_day = _resolve_day(day)
    resolved_user_id = normalize_user_id(user_id)
    logger.info("get_schedule input user_id=%s day=%r resolved_day=%r subject=%r", resolved_user_id, day, resolved_day, subject)

    filters: list[dict[str, Any]] = [{"user_id": resolved_user_id}]
    if resolved_day:
        filters.append({"day": {"$regex": f"^{resolved_day}$", "$options": "i"}})
    if subject:
        filters.append({"subject": {"$regex": subject, "$options": "i"}})

    mongo_query = {"$and": filters} if filters else {}

    try:
        documents = list(get_collection("timetable").find(mongo_query).sort("start_time", 1))
        schedule = serialize_documents(documents)
        logger.info("get_schedule output count=%s", len(schedule))
        return {"success": True, "day": resolved_day, "count": len(schedule), "schedule": schedule}
    except Exception as exc:
        logger.exception("get_schedule failed")
        return {"success": False, "error": str(exc), "schedule": []}


def get_subject_info(subject: str, user_id: str = "") -> dict[str, Any]:
    """
    Get course or subject details from MongoDB.

    Use this when the user asks for subject information, course details,
    teacher/faculty names, credits, syllabus, or lecture metadata.
    """
    resolved_user_id = normalize_user_id(user_id)
    logger.info("get_subject_info input user_id=%s subject=%r", resolved_user_id, subject)
    if not subject.strip():
        return {"success": False, "error": "subject is required"}

    try:
        document = get_collection("subjects").find_one(
            {"user_id": resolved_user_id, "subject": {"$regex": subject, "$options": "i"}}
        )
        if document:
            logger.info("get_subject_info output source=subjects id=%s", document.get("_id"))
            return {"success": True, "subject_info": serialize_document(document)}

        timetable_entries = list(
            get_collection("timetable")
            .find({"user_id": resolved_user_id, "subject": {"$regex": subject, "$options": "i"}})
            .sort("day", 1)
        )
        logger.info("get_subject_info output source=timetable count=%s", len(timetable_entries))
        return {
            "success": bool(timetable_entries),
            "subject_info": {
                "subject": subject,
                "timetable_entries": serialize_documents(timetable_entries),
            },
            "message": "No subject document found; returned timetable matches instead.",
        }
    except Exception as exc:
        logger.exception("get_subject_info failed")
        return {"success": False, "error": str(exc)}
