"""Urgent items retrieval tool backed by MongoDB.

This module checks for assignments and tasks due within a configurable
time window, returning them sorted by urgency so Claude can proactively
warn the student at the start of a conversation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from database.mongo import get_collection, serialize_documents
from services.rag_service import normalize_user_id

logger = logging.getLogger(__name__)


def get_urgent_items(
    hours_ahead: int = 24,
    user_id: str = "",
) -> dict[str, Any]:
    """
    Find assignments and tasks due within the next N hours.

    Use this at the start of a conversation or when the user asks about
    urgent deadlines, upcoming due items, or what needs immediate attention.
    Returns both assignments and tasks sorted by due date so Claude can
    proactively warn the student.
    """
    resolved_user_id = normalize_user_id(user_id)
    safe_hours = max(1, min(int(hours_ahead), 168))  # cap at 1 week
    logger.info(
        "get_urgent_items input user_id=%s hours_ahead=%s",
        resolved_user_id, safe_hours,
    )

    now = datetime.utcnow()
    cutoff = now + timedelta(hours=safe_hours)
    cutoff_iso = cutoff.isoformat()
    now_iso = now.isoformat()

    try:
        # ── Urgent assignments (not completed, due within window) ────
        assignment_filter: dict[str, Any] = {
            "user_id": resolved_user_id,
            "status": {"$nin": ["completed"]},
            "due_date": {"$lte": cutoff_iso},
        }
        urgent_assignments = list(
            get_collection("assignments")
            .find(assignment_filter)
            .sort("due_date", 1)
        )
        assignments_serialized = serialize_documents(urgent_assignments)

        # ── Urgent tasks (pending, due within window) ────────────────
        task_filter: dict[str, Any] = {
            "user_id": resolved_user_id,
            "status": "pending",
            "due_date": {"$lte": cutoff_iso, "$ne": ""},
        }
        urgent_tasks = list(
            get_collection("tasks")
            .find(task_filter)
            .sort("due_date", 1)
        )
        tasks_serialized = serialize_documents(urgent_tasks)

        total_urgent = len(assignments_serialized) + len(tasks_serialized)

        result: dict[str, Any] = {
            "success": True,
            "message": (
                f"Found {total_urgent} urgent item(s) due within the next {safe_hours} hour(s)."
                if total_urgent
                else f"No urgent items due within the next {safe_hours} hour(s). You're all clear!"
            ),
            "data": {
                "user_id": resolved_user_id,
                "hours_ahead": safe_hours,
                "checked_at": now_iso,
                "cutoff": cutoff_iso,
                "total_urgent": total_urgent,
                "urgent_assignments": assignments_serialized,
                "urgent_assignment_count": len(assignments_serialized),
                "urgent_tasks": tasks_serialized,
                "urgent_task_count": len(tasks_serialized),
            },
        }
        logger.info(
            "get_urgent_items output total=%s assignments=%s tasks=%s",
            total_urgent, len(assignments_serialized), len(tasks_serialized),
        )
        return result

    except Exception as exc:
        logger.exception("get_urgent_items failed")
        return {"success": False, "message": str(exc)}
