"""Study plan generation tool backed by MongoDB data retrieval.

This module queries assignments, tasks, and timetable data for upcoming
deadlines, then returns structured JSON for Claude to reason over and
produce a personalised study plan.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from database.mongo import get_collection, serialize_documents
from services.rag_service import normalize_user_id

logger = logging.getLogger(__name__)


def create_study_plan(
    days_ahead: int = 7,
    user_id: str = "",
) -> dict[str, Any]:
    """
    Gather upcoming assignments, tasks, and timetable data for study planning.

    Use this when the user asks Claude to create a study plan, revision
    schedule, weekly planner, or wants to know what to focus on in the
    coming days. The tool retrieves real data; Claude generates the plan.
    """
    resolved_user_id = normalize_user_id(user_id)
    safe_days = max(1, min(int(days_ahead), 30))
    logger.info(
        "create_study_plan input user_id=%s days_ahead=%s",
        resolved_user_id, safe_days,
    )

    now = datetime.utcnow()
    cutoff = now + timedelta(days=safe_days)
    cutoff_iso = cutoff.date().isoformat()

    try:
        # ── Upcoming assignments (pending / in_progress) ─────────────
        assignment_filter: dict[str, Any] = {
            "user_id": resolved_user_id,
            "status": {"$in": ["pending", "in_progress"]},
            "due_date": {"$lte": cutoff_iso},
        }
        assignments = list(
            get_collection("assignments")
            .find(assignment_filter)
            .sort("due_date", 1)
        )
        assignments_serialized = serialize_documents(assignments)

        # ── Pending tasks ────────────────────────────────────────────
        task_filter: dict[str, Any] = {
            "user_id": resolved_user_id,
            "status": "pending",
            "due_date": {"$lte": cutoff_iso, "$ne": ""},
        }
        tasks = list(
            get_collection("tasks")
            .find(task_filter)
            .sort("due_date", 1)
        )
        tasks_serialized = serialize_documents(tasks)

        # ── Timetable for upcoming days ──────────────────────────────
        day_names = []
        for offset in range(safe_days):
            day_names.append((now + timedelta(days=offset)).strftime("%A"))
        unique_days = list(dict.fromkeys(day_names))  # preserve order, deduplicate

        timetable_filter: dict[str, Any] = {
            "user_id": resolved_user_id,
            "day": {"$in": unique_days},
        }
        timetable = list(
            get_collection("timetable")
            .find(timetable_filter)
            .sort("start_time", 1)
        )
        timetable_serialized = serialize_documents(timetable)

        result: dict[str, Any] = {
            "success": True,
            "message": (
                f"Retrieved academic data for the next {safe_days} day(s). "
                "Use this information to create a detailed, day-by-day study plan."
            ),
            "data": {
                "user_id": resolved_user_id,
                "days_ahead": safe_days,
                "cutoff_date": cutoff_iso,
                "upcoming_assignments": assignments_serialized,
                "assignment_count": len(assignments_serialized),
                "pending_tasks": tasks_serialized,
                "task_count": len(tasks_serialized),
                "timetable_entries": timetable_serialized,
                "timetable_days": unique_days,
                "planning_instruction": (
                    "Create a study plan considering: (1) assignment due dates and priorities, "
                    "(2) pending tasks and their urgency, (3) class schedule from the timetable. "
                    "Suggest specific study blocks, revision sessions, and break times."
                ),
            },
        }
        logger.info(
            "create_study_plan output assignments=%s tasks=%s timetable=%s",
            len(assignments_serialized), len(tasks_serialized), len(timetable_serialized),
        )
        return result

    except Exception as exc:
        logger.exception("create_study_plan failed")
        return {"success": False, "message": str(exc)}
