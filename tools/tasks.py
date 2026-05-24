"""Academic task and reminder tools backed by MongoDB."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from database.mongo import get_collection, regex_filter, resolve_date, serialize_document, serialize_documents
from schemas.academic import TaskCreate
from services.rag_service import normalize_user_id

logger = logging.getLogger(__name__)

VALID_TASK_STATUSES = {"pending", "completed"}
VALID_PRIORITIES = {"low", "medium", "high", "urgent"}


def add_task(
    title: str,
    due_date: str = "",
    priority: str = "medium",
    subject: str = "",
    notes: str = "",
    user_id: str = "",
) -> dict[str, Any]:
    """
    Add an academic task or reminder.

    Use this when the user asks Claude to add a reminder, revision task,
    study plan item, deadline reminder, or academic to-do.
    """
    resolved_user_id = normalize_user_id(user_id)
    logger.info("add_task input user_id=%s title=%r due_date=%r priority=%r", resolved_user_id, title, due_date, priority)
    try:
        payload = TaskCreate(
            user_id=resolved_user_id,
            title=title,
            due_date=due_date,
            priority=priority.strip().lower(),
            subject=subject,
            notes=notes,
        )
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    document = {
        "user_id": payload.user_id,
        "title": payload.title,
        "subject": payload.subject,
        "due_date": resolve_date(payload.due_date) if payload.due_date else "",
        "priority": payload.priority,
        "status": "pending",
        "notes": payload.notes,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    try:
        result = get_collection("tasks").insert_one(document)
        document["_id"] = result.inserted_id
        logger.info("add_task output id=%s", result.inserted_id)
        return {"success": True, "task": serialize_document(document)}
    except Exception as exc:
        logger.exception("add_task failed")
        return {"success": False, "error": str(exc)}


def view_tasks(
    status: str = "pending",
    subject: str = "",
    due_date: str = "",
    query: str = "",
    limit: int = 20,
    user_id: str = "",
) -> dict[str, Any]:
    """
    View academic tasks and reminders.

    Use this when the user asks to see pending tasks, completed tasks,
    reminders for a date, or subject-specific academic to-dos.
    """
    logger.info("view_tasks input status=%r subject=%r due_date=%r query=%r", status, subject, due_date, query)
    resolved_user_id = normalize_user_id(user_id)
    safe_limit = max(1, min(int(limit), 50))

    filters: list[dict[str, Any]] = [{"user_id": resolved_user_id}]
    if status:
        normalized_status = status.strip().lower()
        if normalized_status not in VALID_TASK_STATUSES:
            return {"success": False, "error": f"status must be one of {sorted(VALID_TASK_STATUSES)}"}
        filters.append({"status": normalized_status})
    if subject:
        filters.append({"subject": {"$regex": subject, "$options": "i"}})
    if due_date:
        filters.append({"due_date": resolve_date(due_date)})

    text_filter = regex_filter(["title", "subject", "notes"], query)
    if text_filter:
        filters.append(text_filter)

    mongo_query = {"$and": filters} if filters else {}

    try:
        documents = list(get_collection("tasks").find(mongo_query).sort("due_date", 1).limit(safe_limit))
        tasks = serialize_documents(documents)
        logger.info("view_tasks output count=%s", len(tasks))
        return {"success": True, "count": len(tasks), "tasks": tasks}
    except Exception as exc:
        logger.exception("view_tasks failed")
        return {"success": False, "error": str(exc), "tasks": []}


def mark_task_complete(title: str, subject: str = "", user_id: str = "") -> dict[str, Any]:
    """
    Mark a task or reminder complete.

    Use this when the user says a reminder, revision item, or academic task is done.
    """
    resolved_user_id = normalize_user_id(user_id)
    logger.info("mark_task_complete input user_id=%s title=%r subject=%r", resolved_user_id, title, subject)
    filters: list[dict[str, Any]] = [{"user_id": resolved_user_id}, {"title": {"$regex": title, "$options": "i"}}]
    if subject:
        filters.append({"subject": {"$regex": subject, "$options": "i"}})

    try:
        document = get_collection("tasks").find_one_and_update(
            {"$and": filters},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                }
            },
            return_document=True,
        )
        if not document:
            return {"success": False, "error": "No matching task found"}
        logger.info("mark_task_complete output id=%s", document.get("_id"))
        return {"success": True, "task": serialize_document(document)}
    except Exception as exc:
        logger.exception("mark_task_complete failed")
        return {"success": False, "error": str(exc)}
