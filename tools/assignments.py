"""Assignment tracking tools backed by MongoDB."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from database.mongo import get_collection, regex_filter, resolve_date, serialize_document, serialize_documents
from schemas.academic import AssignmentCreate
from services.rag_service import normalize_user_id

logger = logging.getLogger(__name__)

VALID_STATUSES = {"pending", "in_progress", "completed", "overdue"}
VALID_PRIORITIES = {"low", "medium", "high", "urgent"}


def add_assignment(
    subject: str,
    title: str,
    due_date: str,
    priority: str = "medium",
    status: str = "pending",
    description: str = "",
    user_id: str = "",
) -> dict[str, Any]:
    """
    Add a new academic assignment to MongoDB.

    Use this when the user asks Claude to create, save, or remember an
    assignment with a due date, priority, or completion status.
    """
    resolved_user_id = normalize_user_id(user_id)
    logger.info("add_assignment input user_id=%s subject=%r title=%r due_date=%r", resolved_user_id, subject, title, due_date)
    try:
        payload = AssignmentCreate(
            user_id=resolved_user_id,
            subject=subject,
            title=title,
            due_date=due_date,
            priority=priority.strip().lower(),
            status=status.strip().lower(),
            description=description,
        )
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    document = {
        "user_id": payload.user_id,
        "subject": payload.subject,
        "title": payload.title,
        "due_date": resolve_date(payload.due_date),
        "priority": payload.priority,
        "status": payload.status,
        "description": payload.description,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    try:
        result = get_collection("assignments").insert_one(document)
        document["_id"] = result.inserted_id
        logger.info("add_assignment output id=%s", result.inserted_id)
        return {"success": True, "assignment": serialize_document(document)}
    except Exception as exc:
        logger.exception("add_assignment failed")
        return {"success": False, "error": str(exc)}


def search_assignments(
    query: str = "",
    subject: str = "",
    status: str = "",
    due_date: str = "",
    priority: str = "",
    limit: int = 10,
    user_id: str = "",
) -> dict[str, Any]:
    """
    Search assignments by title, subject, due date, priority, or status.

    Use this for questions like pending assignments, assignments due tomorrow,
    high-priority work, or assignment tracking.
    """
    logger.info(
        "search_assignments input query=%r subject=%r status=%r due_date=%r priority=%r",
        query,
        subject,
        status,
        due_date,
        priority,
    )
    resolved_user_id = normalize_user_id(user_id)
    safe_limit = max(1, min(int(limit), 50))

    filters: list[dict[str, Any]] = [{"user_id": resolved_user_id}]
    text_filter = regex_filter(["subject", "title", "description"], query)
    if text_filter:
        filters.append(text_filter)
    if subject:
        filters.append({"subject": {"$regex": subject, "$options": "i"}})
    if status:
        filters.append({"status": status.strip().lower()})
    if due_date:
        filters.append({"due_date": resolve_date(due_date)})
    if priority:
        filters.append({"priority": priority.strip().lower()})

    mongo_query = {"$and": filters} if filters else {}

    try:
        documents = list(get_collection("assignments").find(mongo_query).sort("due_date", 1).limit(safe_limit))
        assignments = serialize_documents(documents)
        logger.info("search_assignments output count=%s", len(assignments))
        return {"success": True, "count": len(assignments), "assignments": assignments}
    except Exception as exc:
        logger.exception("search_assignments failed")
        return {"success": False, "error": str(exc), "assignments": []}


def update_assignment_status(title: str, status: str, subject: str = "", user_id: str = "") -> dict[str, Any]:
    """
    Update an assignment completion status.

    Use this when the user asks to mark an assignment completed, pending,
    in progress, or overdue.
    """
    resolved_user_id = normalize_user_id(user_id)
    logger.info("update_assignment_status input user_id=%s title=%r status=%r subject=%r", resolved_user_id, title, status, subject)
    normalized_status = status.strip().lower()
    if normalized_status not in VALID_STATUSES:
        return {"success": False, "error": f"status must be one of {sorted(VALID_STATUSES)}"}

    filters: list[dict[str, Any]] = [{"user_id": resolved_user_id}, {"title": {"$regex": title, "$options": "i"}}]
    if subject:
        filters.append({"subject": {"$regex": subject, "$options": "i"}})

    update = {"status": normalized_status, "updated_at": datetime.utcnow()}
    if normalized_status == "completed":
        update["completed_at"] = datetime.utcnow()

    try:
        collection = get_collection("assignments")
        document = collection.find_one_and_update(
            {"$and": filters},
            {"$set": update},
            return_document=True,
        )
        if not document:
            return {"success": False, "error": "No matching assignment found"}
        logger.info("update_assignment_status output id=%s", document.get("_id"))
        return {"success": True, "assignment": serialize_document(document)}
    except Exception as exc:
        logger.exception("update_assignment_status failed")
        return {"success": False, "error": str(exc)}
