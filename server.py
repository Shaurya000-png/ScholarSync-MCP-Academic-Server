"""ScholarSync MCP Academic Server."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from database.mongo import get_collection, ping_database, serialize_documents
from tools.assignments import add_assignment, search_assignments, update_assignment_status
from tools.exam import generate_exam_questions
from tools.knowledge_gaps import find_knowledge_gaps
from tools.multilingual import translate_and_search
from tools.notes import retrieve_academic_context, search_notes
from tools.pdf_tools import summarize_pdf
from tools.rag import ingest_academic_knowledge_base, semantic_search_academic_knowledge
from tools.study_plan import create_study_plan
from tools.tasks import add_task, mark_task_complete, view_tasks
from tools.timetable import get_schedule, get_subject_info
from tools.urgent import get_urgent_items

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "server.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("scholarsync_mcp_server")

mcp = FastMCP(
    name="ScholarSync MCP Academic Server",
    instructions=(
        "Use ScholarSync tools to help students manage academic notes, PDFs, assignments, "
        "tasks, timetable data, and course information stored in MongoDB."
    ),
)

ALLOWED_COLLECTIONS = {"notes", "assignments", "tasks", "timetable", "subjects", "academic_chunks"}


def query_student_database(
    collection: str,
    filters: dict[str, Any] | None = None,
    limit: int = 10,
    user_id: str = "",
) -> dict[str, Any]:
    """
    Query student academic data from MongoDB using safe exact-match filters.

    Use this for direct database lookups when a more specific academic tool is
    not enough. Allowed collections are notes, assignments, tasks, timetable,
    and subjects.
    """
    from services.rag_service import normalize_user_id

    resolved_user_id = normalize_user_id(user_id)
    logger.info(
        "query_student_database input user_id=%s collection=%r filters=%r limit=%s",
        resolved_user_id,
        collection,
        filters,
        limit,
    )
    if collection not in ALLOWED_COLLECTIONS:
        return {"success": False, "error": f"collection must be one of {sorted(ALLOWED_COLLECTIONS)}"}

    safe_filters = filters or {}
    for key in safe_filters:
        if key.startswith("$"):
            return {"success": False, "error": "MongoDB operator filters are not allowed"}
    safe_filters = {"user_id": resolved_user_id, **safe_filters}

    safe_limit = max(1, min(int(limit), 50))

    try:
        documents = list(get_collection(collection).find(safe_filters).limit(safe_limit))
        rows = serialize_documents(documents)
        logger.info("query_student_database output collection=%s count=%s", collection, len(rows))
        return {"success": True, "user_id": resolved_user_id, "collection": collection, "count": len(rows), "results": rows}
    except Exception as exc:
        logger.exception("query_student_database failed")
        return {"success": False, "error": str(exc), "results": []}


def health_check() -> dict[str, Any]:
    """
    Check whether the MCP server can reach MongoDB.

    Use this for debugging setup, environment variables, and database
    connectivity before running academic workflows.
    """
    logger.info("health_check input")
    try:
        result = ping_database()
        logger.info("health_check output %s", result)
        return {"success": True, **result}
    except Exception as exc:
        logger.exception("health_check failed")
        return {"success": False, "error": str(exc)}


def register_tools() -> None:
    """Register all academic tools with the MCP server."""
    mcp.tool(
        name="search_notes",
        description="Search academic notes by topic, keyword, subject, or concept from MongoDB and local text notes.",
    )(search_notes)
    mcp.tool(
        name="retrieve_academic_context",
        description="Retrieve notes and study context before answering academic questions using semantic RAG with keyword fallback.",
    )(retrieve_academic_context)
    mcp.tool(
        name="ingest_academic_knowledge_base",
        description="Chunk notes and PDFs, embed them, and store semantic retrieval documents in MongoDB.",
    )(ingest_academic_knowledge_base)
    mcp.tool(
        name="semantic_search_academic_knowledge",
        description="Run semantic vector-style search across ingested academic notes and PDF chunks.",
    )(semantic_search_academic_knowledge)
    mcp.tool(
        name="summarize_pdf",
        description="Extract study PDF text from the pdfs folder and return content Claude can summarize into key points.",
    )(summarize_pdf)
    mcp.tool(
        name="search_assignments",
        description="Search assignments by subject, due date, priority, status, or text query.",
    )(search_assignments)
    mcp.tool(
        name="add_assignment",
        description="Add a new academic assignment with subject, title, due date, priority, and completion status.",
    )(add_assignment)
    mcp.tool(
        name="update_assignment_status",
        description="Update an assignment status, such as pending, in progress, completed, or overdue.",
    )(update_assignment_status)
    mcp.tool(
        name="get_schedule",
        description="Get classes from the timetable for today, tomorrow, a weekday, or a specific subject.",
    )(get_schedule)
    mcp.tool(
        name="get_subject_info",
        description="Get subject/course information such as faculty, credits, syllabus, or timetable details.",
    )(get_subject_info)
    mcp.tool(
        name="add_task",
        description="Add an academic task or reminder with optional due date, subject, priority, and notes.",
    )(add_task)
    mcp.tool(
        name="view_tasks",
        description="View academic tasks and reminders by status, date, subject, or text query.",
    )(view_tasks)
    mcp.tool(
        name="mark_task_complete",
        description="Mark an academic task or reminder as completed.",
    )(mark_task_complete)
    mcp.tool(
        name="query_student_database",
        description="Safely query MongoDB academic collections when a direct database lookup is needed.",
    )(query_student_database)
    mcp.tool(
        name="health_check",
        description="Check MCP server and MongoDB connectivity for debugging.",
    )(health_check)
    mcp.tool(
        name="generate_exam_questions",
        description="Retrieve academic context for a subject/topic and instruct Claude to generate practice exam questions (MCQ, short answer, or true/false).",
    )(generate_exam_questions)
    mcp.tool(
        name="create_study_plan",
        description="Gather upcoming assignments, tasks, and timetable data so Claude can create a personalised study plan for the next N days.",
    )(create_study_plan)
    mcp.tool(
        name="find_knowledge_gaps",
        description="Identify subjects where the student has assignments but few or no ingested notes, flagging knowledge gaps that need more study material.",
    )(find_knowledge_gaps)
    mcp.tool(
        name="get_urgent_items",
        description="Find assignments and tasks due within the next N hours so Claude can proactively warn the student about upcoming deadlines.",
    )(get_urgent_items)
    mcp.tool(
        name="translate_and_search",
        description="Detect the language of a query, translate it to English, and search academic content. Use this when the student asks in Hindi, Marathi, or any non-English language.",
    )(translate_and_search)


register_tools()


if __name__ == "__main__":
    logger.info("Starting ScholarSync MCP Academic Server")
    mcp.run(transport="stdio")
