"""Core Pydantic models for academic data."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Priority = Literal["low", "medium", "high", "urgent"]
AssignmentStatus = Literal["pending", "in_progress", "completed", "overdue"]
TaskStatus = Literal["pending", "completed"]


class ToolResult(BaseModel):
    """Common shape for tool responses."""

    success: bool
    error: str | None = None


class AssignmentCreate(BaseModel):
    """Validated assignment input."""

    model_config = ConfigDict(str_strip_whitespace=True)

    user_id: str
    subject: str = Field(min_length=1)
    title: str = Field(min_length=1)
    due_date: str = Field(min_length=1)
    priority: Priority = "medium"
    status: AssignmentStatus = "pending"
    description: str = ""


class TaskCreate(BaseModel):
    """Validated task input."""

    model_config = ConfigDict(str_strip_whitespace=True)

    user_id: str
    title: str = Field(min_length=1)
    due_date: str = ""
    priority: Priority = "medium"
    subject: str = ""
    notes: str = ""


class AcademicChunk(BaseModel):
    """Text chunk stored for semantic retrieval."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str
    source_type: Literal["note", "pdf", "local_note"]
    source_id: str
    subject: str = ""
    title: str = ""
    page: int | None = None
    chunk_index: int
    text: str
    embedding: list[float]
    embedding_model: str
    created_at: datetime
