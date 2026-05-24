"""Tests for TaskCreate Pydantic model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.academic import TaskCreate


class TestTaskCreateValidPriority:
    """TaskCreate must accept all valid priority values (low/medium/high/urgent)."""

    @pytest.mark.parametrize("priority", ["low", "medium", "high", "urgent"])
    def test_accepts_valid_priority(self, priority: str) -> None:
        task = TaskCreate(
            user_id="test_user",
            title="Revise chapter 5",
            priority=priority,
        )
        assert task.priority == priority
        assert task.title == "Revise chapter 5"


class TestTaskCreateRejectsInvalidPriority:
    """TaskCreate must reject priority values outside the allowed set."""

    @pytest.mark.parametrize("bad_priority", ["critical", "none", "super", ""])
    def test_rejects_invalid_priority(self, bad_priority: str) -> None:
        with pytest.raises(ValidationError):
            TaskCreate(
                user_id="test_user",
                title="Revise chapter 5",
                priority=bad_priority,
            )


class TestTaskCreateOptionalFields:
    """TaskCreate with missing optional fields still passes validation."""

    def test_minimal_creation(self) -> None:
        task = TaskCreate(user_id="test_user", title="Quick reminder")
        assert task.title == "Quick reminder"
        assert task.due_date == ""
        assert task.subject == ""
        assert task.notes == ""
        assert task.priority == "medium"

    def test_partial_optional_fields(self) -> None:
        task = TaskCreate(
            user_id="test_user",
            title="Submit lab work",
            subject="Physics",
        )
        assert task.subject == "Physics"
        assert task.due_date == ""
        assert task.notes == ""
