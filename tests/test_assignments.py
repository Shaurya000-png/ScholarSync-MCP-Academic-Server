"""Tests for AssignmentCreate Pydantic model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.academic import AssignmentCreate


class TestAssignmentCreateValidPriority:
    """AssignmentCreate must accept all four valid priority values."""

    @pytest.mark.parametrize("priority", ["low", "medium", "high", "urgent"])
    def test_accepts_valid_priority(self, priority: str) -> None:
        assignment = AssignmentCreate(
            user_id="test_user",
            subject="AI",
            title="Lab report",
            due_date="2026-06-01",
            priority=priority,
        )
        assert assignment.priority == priority


class TestAssignmentCreateInvalidPriority:
    """AssignmentCreate must reject priority values outside the allowed set."""

    @pytest.mark.parametrize("bad_priority", ["critical", "normal", "URGENT", "1", ""])
    def test_rejects_invalid_priority(self, bad_priority: str) -> None:
        with pytest.raises(ValidationError):
            AssignmentCreate(
                user_id="test_user",
                subject="AI",
                title="Lab report",
                due_date="2026-06-01",
                priority=bad_priority,
            )


class TestAssignmentCreateInvalidStatus:
    """AssignmentCreate must reject status values outside the allowed set."""

    @pytest.mark.parametrize("bad_status", ["done", "cancelled", "active", ""])
    def test_rejects_invalid_status(self, bad_status: str) -> None:
        with pytest.raises(ValidationError):
            AssignmentCreate(
                user_id="test_user",
                subject="AI",
                title="Lab report",
                due_date="2026-06-01",
                status=bad_status,
            )


class TestAssignmentCreateDueDate:
    """AssignmentCreate accepts ISO-format date strings."""

    def test_accepts_iso_date_string(self) -> None:
        assignment = AssignmentCreate(
            user_id="test_user",
            subject="DBMS",
            title="ER diagram",
            due_date="2026-12-31",
        )
        assert assignment.due_date == "2026-12-31"

    def test_accepts_iso_datetime_string(self) -> None:
        assignment = AssignmentCreate(
            user_id="test_user",
            subject="DBMS",
            title="ER diagram",
            due_date="2026-12-31T23:59:00",
        )
        assert assignment.due_date == "2026-12-31T23:59:00"
