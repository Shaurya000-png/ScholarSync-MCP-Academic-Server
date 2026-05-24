"""Tests for the health_check tool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_health_check_returns_dict_with_success_key() -> None:
    """health_check must return a dict containing a 'success' key."""
    with patch("database.mongo.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.admin.command.return_value = {"ok": 1}
        mock_get_client.return_value = mock_client

        # Import after patching so the module picks up the mock
        from server import health_check

        result = health_check()
        assert isinstance(result, dict)
        assert "success" in result


def test_health_check_returns_false_on_bad_connection() -> None:
    """health_check must return success: False when MongoDB is unreachable."""
    with patch("database.mongo.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.admin.command.side_effect = ConnectionError("unreachable")
        mock_get_client.return_value = mock_client

        from server import health_check

        result = health_check()
        assert isinstance(result, dict)
        assert result["success"] is False
