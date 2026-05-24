"""MongoDB connection and serialization helpers."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from config.settings import get_settings

try:
    from bson import ObjectId
except ImportError:  # pragma: no cover - bson is provided by pymongo
    ObjectId = None


load_dotenv()

_client: MongoClient | None = None


def get_client() -> MongoClient:
    """Return a shared MongoDB client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
    return _client


def get_database() -> Database:
    """Return the configured MongoDB database."""
    return get_client()[get_settings().mongodb_database]


def get_collection(name: str) -> Collection:
    """Return a MongoDB collection from the configured database."""
    return get_database()[name]


def ping_database() -> dict[str, Any]:
    """Check MongoDB connectivity."""
    result = get_client().admin.command("ping")
    return {"ok": bool(result.get("ok")), "database": get_settings().mongodb_database}


def serialize_document(value: Any) -> Any:
    """Convert MongoDB/BSON values into JSON-friendly Python values."""
    if isinstance(value, list):
        return [serialize_document(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_document(item) for key, item in value.items()}
    if ObjectId is not None and isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def serialize_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize a list of MongoDB documents."""
    return [serialize_document(document) for document in documents]


def resolve_date(value: str | None) -> str | None:
    """Resolve common user date words into ISO date strings."""
    if not value:
        return None

    normalized = value.strip().lower()
    today = datetime.now().date()

    if normalized == "today":
        return today.isoformat()
    if normalized == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    if normalized == "yesterday":
        return (today - timedelta(days=1)).isoformat()

    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date().isoformat()
    except ValueError:
        return value.strip()


def regex_filter(fields: list[str], query: str) -> dict[str, Any]:
    """Build a case-insensitive MongoDB regex filter across multiple fields."""
    if not query:
        return {}
    return {"$or": [{field: {"$regex": query, "$options": "i"}} for field in fields]}
