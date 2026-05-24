"""Semantic ingestion and retrieval service."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import get_settings
from database.mongo import get_collection, serialize_documents
from services.chunking import chunk_text
from services.embeddings import cosine_similarity, embed_text
from services.pdf_extraction import extract_pdf_pages

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTES_DIR = PROJECT_ROOT / "notes"
PDF_DIR = PROJECT_ROOT / "pdfs"


def normalize_user_id(user_id: str = "") -> str:
    """Return the explicit user ID or the configured default."""
    return user_id.strip() or get_settings().default_user_id


def user_filter(user_id: str = "") -> dict[str, str]:
    """Build a MongoDB filter for the current user."""
    return {"user_id": normalize_user_id(user_id)}


def ingest_academic_materials(
    user_id: str = "",
    include_mongodb_notes: bool = True,
    include_local_notes: bool = True,
    include_pdfs: bool = True,
    subject: str = "",
    reset_existing: bool = False,
    max_pdf_pages: int = 50,
) -> dict[str, Any]:
    """
    Ingest notes and PDFs into semantic chunks with embeddings.

    This creates documents in the academic_chunks collection. It is designed to
    work locally now and can be migrated to MongoDB Atlas Vector Search by
    indexing the embedding field later.
    """
    settings = get_settings()
    resolved_user_id = normalize_user_id(user_id)
    chunk_collection = get_collection(settings.chunk_collection)
    created_count = 0
    sources: list[dict[str, Any]] = []

    logger.info("ingest_academic_materials input user_id=%s subject=%r reset=%s", resolved_user_id, subject, reset_existing)

    if reset_existing:
        delete_filter: dict[str, Any] = {"user_id": resolved_user_id}
        if subject:
            delete_filter["subject"] = {"$regex": subject, "$options": "i"}
        deleted = chunk_collection.delete_many(delete_filter).deleted_count
    else:
        deleted = 0

    def store_chunks(source: dict[str, Any], text: str, page: int | None = None) -> int:
        nonlocal created_count
        chunks = chunk_text(text, settings.chunk_size_words, settings.chunk_overlap_words)
        if not chunks:
            return 0

        documents = []
        for index, chunk in enumerate(chunks):
            embedding, model_name = embed_text(chunk)
            documents.append(
                {
                    "user_id": resolved_user_id,
                    "source_type": source["source_type"],
                    "source_id": source["source_id"],
                    "subject": source.get("subject", ""),
                    "title": source.get("title", ""),
                    "page": page,
                    "chunk_index": index,
                    "text": chunk,
                    "embedding": embedding,
                    "embedding_model": model_name,
                    "created_at": datetime.utcnow(),
                }
            )

        chunk_collection.insert_many(documents)
        created_count += len(documents)
        return len(documents)

    try:
        if include_mongodb_notes:
            note_filter: dict[str, Any] = {"user_id": resolved_user_id}
            if subject:
                note_filter["subject"] = {"$regex": subject, "$options": "i"}
            for note in get_collection("notes").find(note_filter):
                text = str(note.get("content", "")).strip()
                source = {
                    "source_type": "note",
                    "source_id": str(note.get("_id")),
                    "subject": note.get("subject", ""),
                    "title": note.get("title") or note.get("topic") or "",
                }
                count = store_chunks(source, text)
                if count:
                    sources.append({**source, "chunks": count})

        if include_local_notes and NOTES_DIR.exists():
            for path in sorted(NOTES_DIR.glob("*.txt")):
                if subject and subject.lower() not in path.stem.lower():
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                source = {
                    "source_type": "local_note",
                    "source_id": path.name,
                    "subject": path.stem,
                    "title": path.stem,
                }
                count = store_chunks(source, text)
                if count:
                    sources.append({**source, "chunks": count})

        if include_pdfs and PDF_DIR.exists():
            for path in sorted(PDF_DIR.glob("*.pdf")):
                if subject and subject.lower() not in path.stem.lower():
                    continue
                result = extract_pdf_pages(path, max_pages=max_pdf_pages)
                # Support both old 3-tuple and new 4-tuple return format
                if len(result) == 4:
                    _total_pages, pages, extractor, detected_subject = result
                else:
                    _total_pages, pages, extractor = result
                    detected_subject = ""
                # Use detected subject if no explicit subject was provided
                effective_subject = subject or detected_subject or path.stem
                pdf_chunks = 0
                for page in pages:
                    source = {
                        "source_type": "pdf",
                        "source_id": path.name,
                        "subject": effective_subject,
                        "title": path.stem,
                    }
                    pdf_chunks += store_chunks(source, str(page["text"]), page=int(page["page"]))
                if pdf_chunks:
                    sources.append(
                        {
                            "source_type": "pdf",
                            "source_id": path.name,
                            "subject": effective_subject,
                            "title": path.stem,
                            "chunks": pdf_chunks,
                            "extractor": extractor,
                            "detected_subject": detected_subject,
                        }
                    )

        logger.info("ingest_academic_materials output chunks=%s sources=%s", created_count, len(sources))
        return {
            "success": True,
            "user_id": resolved_user_id,
            "deleted_existing_chunks": deleted,
            "chunks_created": created_count,
            "sources_ingested": sources,
            "collection": settings.chunk_collection,
        }
    except Exception as exc:
        logger.exception("ingest_academic_materials failed")
        return {"success": False, "error": str(exc), "chunks_created": created_count, "sources_ingested": sources}


def semantic_search_academic_context(
    query: str,
    user_id: str = "",
    subject: str = "",
    top_k: int = 5,
    min_score: float = 0.0,
) -> dict[str, Any]:
    """Search semantic academic chunks using local cosine similarity."""
    settings = get_settings()
    resolved_user_id = normalize_user_id(user_id)
    safe_top_k = max(1, min(int(top_k), 20))
    logger.info("semantic_search_academic_context input user_id=%s query=%r", resolved_user_id, query)

    try:
        query_embedding, model_name = embed_text(query)
        filters: dict[str, Any] = {"user_id": resolved_user_id}
        if subject:
            filters["subject"] = {"$regex": subject, "$options": "i"}

        if settings.vector_search_mode.lower() == "atlas":
            try:
                vector_filter: dict[str, Any] = {"user_id": resolved_user_id}
                if subject:
                    vector_filter["subject"] = {"$regex": subject, "$options": "i"}

                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": settings.atlas_vector_index_name,
                            "path": "embedding",
                            "queryVector": query_embedding,
                            "numCandidates": max(100, safe_top_k * 20),
                            "limit": safe_top_k,
                            "filter": vector_filter,
                        }
                    },
                    {
                        "$project": {
                            "embedding": 0,
                            "score": {"$meta": "vectorSearchScore"},
                            "source_type": 1,
                            "source_id": 1,
                            "subject": 1,
                            "title": 1,
                            "page": 1,
                            "chunk_index": 1,
                            "text": 1,
                            "embedding_model": 1,
                        }
                    },
                ]
                atlas_matches = serialize_documents(list(get_collection(settings.chunk_collection).aggregate(pipeline)))
                return {
                    "success": True,
                    "user_id": resolved_user_id,
                    "query": query,
                    "subject": subject,
                    "search_mode": "atlas_vector_search",
                    "embedding_model": model_name,
                    "candidate_count": None,
                    "match_count": len(atlas_matches),
                    "matches": atlas_matches,
                }
            except Exception as exc:
                logger.warning("Atlas vector search failed; falling back to local cosine: %s", exc)

        candidates = list(
            get_collection(settings.chunk_collection).find(
                filters,
                {
                    "embedding": 1,
                    "source_type": 1,
                    "source_id": 1,
                    "subject": 1,
                    "title": 1,
                    "page": 1,
                    "chunk_index": 1,
                    "text": 1,
                    "embedding_model": 1,
                },
            )
        )

        scored = []
        for document in candidates:
            score = cosine_similarity(query_embedding, document.get("embedding", []))
            if score >= min_score:
                document["score"] = round(score, 4)
                document.pop("embedding", None)
                scored.append(document)

        scored.sort(key=lambda item: item["score"], reverse=True)
        matches = serialize_documents(scored[:safe_top_k])
        logger.info("semantic_search_academic_context output matches=%s candidates=%s", len(matches), len(candidates))
        return {
            "success": True,
            "user_id": resolved_user_id,
            "query": query,
            "subject": subject,
            "search_mode": "local_cosine",
            "embedding_model": model_name,
            "candidate_count": len(candidates),
            "match_count": len(matches),
            "matches": matches,
        }
    except Exception as exc:
        logger.exception("semantic_search_academic_context failed")
        return {"success": False, "error": str(exc), "matches": []}
