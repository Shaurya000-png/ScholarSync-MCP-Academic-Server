"""Hybrid search combining BM25 keyword scoring with vector similarity.

Results from both methods are fused using Reciprocal Rank Fusion (RRF)
to surface chunks that are relevant by both lexical and semantic criteria.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import get_settings
from database.mongo import get_collection, serialize_documents
from services.embeddings import cosine_similarity, embed_text

logger = logging.getLogger(__name__)

# RRF constant — standard value used in the literature.
RRF_K = 60


def _vector_search(
    query_embedding: list[float],
    filters: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """Run local cosine-similarity search and return scored documents."""
    settings = get_settings()
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
            },
        )
    )

    scored: list[dict[str, Any]] = []
    for doc in candidates:
        score = cosine_similarity(query_embedding, doc.get("embedding", []))
        doc["vector_score"] = round(score, 6)
        doc.pop("embedding", None)
        scored.append(doc)

    scored.sort(key=lambda d: d["vector_score"], reverse=True)
    return scored[:top_k]


def _bm25_search(
    query: str,
    filters: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """Run BM25 keyword search over chunk texts stored in MongoDB."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed; skipping BM25 search")
        return []

    settings = get_settings()
    candidates = list(
        get_collection(settings.chunk_collection).find(
            filters,
            {
                "source_type": 1,
                "source_id": 1,
                "subject": 1,
                "title": 1,
                "page": 1,
                "chunk_index": 1,
                "text": 1,
            },
        )
    )

    if not candidates:
        return []

    corpus = [doc.get("text", "").lower().split() for doc in candidates]
    bm25 = BM25Okapi(corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    for idx, doc in enumerate(candidates):
        doc["bm25_score"] = round(float(scores[idx]), 6)

    candidates.sort(key=lambda d: d["bm25_score"], reverse=True)
    return candidates[:top_k]


def _reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Combine two ranked lists using Reciprocal Rank Fusion."""
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict[str, Any]] = {}

    def _doc_key(doc: dict[str, Any]) -> str:
        return f"{doc.get('source_id', '')}:{doc.get('chunk_index', '')}:{doc.get('page', '')}"

    for rank, doc in enumerate(vector_results):
        key = _doc_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        if key not in doc_map:
            doc_map[key] = doc

    for rank, doc in enumerate(bm25_results):
        key = _doc_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        if key not in doc_map:
            doc_map[key] = doc

    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    fused: list[dict[str, Any]] = []
    for key in sorted_keys[:top_k]:
        doc = doc_map[key]
        doc["rrf_score"] = round(rrf_scores[key], 6)
        doc.pop("embedding", None)
        doc.pop("vector_score", None)
        doc.pop("bm25_score", None)
        fused.append(doc)

    return fused


def hybrid_search(
    query: str,
    user_id: str,
    subject: str = "",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Execute hybrid search (BM25 + vector) with Reciprocal Rank Fusion.

    Returns the top-k chunks sorted by combined RRF score.  Each result
    includes: text, subject, source_id, page, chunk_index, rrf_score.
    """
    settings = get_settings()
    filters: dict[str, Any] = {"user_id": user_id}
    if subject:
        filters["subject"] = {"$regex": subject, "$options": "i"}

    safe_k = max(1, min(int(top_k), 20))
    # Fetch more candidates from each method to improve fusion quality.
    fetch_k = safe_k * 3

    logger.info(
        "hybrid_search input user_id=%s query=%r subject=%r top_k=%s",
        user_id, query, subject, safe_k,
    )

    query_embedding, _model_name = embed_text(query)

    vector_results = _vector_search(query_embedding, filters, fetch_k)
    bm25_results = _bm25_search(query, filters, fetch_k)

    fused = _reciprocal_rank_fusion(vector_results, bm25_results, safe_k)

    # Serialize ObjectIds etc.
    fused = serialize_documents(fused)

    logger.info("hybrid_search output results=%s", len(fused))
    return fused
