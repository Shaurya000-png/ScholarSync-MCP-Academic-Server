"""Embedding provider with sentence-transformers and a deterministic fallback."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from functools import lru_cache

from config.settings import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        return SentenceTransformer(settings.embedding_model_name)
    except Exception as exc:
        logger.warning("sentence-transformers unavailable; using fallback embeddings: %s", exc)
        return None


def _fallback_embedding(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed_text(text: str) -> tuple[list[float], str]:
    """Create an embedding for text and return the vector plus model name."""
    settings = get_settings()
    model = _load_sentence_transformer()
    if model is not None:
        vector = model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector.tolist()], settings.embedding_model_name

    model_name = f"fallback-hashing-{settings.embedding_dimension}"
    return _fallback_embedding(text, settings.embedding_dimension), model_name


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity for two vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
