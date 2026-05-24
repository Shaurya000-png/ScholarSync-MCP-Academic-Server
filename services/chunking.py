"""Text chunking helpers for retrieval pipelines."""

from __future__ import annotations


def normalize_text(text: str) -> str:
    """Collapse whitespace while preserving readable text."""
    return " ".join(text.split())


def chunk_text(text: str, chunk_size_words: int = 420, overlap_words: int = 80) -> list[str]:
    """Split text into overlapping word chunks."""
    words = normalize_text(text).split()
    if not words:
        return []

    safe_size = max(50, chunk_size_words)
    safe_overlap = max(0, min(overlap_words, safe_size - 1))
    step = safe_size - safe_overlap

    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = words[start : start + safe_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + safe_size >= len(words):
            break
    return chunks
