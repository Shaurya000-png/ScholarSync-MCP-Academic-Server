"""Tests for the RAG pipeline: chunking behaviour and cosine similarity edge cases."""

from __future__ import annotations

import math

from services.chunking import chunk_text
from services.embeddings import cosine_similarity


class TestChunkingProducesCorrectCount:
    """Verify chunk_text returns the expected number of chunks for a known input."""

    def test_exact_chunk_count(self) -> None:
        # 200 words, chunk_size=100, overlap=20 → step=80
        # Chunks: [0:100], [80:180], [160:200] → 3 chunks
        text = " ".join(f"word{i}" for i in range(200))
        chunks = chunk_text(text, chunk_size_words=100, overlap_words=20)
        assert len(chunks) == 3

    def test_single_chunk_for_short_text(self) -> None:
        text = "hello world"
        chunks = chunk_text(text, chunk_size_words=100, overlap_words=10)
        assert len(chunks) == 1

    def test_large_input_chunk_count(self) -> None:
        text = " ".join(f"w{i}" for i in range(500))
        chunks = chunk_text(text, chunk_size_words=100, overlap_words=20)
        # step=80: ceil(500/80) chunks but bounded by reaching end
        assert len(chunks) >= 5


class TestChunkTextLength:
    """Each chunk must not exceed the configured maximum word count."""

    def test_chunk_word_count_within_limit(self) -> None:
        text = " ".join(f"token{i}" for i in range(1000))
        max_words = 150
        chunks = chunk_text(text, chunk_size_words=max_words, overlap_words=30)
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count <= max_words, f"Chunk has {word_count} words, max is {max_words}"


class TestCosineSimilarityIdentical:
    """Cosine similarity of a vector with itself must be 1.0."""

    def test_identical_vectors(self) -> None:
        vec = [1.0, 2.0, 3.0, 4.0]
        assert math.isclose(cosine_similarity(vec, vec), 1.0, abs_tol=1e-9)

    def test_identical_unit_vectors(self) -> None:
        vec = [0.0, 1.0]
        assert cosine_similarity(vec, vec) == 1.0


class TestCosineSimilarityOrthogonal:
    """Cosine similarity of orthogonal vectors must be 0.0."""

    def test_orthogonal_2d(self) -> None:
        assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9)

    def test_orthogonal_3d(self) -> None:
        assert math.isclose(cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), 0.0, abs_tol=1e-9)


class TestChunkingEmptyInput:
    """Empty or whitespace-only text must return an empty list."""

    def test_empty_string(self) -> None:
        assert chunk_text("") == []

    def test_whitespace_only(self) -> None:
        assert chunk_text("   \n\t  ") == []
