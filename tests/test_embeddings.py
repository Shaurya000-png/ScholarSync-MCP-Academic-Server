from services.embeddings import cosine_similarity


def test_cosine_similarity_handles_matching_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_handles_empty_vectors():
    assert cosine_similarity([], []) == 0.0
