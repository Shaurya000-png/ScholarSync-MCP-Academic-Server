from services.chunking import chunk_text, normalize_text


def test_normalize_text_collapses_whitespace():
    assert normalize_text("a\n\n b\t c") == "a b c"


def test_chunk_text_overlaps_words():
    text = " ".join(str(index) for index in range(120))
    chunks = chunk_text(text, chunk_size_words=50, overlap_words=10)

    assert chunks[0].startswith("0 1 2")
    assert chunks[1].startswith("40 41 42")
    assert chunks[-1].endswith("117 118 119")
