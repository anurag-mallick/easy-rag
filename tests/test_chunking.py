import pytest

from easy_rag.chunking import split_text


def test_short_text_is_a_single_chunk():
    text = "This is a short paragraph."
    chunks = split_text(text, chunk_size=800, overlap=120)
    assert chunks == [text]


def test_long_text_is_split_into_multiple_chunks():
    text = "\n\n".join(f"Sentence number {i}." for i in range(200))
    chunks = split_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunks_overlap():
    text = "\n\n".join(f"This is paragraph number {i} with some extra words." for i in range(50))
    chunks = split_text(text, chunk_size=150, overlap=50)
    assert len(chunks) > 1
    # Consecutive chunks should share some trailing/leading text.
    for a, b in zip(chunks, chunks[1:]):
        assert any(word in b for word in a.split()[-3:])


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        split_text("hello world", chunk_size=10, overlap=10)


def test_overlap_more_than_half_chunk_size_raises():
    # An overlap this close to chunk_size only advances one character per
    # chunk, which explodes into an enormous number of chunks for any
    # realistically sized document -- this must be rejected up front.
    with pytest.raises(ValueError, match="at most half"):
        split_text("x" * 1000, chunk_size=100, overlap=99)


def test_overlap_at_exactly_half_chunk_size_is_allowed():
    chunks = split_text("word " * 500, chunk_size=100, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_empty_text_returns_no_chunks():
    assert split_text("   \n\n  ", chunk_size=100, overlap=10) == []
