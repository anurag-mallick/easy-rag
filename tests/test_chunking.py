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


def _mangled_words(original_text, chunks):
    """Words that appear in the chunked output but not in the source text
    at all -- i.e. fragments created by cutting a word in half."""
    full_words = set(original_text.replace(",", " ").replace(":", " ").split())
    chunk_words = set(" ".join(chunks).replace(",", " ").replace(":", " ").split())
    return chunk_words - full_words


def test_overlap_seed_does_not_split_a_word_in_half():
    # A stream of short, similar-length rows (no punctuation) is exactly
    # the shape that makes a raw current[-overlap:] slice land mid-word on
    # nearly every chunk -- e.g. "name: PersonN, ..." repeatedly getting
    # sliced to "me: PersonN, ...". This must never happen.
    rows = [f"name: Person{i}, role: Engineer, dept: Team{i % 5}" for i in range(80)]
    text = "\n\n".join(rows)

    chunks = split_text(text, chunk_size=200, overlap=40)

    assert not _mangled_words(text, chunks)
    assert not any(c.startswith(("me:", "e:", "am:")) for c in chunks[1:])


def test_hard_wrap_of_an_unpunctuated_paragraph_does_not_split_words():
    # A long paragraph with zero sentence-ending punctuation (a word list,
    # a log line, unpunctuated bullet points) is treated as one giant
    # "sentence" and hard-wrapped by chunk_size -- the raw character-count
    # wrap this used to do would slice words in half at every boundary.
    text = " ".join(["consectetur", "adipiscing", "elit", "department", "engineer"] * 40)

    chunks = split_text(text, chunk_size=60, overlap=20)

    assert not _mangled_words(text, chunks)
    assert all(len(c) <= 60 for c in chunks)
