"""Split long text into overlapping chunks sized for embedding models.

Splits on paragraph, then sentence, then hard character boundaries — in that
order of preference — so chunks break at natural points whenever possible.
"""

import re

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_oversized(piece, chunk_size):
    sentences = _SENTENCE_SPLIT.split(piece)
    out = []
    for s in sentences:
        if len(s) <= chunk_size:
            out.append(s)
        else:
            # Still too long (e.g. one giant run-on sentence): hard-wrap.
            for i in range(0, len(s), chunk_size):
                out.append(s[i : i + chunk_size])
    return out


def split_text(text, chunk_size=800, overlap=120):
    """Return a list of overlapping text chunks, each at most ~chunk_size
    characters, with `overlap` characters of context repeated between
    consecutive chunks so a fact split across a boundary is still findable."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size")

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    units = []
    for p in paragraphs:
        units.extend(_split_oversized(p, chunk_size) if len(p) > chunk_size else [p])

    chunks = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = (current[-overlap:] + " " + unit).strip() if current else unit
            # If a single unit alone still exceeds chunk_size, hard-wrap it.
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap :]
    if current:
        chunks.append(current)
    return chunks
