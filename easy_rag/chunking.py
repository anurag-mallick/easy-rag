"""Split long text into overlapping chunks sized for embedding models.

Splits on paragraph, then sentence, then hard character boundaries — in that
order of preference — so chunks break at natural points whenever possible.
Every boundary this module ever cuts at -- paragraph, sentence, hard-wrap,
and the overlap seed between consecutive chunks -- is nudged to the nearest
space, so chunking never slices a word in half except when a single
unbroken run of non-space characters (e.g. a long URL) leaves no boundary
to snap to.
"""

import re

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _word_safe_cut(text, max_len):
    """Return an index <= max_len at which `text` can be split without
    landing mid-word: the last space at or before max_len, or max_len
    itself if there's no space in that window to snap to."""
    cut = text.rfind(" ", 0, max_len + 1)
    return cut if cut > 0 else max_len


def _word_safe_suffix(text, target_len):
    """Return a suffix of `text` at most target_len characters long, nudged
    forward to start right after a space so it begins at a whole word
    instead of mid-word. A raw `text[-target_len:]` slice has no notion of
    word boundaries, so on structured input (e.g. a stream of short,
    similar-length rows) it lands mid-word on almost every single chunk --
    not a rare edge case. Falls back to the raw slice if there's no space
    in the tail window at all (e.g. one long unbroken token)."""
    if len(text) <= target_len:
        return text
    tail = text[-target_len:]
    space_idx = tail.find(" ")
    if 0 <= space_idx < len(tail) - 1:
        return tail[space_idx + 1 :]
    return tail


def _hard_wrap(text, chunk_size):
    """Split text into pieces of at most chunk_size characters, breaking at
    a word boundary rather than an arbitrary character count. This matters
    beyond rare edge cases: any long paragraph with no sentence-ending
    punctuation at all (a word list, a log dump, unpunctuated bullet
    points) is treated as one giant "sentence" and hard-wrapped here."""
    pieces = []
    while len(text) > chunk_size:
        cut = _word_safe_cut(text, chunk_size)
        pieces.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        pieces.append(text)
    return pieces


def _split_oversized(piece, chunk_size):
    sentences = _SENTENCE_SPLIT.split(piece)
    out = []
    for s in sentences:
        if len(s) <= chunk_size:
            out.append(s)
        else:
            out.extend(_hard_wrap(s, chunk_size))
    return out


def split_text(text, chunk_size=800, overlap=120):
    """Return a list of overlapping text chunks, each at most ~chunk_size
    characters, with `overlap` characters of context repeated between
    consecutive chunks so a fact split across a boundary is still findable.

    Every cut this function makes is nudged to a word boundary, so words
    are not split in half -- with one narrow exception: if `overlap` is
    smaller than a single word in the text (e.g. overlap=10 against an
    11-character word), there may not be enough room in the overlap seed
    to include that word whole, and it can be truncated there. This does
    not affect the chunks themselves, only the small repeated-context seed
    between them, and does not occur at the library's default overlap or
    any reasonably word-sized overlap value.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap > chunk_size // 2:
        raise ValueError(
            "overlap must be >= 0 and at most half of chunk_size -- each new "
            "chunk only advances by (chunk_size - overlap) characters, so a "
            "larger overlap makes chunking produce an explosively large "
            "number of chunks for even modest-sized documents"
        )

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
            current = (_word_safe_suffix(current, overlap) + " " + unit).strip() if current else unit
            # If a single unit alone still exceeds chunk_size, hard-wrap it,
            # carrying a word-safe overlap seed forward at each cut.
            while len(current) > chunk_size:
                cut = _word_safe_cut(current, chunk_size)
                chunks.append(current[:cut].rstrip())
                current = _word_safe_suffix(current[:cut], overlap) + " " + current[cut:].lstrip()
    if current:
        chunks.append(current)
    return chunks
