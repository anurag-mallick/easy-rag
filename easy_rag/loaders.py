"""Load plain-text, Markdown, and PDF files from a path into Document objects."""

import os
from dataclasses import dataclass, field


@dataclass
class Document:
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst"}


def _load_text_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_pdf_file(path):
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "Reading PDF files requires pypdf. Install it with: pip install easy-rag[pdf]"
        ) from e
    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(path):
    """Load every supported file under `path` (a single file or a directory,
    searched recursively) into a list of Document objects."""
    paths = []
    if os.path.isfile(path):
        paths = [path]
    elif os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for name in files:
                paths.append(os.path.join(root, name))
    else:
        raise FileNotFoundError(f"No such file or directory: {path}")

    documents = []
    for p in sorted(paths):
        ext = os.path.splitext(p)[1].lower()
        if ext in TEXT_EXTENSIONS:
            text = _load_text_file(p)
        elif ext == ".pdf":
            text = _load_pdf_file(p)
        else:
            continue
        if text.strip():
            documents.append(Document(text=text, source=p))
    return documents
