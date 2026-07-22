"""Load documents of several file types from a path into Document objects.

Supported out of the box: .txt, .md, .markdown, .rst, .csv (core, no extra
dependencies). .pdf, .docx, and images (.png/.jpg/.jpeg/.bmp/.tiff) need one
extra each -- see the ImportError messages below for the exact install
command; the base install works without any of them.
"""

import csv
import os
from dataclasses import dataclass, field


@dataclass
class Document:
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def _load_text_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_csv_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return ""
    header, data_rows = rows[0], rows[1:]
    lines = []
    for row in data_rows:
        pairs = ", ".join(f"{h}: {v}" for h, v in zip(header, row))
        lines.append(pairs)
    return "\n".join(lines)


def _load_pdf_file(path):
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "Reading PDF files requires pypdf. Install it with: pip install easy-rag[pdf]"
        ) from e
    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _load_docx_file(path):
    try:
        from docx import Document as DocxDocument
    except ImportError as e:
        raise ImportError(
            "Reading .docx files requires python-docx. Install it with: pip install easy-rag[docx]"
        ) from e
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            paragraphs.append(" | ".join(cell.text for cell in row.cells))
    return "\n\n".join(paragraphs)


def _load_image_file(path):
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Reading images requires pytesseract and Pillow, plus the Tesseract "
            "OCR engine installed on your system (https://github.com/tesseract-ocr/tesseract). "
            "Install the Python packages with: pip install easy-rag[ocr]"
        ) from e
    return pytesseract.image_to_string(Image.open(path))


_LOADERS = {
    ".csv": _load_csv_file,
    ".pdf": _load_pdf_file,
    ".docx": _load_docx_file,
}
for _ext in IMAGE_EXTENSIONS:
    _LOADERS[_ext] = _load_image_file


def load_documents(path):
    """Load every supported file under `path` (a single file or a directory,
    searched recursively) into a list of Document objects. Unsupported file
    types are silently skipped."""
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
        elif ext in _LOADERS:
            text = _LOADERS[ext](p)
        else:
            continue
        if text.strip():
            documents.append(Document(text=text, source=p))
    return documents
