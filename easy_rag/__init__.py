"""easy_rag: build a retrieval-augmented-generation pipeline from a folder of
documents in a few lines of code, with a zero-dependency default and
pluggable local/API-backed embedding and generation providers."""

from .pipeline import Pipeline
from .chunking import split_text
from .loaders import load_documents, Document

__version__ = "0.1.0"

__all__ = ["Pipeline", "split_text", "load_documents", "Document", "__version__"]
