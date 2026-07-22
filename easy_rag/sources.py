"""Manage the list of folders a given index should ingest from.

The source list is a small JSON file saved next to the index (`<index
path>.sources.json`), so `easyrag ingest` and `easyrag watch` know which
folders to scan without the user having to re-type them every time.
"""

import json
import os


def _sources_path(index_path):
    return index_path + ".sources.json"


def load_sources(index_path):
    """Return the list of absolute source folder paths registered for this
    index, or an empty list if none have been added yet."""
    path = _sources_path(index_path)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_sources(index_path, sources):
    path = _sources_path(index_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2)


def add_source(index_path, folder):
    """Register a folder to be ingested for this index. Returns the updated
    list of source folders."""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Not a directory: {folder}")
    sources = load_sources(index_path)
    if folder not in sources:
        sources.append(folder)
        save_sources(index_path, sources)
    return sources


def remove_source(index_path, folder):
    """Unregister a folder from this index. Returns the updated list of
    source folders. Does not remove any already-ingested chunks; ingest a
    fresh index if you also want the folder's content removed from it."""
    folder = os.path.abspath(folder)
    sources = load_sources(index_path)
    if folder in sources:
        sources.remove(folder)
        save_sources(index_path, sources)
    return sources
