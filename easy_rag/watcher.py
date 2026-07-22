"""Watch a set of source folders and incrementally ingest new or changed
files as they appear -- e.g. so dropping a file into a watched folder gets
it added to the index automatically, without re-processing anything else.

Uses simple polling (no extra dependency) rather than OS filesystem events,
since it only needs to notice a change within a few seconds, not instantly.
"""

import time


def scan_once(pipeline, sources):
    """Ingest every registered source folder once. Thanks to Pipeline's
    manifest, only files that are new or changed since the last scan are
    actually re-embedded. Returns the number of chunks added this scan."""
    total = 0
    for folder in sources:
        total += pipeline.ingest(folder)
    return total


def watch(pipeline, sources, index_path, interval=5, on_scan=None):
    """Poll `sources` every `interval` seconds, ingesting new/changed files
    and saving the index whenever a scan adds something new. Runs until
    interrupted (e.g. Ctrl+C), then returns.

    `on_scan(n_chunks_added)` is called after every scan if provided --
    useful for CLI progress output.
    """
    if not sources:
        raise ValueError(
            "No source folders are registered for this index. "
            "Add one with: easyrag sources add <folder> --index " + index_path
        )
    try:
        while True:
            n_chunks = scan_once(pipeline, sources)
            if n_chunks:
                pipeline.save(index_path)
            if on_scan:
                on_scan(n_chunks)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
