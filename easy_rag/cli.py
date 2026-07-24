"""Command-line interface: `easyrag ingest`, `query`, `sources`, `watch`."""

import argparse
import os
import sys

from .pipeline import Pipeline
from .sources import add_source, load_sources, remove_source
from .watcher import watch as watch_loop


def _add_common_args(parser):
    parser.add_argument("--index", default=".easyrag/index", help="Index path prefix (default: .easyrag/index)")


def _load_or_create_pipeline(args):
    """Load the existing index if there is one, so ingestion is incremental
    and reuses the original embedder/vectorstore config; otherwise create a
    fresh pipeline from the CLI flags."""
    if os.path.exists(args.index + ".config.json"):
        return Pipeline.load(args.index), True
    return (
        Pipeline(
            embedder=args.embedder,
            vectorstore=args.vectorstore,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        ),
        False,
    )


def cmd_ingest(args):
    pipeline, existing = _load_or_create_pipeline(args)
    if existing:
        print(f"Index {args.index!r} already exists -- ingesting incrementally (new/changed files only).")

    if args.path:
        folders = [args.path]
    else:
        folders = load_sources(args.index)
        if not folders:
            print(
                "No path given and no source folders registered. Either pass a path, "
                f"or add one with: easyrag sources add <folder> --index {args.index}",
                file=sys.stderr,
            )
            sys.exit(1)

    total = 0
    for folder in folders:
        total += pipeline.ingest(folder, force=args.force)
    pipeline.save(args.index)
    verb = "Re-ingested" if args.force else "Added"
    print(f"{verb} {total} chunk(s) from {len(folders)} folder(s) into index {args.index!r}.")


def cmd_query(args):
    llm_kwargs = {}
    if args.model:
        llm_kwargs["model"] = args.model
    if args.model_path:
        llm_kwargs["model_path"] = args.model_path
    if args.base_url:
        llm_kwargs["base_url"] = args.base_url
    pipeline = Pipeline.load(args.index, llm=args.llm, llm_kwargs=llm_kwargs or None)
    answer = pipeline.query(args.question, top_k=args.top_k, min_score=args.min_score)
    print(answer)


def cmd_sources_add(args):
    folders = add_source(args.index, args.folder)
    print(f"Registered {os.path.abspath(args.folder)!r}. Source folders for {args.index!r}:")
    for f in folders:
        print(f"  {f}")


def cmd_sources_remove(args):
    folders = remove_source(args.index, args.folder)
    print(f"Unregistered {os.path.abspath(args.folder)!r}. Source folders for {args.index!r}:")
    for f in folders:
        print(f"  {f}")


def cmd_sources_list(args):
    folders = load_sources(args.index)
    if not folders:
        print(f"No source folders registered for {args.index!r}.")
        return
    print(f"Source folders for {args.index!r}:")
    for f in folders:
        print(f"  {f}")


def cmd_watch(args):
    pipeline, existing = _load_or_create_pipeline(args)
    folders = load_sources(args.index)
    if not existing:
        pipeline.save(args.index)  # so `easyrag sources ...` sees a config to attach to

    def on_scan(n_chunks):
        if n_chunks:
            print(f"Added {n_chunks} chunk(s).")

    def on_error(folder, error):
        print(f"Warning: failed to ingest {folder!r}: {error} (will retry next scan)", file=sys.stderr)

    print(f"Watching {len(folders)} folder(s) for {args.index!r} (checking every {args.interval}s, Ctrl+C to stop):")
    for f in folders:
        print(f"  {f}")
    watch_loop(pipeline, folders, args.index, interval=args.interval, on_scan=on_scan, on_error=on_error)
    print("Stopped watching.")


def build_parser():
    parser = argparse.ArgumentParser(prog="easyrag", description="Build and query a RAG pipeline from a folder of documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Load, chunk, embed, and index documents.")
    p_ingest.add_argument("path", nargs="?", default=None, help="File or directory to ingest. Omit to ingest all registered source folders instead.")
    _add_common_args(p_ingest)
    p_ingest.add_argument("--embedder", default="hashing", choices=["hashing", "local", "openai", "gemini", "llamacpp"])
    p_ingest.add_argument("--vectorstore", default="numpy", choices=["numpy", "faiss"])
    p_ingest.add_argument("--chunk-size", type=int, default=800)
    p_ingest.add_argument("--chunk-overlap", type=int, default=120)
    p_ingest.add_argument("--force", action="store_true", help="Re-ingest every file regardless of the manifest, instead of skipping unchanged ones")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="Ask a question against a previously built index.")
    p_query.add_argument("question")
    _add_common_args(p_query)
    p_query.add_argument("--llm", default="none", choices=["none", "anthropic", "openai", "gemini", "llamacpp"])
    p_query.add_argument("--model", default=None, help="Override the default model for the chosen --llm provider")
    p_query.add_argument("--model-path", default=None, help="Local .gguf file path (only used with --llm llamacpp; omit to auto-download a small default model)")
    p_query.add_argument("--base-url", default=None, help="Point --llm openai at a local OpenAI-compatible server instead of the real OpenAI API (e.g. a running llama-server)")
    p_query.add_argument("--top-k", type=int, default=4)
    p_query.add_argument("--min-score", type=float, default=None, help="Drop results below this similarity score instead of always returning top-k regardless of relevance (threshold is embedder-specific -- see retrieve()'s docstring)")
    p_query.set_defaults(func=cmd_query)

    p_sources = sub.add_parser("sources", help="Manage which folders an index ingests from.")
    sources_sub = p_sources.add_subparsers(dest="sources_command", required=True)

    p_sources_add = sources_sub.add_parser("add", help="Register a folder as a source for this index.")
    p_sources_add.add_argument("folder")
    _add_common_args(p_sources_add)
    p_sources_add.set_defaults(func=cmd_sources_add)

    p_sources_remove = sources_sub.add_parser("remove", help="Unregister a folder from this index.")
    p_sources_remove.add_argument("folder")
    _add_common_args(p_sources_remove)
    p_sources_remove.set_defaults(func=cmd_sources_remove)

    p_sources_list = sources_sub.add_parser("list", help="List the folders registered for this index.")
    _add_common_args(p_sources_list)
    p_sources_list.set_defaults(func=cmd_sources_list)

    p_watch = sub.add_parser("watch", help="Continuously watch registered source folders and auto-ingest new/changed files.")
    _add_common_args(p_watch)
    p_watch.add_argument("--interval", type=int, default=5, help="Seconds between folder scans (default: 5)")
    p_watch.add_argument("--embedder", default="hashing", choices=["hashing", "local", "openai", "gemini", "llamacpp"])
    p_watch.add_argument("--vectorstore", default="numpy", choices=["numpy", "faiss"])
    p_watch.add_argument("--chunk-size", type=int, default=800)
    p_watch.add_argument("--chunk-overlap", type=int, default=120)
    p_watch.set_defaults(func=cmd_watch)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
