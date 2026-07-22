"""Command-line interface: `easyrag ingest` and `easyrag query`."""

import argparse
import os
import sys

from .pipeline import Pipeline


def _add_common_args(parser):
    parser.add_argument("--index", default=".easyrag/index", help="Index path prefix (default: .easyrag/index)")


def cmd_ingest(args):
    existing = os.path.exists(args.index + ".config.json")
    if existing:
        pipeline = Pipeline.load(args.index)
        print(f"Index {args.index!r} already exists -- appending to it (using its original embedder/vectorstore config).")
    else:
        pipeline = Pipeline(
            embedder=args.embedder,
            vectorstore=args.vectorstore,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    n_chunks = pipeline.ingest(args.path)
    pipeline.save(args.index)
    verb = "Added" if existing else "Ingested"
    print(f"{verb} {n_chunks} chunk(s) from {args.path!r} into index {args.index!r}.")


def cmd_query(args):
    llm_kwargs = {"model": args.model} if args.model else None
    pipeline = Pipeline.load(args.index, llm=args.llm, llm_kwargs=llm_kwargs)
    answer = pipeline.query(args.question, top_k=args.top_k)
    print(answer)


def build_parser():
    parser = argparse.ArgumentParser(prog="easyrag", description="Build and query a RAG pipeline from a folder of documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Load, chunk, embed, and index documents.")
    p_ingest.add_argument("path", help="File or directory to ingest (.txt, .md, .pdf)")
    _add_common_args(p_ingest)
    p_ingest.add_argument("--embedder", default="hashing", choices=["hashing", "local", "openai"])
    p_ingest.add_argument("--vectorstore", default="numpy", choices=["numpy", "faiss"])
    p_ingest.add_argument("--chunk-size", type=int, default=800)
    p_ingest.add_argument("--chunk-overlap", type=int, default=120)
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="Ask a question against a previously built index.")
    p_query.add_argument("question")
    _add_common_args(p_query)
    p_query.add_argument("--llm", default="none", choices=["none", "anthropic", "openai"])
    p_query.add_argument("--model", default=None, help="Override the default model for the chosen --llm provider")
    p_query.add_argument("--top-k", type=int, default=4)
    p_query.set_defaults(func=cmd_query)

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
