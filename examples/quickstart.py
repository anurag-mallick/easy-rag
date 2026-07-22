"""Minimal end-to-end example: ingest the sample policy docs in this folder
and ask a question about them. Runs with no API keys and no extra installs.
"""

import os

from easy_rag import Pipeline

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")


def main():
    pipeline = Pipeline(embedder="hashing", vectorstore="numpy", llm="none")

    n_chunks = pipeline.ingest(DOCS_DIR)
    print(f"Indexed {n_chunks} chunk(s) from {DOCS_DIR}\n")

    question = "How long does a refund take to process?"
    print(f"Q: {question}\n")
    print(pipeline.query(question))


if __name__ == "__main__":
    main()
