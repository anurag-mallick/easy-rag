"""The Pipeline class: the one object most users need. Ingests documents,
embeds and stores them, retrieves relevant chunks for a question, and
optionally generates a synthesized answer from them."""

import json
import os

from .chunking import split_text
from .embeddings import get_embedder
from .loaders import load_documents
from .llm import get_generator
from .vectorstore import get_vectorstore


class Pipeline:
    def __init__(
        self,
        embedder="hashing",
        vectorstore="numpy",
        llm="none",
        chunk_size=800,
        chunk_overlap=120,
        embedder_kwargs=None,
        vectorstore_kwargs=None,
        llm_kwargs=None,
    ):
        self._embedder_name = embedder if isinstance(embedder, str) else embedder.name
        self._embedder_kwargs = embedder_kwargs or {}
        self._vectorstore_name = vectorstore if isinstance(vectorstore, str) else vectorstore.name
        self._vectorstore_kwargs = dict(vectorstore_kwargs or {})

        self.embedder = embedder if hasattr(embedder, "embed") else get_embedder(
            embedder, **self._embedder_kwargs
        )
        self.generator = llm if hasattr(llm, "generate") else get_generator(
            llm, **(llm_kwargs or {})
        )
        if self._vectorstore_name == "faiss" and "dim" not in self._vectorstore_kwargs:
            self._vectorstore_kwargs["dim"] = self.embedder.dim
        self.vectorstore = vectorstore if hasattr(vectorstore, "search") else get_vectorstore(
            self._vectorstore_name, **self._vectorstore_kwargs
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._manifest = {}  # absolute source path -> [mtime, size] at last ingest

    def ingest(self, path, force=False):
        """Load, chunk, embed, and index every supported file under `path`.

        Files already ingested (same path, mtime, and size as last time) are
        skipped, so calling ingest() repeatedly on the same folder -- e.g.
        from a folder watcher -- only processes files that are new or have
        changed. A changed file has its old chunks removed first, so it
        doesn't leave stale duplicates behind. Pass force=True to re-ingest
        everything regardless of the manifest.

        Returns the number of chunks added.
        """
        documents = load_documents(path)
        to_process = []  # (abs_source, fingerprint, chunks) for changed files
        for doc in documents:
            # Always key by the absolute path, so the same file ingested via
            # a relative path one time and an absolute path another still
            # dedups/replaces correctly instead of being treated as new.
            abs_source = os.path.abspath(doc.source)
            stat = os.stat(doc.source)
            fingerprint = [stat.st_mtime, stat.st_size]
            if not force and self._manifest.get(abs_source) == fingerprint:
                continue
            chunks = split_text(doc.text, self.chunk_size, self.chunk_overlap)
            to_process.append((abs_source, fingerprint, chunks))

        if not to_process:
            return 0

        all_chunks, all_sources = [], []
        for abs_source, _fingerprint, chunks in to_process:
            all_chunks.extend(chunks)
            all_sources.extend([abs_source] * len(chunks))

        # Embed before mutating any state. If this raises (a network error
        # for an API-backed embedder, say), nothing below has run yet, so a
        # failed ingest leaves the manifest and vector store exactly as they
        # were -- the file is retried on the next ingest() call instead of
        # being wrongly marked done with its content silently lost.
        vectors = self.embedder.embed(all_chunks) if all_chunks else None

        for abs_source, fingerprint, _chunks in to_process:
            if abs_source in self._manifest:
                self.vectorstore.remove_source(abs_source)
            self._manifest[abs_source] = fingerprint
        if all_chunks:
            self.vectorstore.add(vectors, all_chunks, all_sources)
        return len(all_chunks)

    def retrieve(self, question, top_k=4):
        """Return the top_k most relevant (score, chunk_text, source) tuples."""
        query_vector = self.embedder.embed([question])[0]
        return self.vectorstore.search(query_vector, top_k=top_k)

    def query(self, question, top_k=4):
        """Retrieve relevant context and generate an answer from it."""
        contexts = self.retrieve(question, top_k=top_k)
        return self.generator.generate(question, contexts)

    def save(self, path):
        """Persist the index and its embedder/vectorstore config to `path`
        (a path prefix; several files sharing that prefix will be written)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        config = {
            "embedder": self._embedder_name,
            "embedder_kwargs": self._embedder_kwargs,
            "vectorstore": self._vectorstore_name,
            "vectorstore_kwargs": self._vectorstore_kwargs,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        with open(path + ".config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)
        with open(path + ".manifest.json", "w", encoding="utf-8") as f:
            json.dump(self._manifest, f)
        self.vectorstore.save(path)

    @classmethod
    def load(cls, path, llm="none", llm_kwargs=None):
        """Reconstruct a Pipeline from an index saved with `save()`, using the
        same embedder and vector store it was built with."""
        with open(path + ".config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        pipeline = cls(
            embedder=config["embedder"],
            vectorstore=config["vectorstore"],
            llm=llm,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            embedder_kwargs=config["embedder_kwargs"],
            vectorstore_kwargs=config["vectorstore_kwargs"],
            llm_kwargs=llm_kwargs,
        )
        pipeline.vectorstore.load(path)
        manifest_path = path + ".manifest.json"
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                pipeline._manifest = json.load(f)
        return pipeline
