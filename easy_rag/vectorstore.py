"""Vector storage and similarity search, from a zero-dependency default up to
a faster approximate-nearest-neighbor backend for larger corpora."""

import json
import os

import numpy as np


class VectorStore:
    name = "base"

    def add(self, vectors, chunks, sources):
        raise NotImplementedError

    def remove_source(self, source):
        """Remove every chunk previously added with this exact source path.
        Used to replace a file's old chunks when it changes, so re-ingesting
        a modified file doesn't leave stale duplicates behind."""
        raise NotImplementedError

    def search(self, query_vector, top_k=4):
        """Return a list of (score, chunk_text, source) tuples, best first."""
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


class NumpyVectorStore(VectorStore):
    """Brute-force cosine-similarity search over an in-memory matrix.

    No extra dependencies beyond numpy. Fine for corpora up to roughly a few
    tens of thousands of chunks; switch to FaissVectorStore for larger ones.
    """

    name = "numpy"

    def __init__(self):
        self._vectors = None  # (N, dim) float32, L2-normalized
        self._chunks = []
        self._sources = []

    def add(self, vectors, chunks, sources):
        vectors = np.asarray(vectors, dtype=np.float32)
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._chunks.extend(chunks)
        self._sources.extend(sources)

    def remove_source(self, source):
        if not self._chunks:
            return
        keep = [i for i, s in enumerate(self._sources) if s != source]
        if len(keep) == len(self._chunks):
            return  # nothing to remove
        self._vectors = self._vectors[keep] if keep else None
        self._chunks = [self._chunks[i] for i in keep]
        self._sources = [self._sources[i] for i in keep]

    def search(self, query_vector, top_k=4):
        if self._vectors is None or len(self._chunks) == 0:
            return []
        scores = self._vectors @ np.asarray(query_vector, dtype=np.float32)
        top_k = min(top_k, len(scores))
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [(float(scores[i]), self._chunks[i], self._sources[i]) for i in idx]

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path + ".vectors.npy", self._vectors if self._vectors is not None else np.zeros((0, 0)))
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": self._chunks, "sources": self._sources}, f)

    def load(self, path):
        vectors = np.load(path + ".vectors.npy")
        # An index saved before anything was ever added stores a (0, 0)
        # placeholder (see save()). Keeping that as-is instead of None would
        # make the next add() try to np.vstack it against real (N, dim)
        # vectors and crash on a dimension mismatch.
        self._vectors = vectors if vectors.size > 0 else None
        with open(path + ".meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._chunks = meta["chunks"]
        self._sources = meta["sources"]
        return self

    def __len__(self):
        return len(self._chunks)


class FaissVectorStore(VectorStore):
    """Approximate nearest-neighbor search via FAISS.
    Install with: pip install easy-rag[local]

    FAISS's flat index has no in-place removal, so a copy of every vector is
    kept in `_vectors` to allow rebuilding the index on `remove_source()` --
    an operation expected to be rare (only when a previously-ingested file
    changes), not on the hot path of normal ingestion or search.
    """

    name = "faiss"

    def __init__(self, dim):
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "The 'faiss' vector store requires faiss-cpu. "
                "Install it with: pip install easy-rag[local]"
            ) from e
        self._faiss = faiss
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._vectors = []  # list of 1-D float32 arrays, parallel to _chunks
        self._chunks = []
        self._sources = []

    def add(self, vectors, chunks, sources):
        vectors = np.asarray(vectors, dtype=np.float32)
        self._index.add(vectors)
        self._vectors.extend(list(vectors))
        self._chunks.extend(chunks)
        self._sources.extend(sources)

    def _rebuild(self):
        self._index = self._faiss.IndexFlatIP(self._dim)
        if self._vectors:
            self._index.add(np.stack(self._vectors).astype(np.float32))

    def remove_source(self, source):
        if not self._chunks:
            return
        keep = [i for i, s in enumerate(self._sources) if s != source]
        if len(keep) == len(self._chunks):
            return  # nothing to remove
        self._vectors = [self._vectors[i] for i in keep]
        self._chunks = [self._chunks[i] for i in keep]
        self._sources = [self._sources[i] for i in keep]
        self._rebuild()

    def search(self, query_vector, top_k=4):
        if self._index.ntotal == 0:
            return []
        top_k = min(top_k, self._index.ntotal)
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, idxs = self._index.search(q, top_k)
        return [
            (float(scores[0][i]), self._chunks[idxs[0][i]], self._sources[idxs[0][i]])
            for i in range(len(idxs[0]))
            if idxs[0][i] != -1
        ]

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._faiss.write_index(self._index, path + ".faiss")
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": self._chunks, "sources": self._sources, "dim": self._dim}, f)

    def load(self, path):
        self._index = self._faiss.read_index(path + ".faiss")
        with open(path + ".meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._chunks = meta["chunks"]
        self._sources = meta["sources"]
        self._dim = meta.get("dim", self._index.d)
        # Recover a parallel vector list from the index itself so
        # remove_source() can rebuild after a load().
        self._vectors = [self._index.reconstruct(i) for i in range(self._index.ntotal)]
        return self

    def __len__(self):
        return len(self._chunks)


def get_vectorstore(name="numpy", **kwargs):
    """Look up a vector store by name: 'numpy' (default, zero deps) or
    'faiss' (approximate nearest neighbor, needs faiss-cpu)."""
    if name == "numpy":
        return NumpyVectorStore()
    if name == "faiss":
        return FaissVectorStore(**kwargs)
    raise ValueError(f"Unknown vector store '{name}'. Choose from: ['numpy', 'faiss']")
