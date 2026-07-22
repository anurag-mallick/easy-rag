"""Vector storage and similarity search, from a zero-dependency default up to
a faster approximate-nearest-neighbor backend for larger corpora."""

import json
import os

import numpy as np


class VectorStore:
    name = "base"

    def add(self, vectors, chunks, sources):
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
        self._vectors = np.load(path + ".vectors.npy")
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
        self._index = faiss.IndexFlatIP(dim)
        self._chunks = []
        self._sources = []

    def add(self, vectors, chunks, sources):
        self._index.add(np.asarray(vectors, dtype=np.float32))
        self._chunks.extend(chunks)
        self._sources.extend(sources)

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
            json.dump({"chunks": self._chunks, "sources": self._sources}, f)

    def load(self, path):
        self._index = self._faiss.read_index(path + ".faiss")
        with open(path + ".meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._chunks = meta["chunks"]
        self._sources = meta["sources"]
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
