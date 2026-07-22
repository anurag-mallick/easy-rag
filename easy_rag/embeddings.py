"""Embedding providers, from a zero-dependency default up to real models.

Every embedder implements `embed(texts: list[str]) -> np.ndarray` returning an
(N, dim) float32 array of L2-normalized vectors, so cosine similarity is a
plain dot product regardless of which provider produced them.
"""

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder:
    name = "base"
    dim = 0

    def embed(self, texts):
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Deterministic bag-of-n-grams embedding via the hashing trick.

    No model download, no API key, no GPU — just numpy. Good enough to
    demonstrate and test a full RAG pipeline instantly; swap in
    SentenceTransformerEmbedder or an API-backed embedder for real accuracy.
    """

    name = "hashing"

    def __init__(self, dim=512, ngram_range=(1, 2)):
        self.dim = dim
        self.ngram_range = ngram_range

    def _tokens(self, text):
        words = _TOKEN_RE.findall(text.lower())
        tokens = []
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            for i in range(len(words) - n + 1):
                tokens.append(" ".join(words[i : i + n]))
        return tokens or words

    def _vector(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(text):
            h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            v[idx] += sign
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def embed(self, texts):
        return np.stack([self._vector(t) for t in texts]).astype(np.float32)


class SentenceTransformerEmbedder(Embedder):
    """Real sentence embeddings via the `sentence-transformers` library.
    Install with: pip install easy-rag[local]
    """

    name = "local"

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "The 'local' embedder requires sentence-transformers. "
                "Install it with: pip install easy-rag[local]"
            ) from e
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts):
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


class OpenAIEmbedder(Embedder):
    """Embeddings via the OpenAI API. Install with: pip install easy-rag[openai]
    Requires the OPENAI_API_KEY environment variable.
    """

    name = "openai"

    def __init__(self, model="text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' embedder requires the openai package. "
                "Install it with: pip install easy-rag[openai]"
            ) from e
        self._client = OpenAI()
        self._model = model
        self.dim = 1536 if "small" in model else 3072

    def embed(self, texts):
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


_REGISTRY = {
    "hashing": HashingEmbedder,
    "local": SentenceTransformerEmbedder,
    "openai": OpenAIEmbedder,
}


def get_embedder(name="hashing", **kwargs):
    """Look up an embedder by name: 'hashing' (default, zero deps), 'local'
    (sentence-transformers), or 'openai' (OpenAI API)."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown embedder '{name}'. Choose from: {sorted(_REGISTRY)}")
    return cls(**kwargs)
