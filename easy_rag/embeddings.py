"""Embedding providers, from a zero-dependency default up to real models.

Every embedder implements `embed(texts: list[str]) -> np.ndarray` returning an
(N, dim) float32 array of L2-normalized vectors, so cosine similarity is a
plain dot product regardless of which provider produced them.
"""

import hashlib
import re
import time

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _batched_call(texts, call_fn, batch_size=100, max_retries=3, backoff_base=1.0):
    """Call call_fn(batch_of_texts) -> (batch_n, dim) ndarray once per
    batch_size-sized slice of `texts`, concatenating the results into one
    (N, dim) array.

    Shared by every API-backed embedder (OpenAI, Gemini, ...) so this
    behavior lives in one place rather than being duplicated -- and
    potentially duplicated inconsistently -- per provider:

    - Splitting into batches avoids exceeding a provider's per-request
      token/item limit on a large ingest() call.
    - Retrying a failed batch with exponential backoff, instead of letting
      it abort the whole ingest immediately, rides out a transient rate
      limit or network blip. A permanent error (e.g. a bad API key) still
      raises after max_retries -- retrying just costs a few bounded
      seconds in that case rather than hanging indefinitely.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(max_retries + 1):
            try:
                results.append(call_fn(batch))
                break
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(backoff_base * (2**attempt))
    return np.concatenate(results, axis=0)


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


# Dimensions of OpenAI's current embedding models, keyed by model name --
# NOT derivable from the name alone (e.g. "text-embedding-ada-002" contains
# neither "small" nor "large" but is 1536-dimensional, the same as -3-small).
_OPENAI_EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(Embedder):
    """Embeddings via the OpenAI API, or any OpenAI-wire-compatible server.
    Install with: pip install easy-rag[openai]
    Requires the OPENAI_API_KEY environment variable (any non-empty string
    if base_url points at a local server that doesn't check it).

    Passing base_url lets this same class talk to a locally running
    `llama-server` (from llama.cpp) instead of the real OpenAI API -- its
    /v1/embedding endpoint is wire-compatible with the OpenAI embeddings API.
    See the 'llamacpp' embedder below for an alternative that runs the model
    in-process instead of managing a separate server.
    """

    name = "openai"

    def __init__(self, model="text-embedding-3-small", base_url=None, dim=None, batch_size=100):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' embedder requires the openai package. "
                "Install it with: pip install easy-rag[openai]"
            ) from e
        self._client = OpenAI(base_url=base_url) if base_url else OpenAI()
        self._model = model
        self._batch_size = batch_size
        if dim is not None:
            # Talking to a local/self-hosted server serving a model OpenAI
            # doesn't know about -- the caller must state its dimension.
            self.dim = dim
        elif model in _OPENAI_EMBEDDING_DIMS:
            self.dim = _OPENAI_EMBEDDING_DIMS[model]
        elif base_url:
            raise ValueError(
                f"Unknown model '{model}' for a custom base_url; pass dim=<embedding size> explicitly."
            )
        else:
            raise ValueError(
                f"Unknown OpenAI embedding model '{model}'; dim can't be inferred safely. "
                f"Known models: {sorted(_OPENAI_EMBEDDING_DIMS)}"
            )

    def embed(self, texts):
        def call(batch):
            resp = self._client.embeddings.create(model=self._model, input=list(batch))
            vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vecs / norms

        return _batched_call(list(texts), call, batch_size=self._batch_size)


# Dimensions of Google's current Gemini embedding models. gemini-embedding-001
# supports Matryoshka Representation Learning (configurable smaller output
# sizes via output_dim below), but its un-truncated default is 3072.
_GEMINI_EMBEDDING_DIMS = {
    "gemini-embedding-001": 3072,
    "text-embedding-004": 768,
}


class GeminiEmbedder(Embedder):
    """Embeddings via Google's Gemini API. Install with: pip install easy-rag[gemini]
    Requires the GEMINI_API_KEY environment variable.
    """

    name = "gemini"

    def __init__(self, model="gemini-embedding-001", output_dim=None, batch_size=100):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "The 'gemini' embedder requires the google-genai package. "
                "Install it with: pip install easy-rag[gemini]"
            ) from e
        self._genai = genai
        self._client = genai.Client()
        self._model = model
        self._output_dim = output_dim
        self._batch_size = batch_size
        if output_dim is not None:
            self.dim = output_dim
        elif model in _GEMINI_EMBEDDING_DIMS:
            self.dim = _GEMINI_EMBEDDING_DIMS[model]
        else:
            raise ValueError(
                f"Unknown Gemini embedding model '{model}'; dim can't be inferred safely. "
                f"Known models: {sorted(_GEMINI_EMBEDDING_DIMS)}, or pass output_dim=<size> explicitly."
            )

    def embed(self, texts):
        config = {"output_dimensionality": self._output_dim} if self._output_dim else None

        def call(batch):
            resp = self._client.models.embed_content(model=self._model, contents=list(batch), config=config)
            vecs = np.array([e.values for e in resp.embeddings], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vecs / norms

        return _batched_call(list(texts), call, batch_size=self._batch_size)


class LlamaCppEmbedder(Embedder):
    """Fully local embeddings via llama.cpp, in-process -- no server to run,
    no API key, no internet needed after the model is downloaded once.
    Install with: pip install easy-rag[llamacpp]

    With no model_path given, downloads (and caches) a small embedding GGUF
    model from Hugging Face Hub the first time it's used -- see the README
    for the exact model and its size before relying on this in an
    environment with restricted bandwidth or disk space.
    """

    name = "llamacpp"

    DEFAULT_REPO = "Qwen/Qwen3-Embedding-0.6B-GGUF"
    DEFAULT_FILENAME = "*q4_k_m.gguf"

    def __init__(self, model_path=None, repo_id=None, filename=None, n_ctx=2048, **llama_kwargs):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "The 'llamacpp' embedder requires llama-cpp-python. "
                "Install it with: pip install easy-rag[llamacpp]"
            ) from e
        if model_path:
            self._llm = Llama(model_path=model_path, embedding=True, n_ctx=n_ctx, verbose=False, **llama_kwargs)
        else:
            self._llm = Llama.from_pretrained(
                repo_id=repo_id or self.DEFAULT_REPO,
                filename=filename or self.DEFAULT_FILENAME,
                embedding=True,
                n_ctx=n_ctx,
                verbose=False,
                **llama_kwargs,
            )
        self.dim = self._llm.n_embd()

    def embed(self, texts):
        resp = self._llm.create_embedding(list(texts))
        vecs = np.array([d["embedding"] for d in resp["data"]], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


_REGISTRY = {
    "hashing": HashingEmbedder,
    "local": SentenceTransformerEmbedder,
    "openai": OpenAIEmbedder,
    "gemini": GeminiEmbedder,
    "llamacpp": LlamaCppEmbedder,
}


def get_embedder(name="hashing", **kwargs):
    """Look up an embedder by name: 'hashing' (default, zero deps), 'local'
    (sentence-transformers), 'openai' (OpenAI API, or any OpenAI-compatible
    server via base_url), 'gemini' (Google's Gemini API), or 'llamacpp'
    (fully local GGUF model, no server)."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown embedder '{name}'. Choose from: {sorted(_REGISTRY)}")
    return cls(**kwargs)
