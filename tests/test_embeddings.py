import numpy as np
import pytest

from easy_rag.embeddings import _batched_call, _OPENAI_EMBEDDING_DIMS, HashingEmbedder


def test_batched_call_splits_into_multiple_batches():
    calls = []

    def call_fn(batch):
        calls.append(list(batch))
        return np.ones((len(batch), 2), dtype=np.float32)

    result = _batched_call([f"t{i}" for i in range(25)], call_fn, batch_size=10)

    assert result.shape == (25, 2)
    assert len(calls) == 3  # 10 + 10 + 5
    assert [len(c) for c in calls] == [10, 10, 5]


def test_batched_call_with_no_texts_returns_empty_array():
    result = _batched_call([], lambda batch: np.ones((len(batch), 2)), batch_size=10)
    assert result.shape == (0, 0)


def test_batched_call_retries_a_transient_failure_then_succeeds(monkeypatch):
    monkeypatch.setattr("easy_rag.embeddings.time.sleep", lambda _seconds: None)
    attempts = {"n": 0}

    def call_fn(batch):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("simulated transient rate limit")
        return np.ones((len(batch), 2), dtype=np.float32)

    result = _batched_call(["a", "b"], call_fn, batch_size=10, max_retries=3)

    assert result.shape == (2, 2)
    assert attempts["n"] == 3


def test_batched_call_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("easy_rag.embeddings.time.sleep", lambda _seconds: None)

    def call_fn(_batch):
        raise RuntimeError("permanent failure (e.g. bad API key)")

    with pytest.raises(RuntimeError, match="permanent failure"):
        _batched_call(["a"], call_fn, batch_size=10, max_retries=2)


def test_hashing_embedder_returns_normalized_vectors():
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.embed(["hello world", "goodbye world"])
    assert vectors.shape == (2, 64)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0)


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder(dim=64)
    a = embedder.embed(["the quick brown fox"])
    b = embedder.embed(["the quick brown fox"])
    assert np.allclose(a, b)


def test_known_openai_model_dims_are_correct():
    # text-embedding-ada-002 contains neither "small" nor "large" -- a
    # name-based heuristic previously mapped it to 3072, but it is actually
    # 1536-dimensional. This must come from an explicit lookup, not a guess.
    assert _OPENAI_EMBEDDING_DIMS["text-embedding-ada-002"] == 1536
    assert _OPENAI_EMBEDDING_DIMS["text-embedding-3-small"] == 1536
    assert _OPENAI_EMBEDDING_DIMS["text-embedding-3-large"] == 3072


def test_openai_embedder_rejects_unknown_model(monkeypatch):
    # importorskip called here, inside the test, only skips this test when
    # the openai package isn't installed -- calling it at module level
    # instead would skip every test in this file, hashing tests included.
    pytest.importorskip("openai", reason="openai package not installed")
    from easy_rag.embeddings import OpenAIEmbedder

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    with pytest.raises(ValueError, match="Unknown OpenAI embedding model"):
        OpenAIEmbedder(model="some-future-model-name")


def test_openai_embedder_known_model_sets_correct_dim(monkeypatch):
    pytest.importorskip("openai", reason="openai package not installed")
    from easy_rag.embeddings import OpenAIEmbedder

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    embedder = OpenAIEmbedder(model="text-embedding-3-large")
    assert embedder.dim == 3072


def test_openai_embedder_unknown_model_with_base_url_needs_explicit_dim(monkeypatch):
    # Pointing at a self-hosted server (e.g. llama-server) serving a model
    # OpenAI doesn't know about -- dim must be stated explicitly rather than
    # silently guessed or left to crash later inside the vector store.
    pytest.importorskip("openai", reason="openai package not installed")
    from easy_rag.embeddings import OpenAIEmbedder

    monkeypatch.setenv("OPENAI_API_KEY", "unused-placeholder")
    with pytest.raises(ValueError, match="pass dim="):
        OpenAIEmbedder(model="my-local-model", base_url="http://localhost:8080/v1")


def test_openai_embedder_base_url_with_explicit_dim_is_accepted(monkeypatch):
    pytest.importorskip("openai", reason="openai package not installed")
    from easy_rag.embeddings import OpenAIEmbedder

    monkeypatch.setenv("OPENAI_API_KEY", "unused-placeholder")
    embedder = OpenAIEmbedder(model="my-local-model", base_url="http://localhost:8080/v1", dim=768)
    assert embedder.dim == 768
    assert str(embedder._client.base_url).startswith("http://localhost:8080/v1")


def test_openai_embedder_batches_large_requests(monkeypatch):
    # Confirms embed() actually goes through _batched_call rather than
    # sending every text in one request regardless of how many there are.
    openai_pkg = pytest.importorskip("openai", reason="openai package not installed")
    from easy_rag.embeddings import OpenAIEmbedder

    monkeypatch.setenv("OPENAI_API_KEY", "unused-placeholder")

    class FakeEmbeddingItem:
        def __init__(self, embedding):
            self.embedding = embedding

    class FakeResponse:
        def __init__(self, n):
            self.data = [FakeEmbeddingItem([1.0, 0.0]) for _ in range(n)]

    calls = []

    class FakeEmbeddings:
        def create(self, model, input):
            calls.append(list(input))
            return FakeResponse(len(input))

    class FakeClient:
        def __init__(self, *a, **kw):
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(openai_pkg, "OpenAI", FakeClient)

    embedder = OpenAIEmbedder(model="text-embedding-3-small", batch_size=4)
    vectors = embedder.embed([f"t{i}" for i in range(10)])

    assert vectors.shape == (10, 2)
    assert [len(c) for c in calls] == [4, 4, 2]


def test_known_gemini_model_dims_are_correct():
    from easy_rag.embeddings import _GEMINI_EMBEDDING_DIMS

    assert _GEMINI_EMBEDDING_DIMS["gemini-embedding-001"] == 3072
    assert _GEMINI_EMBEDDING_DIMS["text-embedding-004"] == 768


def test_gemini_embedder_requires_install_with_helpful_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "google":  # `from google import genai` imports "google", not "google.genai"
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from easy_rag.embeddings import GeminiEmbedder

    with pytest.raises(ImportError, match=r"easy-rag\[gemini\]"):
        GeminiEmbedder()


def test_gemini_embedder_rejects_unknown_model_without_output_dim(monkeypatch):
    genai = pytest.importorskip("google.genai", reason="google-genai package not installed")
    monkeypatch.setenv("GEMINI_API_KEY", "unused-placeholder")

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr(genai, "Client", FakeClient)

    from easy_rag.embeddings import GeminiEmbedder

    with pytest.raises(ValueError, match="Unknown Gemini embedding model"):
        GeminiEmbedder(model="some-future-model-name")


def test_gemini_embedder_wiring_with_a_stubbed_client(monkeypatch):
    # Verifies our extraction/normalization code without a real network call:
    # stub genai.Client itself so this stays a fast, offline unit test.
    genai = pytest.importorskip("google.genai", reason="google-genai package not installed")
    monkeypatch.setenv("GEMINI_API_KEY", "unused-placeholder")

    class FakeEmbedding:
        def __init__(self, values):
            self.values = values

    class FakeResponse:
        def __init__(self, embeddings):
            self.embeddings = embeddings

    class FakeModels:
        def embed_content(self, model, contents, config=None):
            return FakeResponse([FakeEmbedding([1.0, 0.0, 0.0]) for _ in contents])

    class FakeClient:
        def __init__(self, *a, **kw):
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)

    from easy_rag.embeddings import GeminiEmbedder

    embedder = GeminiEmbedder(model="gemini-embedding-001", output_dim=3)
    assert embedder.dim == 3
    vectors = embedder.embed(["a", "b"])
    assert vectors.shape == (2, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_gemini_embedder_batches_large_requests(monkeypatch):
    genai = pytest.importorskip("google.genai", reason="google-genai package not installed")
    monkeypatch.setenv("GEMINI_API_KEY", "unused-placeholder")

    class FakeEmbedding:
        def __init__(self, values):
            self.values = values

    class FakeResponse:
        def __init__(self, n):
            self.embeddings = [FakeEmbedding([1.0, 0.0]) for _ in range(n)]

    calls = []

    class FakeModels:
        def embed_content(self, model, contents, config=None):
            calls.append(list(contents))
            return FakeResponse(len(contents))

    class FakeClient:
        def __init__(self, *a, **kw):
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)

    from easy_rag.embeddings import GeminiEmbedder

    embedder = GeminiEmbedder(model="gemini-embedding-001", output_dim=2, batch_size=4)
    vectors = embedder.embed([f"t{i}" for i in range(10)])

    assert vectors.shape == (10, 2)
    assert [len(c) for c in calls] == [4, 4, 2]


def test_llamacpp_embedder_requires_install_with_helpful_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "llama_cpp":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from easy_rag.embeddings import LlamaCppEmbedder

    with pytest.raises(ImportError, match=r"easy-rag\[llamacpp\]"):
        LlamaCppEmbedder(model_path="does_not_matter.gguf")


def test_llamacpp_embedder_wiring_with_a_stubbed_model(monkeypatch, tmp_path):
    # Verifies our extraction/normalization code is correct without
    # downloading or running a real several-hundred-MB GGUF model: stub out
    # llama_cpp.Llama itself so this stays a fast, offline unit test.
    llama_cpp = pytest.importorskip("llama_cpp", reason="llama-cpp-python not installed")

    class FakeLlama:
        def __init__(self, model_path=None, embedding=None, n_ctx=None, verbose=None, **kw):
            assert embedding is True

        def n_embd(self):
            return 3

        def create_embedding(self, texts):
            return {"data": [{"embedding": [1.0, 0.0, 0.0]} for _ in texts]}

    monkeypatch.setattr(llama_cpp, "Llama", FakeLlama)

    from easy_rag.embeddings import LlamaCppEmbedder

    embedder = LlamaCppEmbedder(model_path=str(tmp_path / "fake.gguf"))
    assert embedder.dim == 3
    vectors = embedder.embed(["a", "b"])
    assert vectors.shape == (2, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
