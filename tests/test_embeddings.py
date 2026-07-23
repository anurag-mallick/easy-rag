import numpy as np
import pytest

from easy_rag.embeddings import _OPENAI_EMBEDDING_DIMS, HashingEmbedder


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
