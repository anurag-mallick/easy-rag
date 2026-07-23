import numpy as np
import pytest

from easy_rag.vectorstore import FaissVectorStore, NumpyVectorStore


def test_search_returns_most_similar_first():
    store = NumpyVectorStore()
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )
    store.add(vectors, ["chunk_a", "chunk_b", "chunk_c"], ["a.txt", "b.txt", "c.txt"])

    results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)

    assert len(results) == 2
    assert results[0][1] == "chunk_a"
    assert results[0][0] >= results[1][0]


def test_empty_store_returns_no_results():
    store = NumpyVectorStore()
    results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)
    assert results == []


def test_save_and_load_roundtrip(tmp_path):
    store = NumpyVectorStore()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store.add(vectors, ["chunk_a", "chunk_b"], ["a.txt", "b.txt"])

    prefix = str(tmp_path / "index")
    store.save(prefix)

    reloaded = NumpyVectorStore().load(prefix)
    results = reloaded.search(np.array([0.0, 1.0], dtype=np.float32), top_k=1)

    assert len(reloaded) == 2
    assert results[0][1] == "chunk_b"


def test_saving_and_reloading_an_empty_store_then_adding_does_not_crash(tmp_path):
    # A store saved before anything was ever added persists a (0, 0)
    # placeholder array; reloading it must not leave that placeholder in
    # place, or the next add() crashes trying to vstack it against real
    # (N, dim) vectors.
    prefix = str(tmp_path / "index")
    NumpyVectorStore().save(prefix)

    reloaded = NumpyVectorStore().load(prefix)
    assert len(reloaded) == 0

    reloaded.add(np.array([[1.0, 0.0]], dtype=np.float32), ["chunk_a"], ["a.txt"])
    assert len(reloaded) == 1


def test_remove_source_drops_only_matching_chunks():
    store = NumpyVectorStore()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
    store.add(vectors, ["chunk_a", "chunk_b", "chunk_c"], ["a.txt", "b.txt", "a.txt"])

    store.remove_source("a.txt")

    assert len(store) == 1
    assert store._chunks == ["chunk_b"]
    assert store._sources == ["b.txt"]


def test_remove_source_with_no_match_is_a_no_op():
    store = NumpyVectorStore()
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    store.add(vectors, ["chunk_a"], ["a.txt"])

    store.remove_source("does_not_exist.txt")

    assert len(store) == 1


def test_remove_all_chunks_leaves_store_searchable_and_empty():
    store = NumpyVectorStore()
    store.add(np.array([[1.0, 0.0]], dtype=np.float32), ["chunk_a"], ["a.txt"])

    store.remove_source("a.txt")

    assert len(store) == 0
    assert store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1) == []


def test_faiss_remove_source_and_reload_roundtrip(tmp_path):
    # Calling importorskip here, inside the test, only skips this one test
    # when faiss-cpu isn't installed. Calling it at module level instead (as
    # a previous version of this file did) skips the ENTIRE module -- every
    # NumpyVectorStore test above included -- the moment faiss is missing,
    # silently discarding real test coverage rather than just this one test.
    pytest.importorskip("faiss", reason="faiss-cpu not installed")
    store = FaissVectorStore(dim=2)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
    store.add(vectors, ["chunk_a", "chunk_b", "chunk_c"], ["a.txt", "b.txt", "a.txt"])

    store.remove_source("a.txt")
    assert len(store) == 1
    assert store._chunks == ["chunk_b"]

    prefix = str(tmp_path / "index")
    store.save(prefix)
    reloaded = FaissVectorStore(dim=2).load(prefix)
    reloaded.remove_source("b.txt")

    assert len(reloaded) == 0
