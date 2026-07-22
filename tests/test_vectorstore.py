import numpy as np

from easy_rag.vectorstore import NumpyVectorStore


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
