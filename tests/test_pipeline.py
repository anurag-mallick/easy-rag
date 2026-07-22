from easy_rag.pipeline import Pipeline


def _sample_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "cats.txt").write_text(
        "Cats are small carnivorous mammals. They are often kept as pets and "
        "are known for being independent and low-maintenance compared to dogs.",
        encoding="utf-8",
    )
    (d / "dogs.txt").write_text(
        "Dogs are domesticated mammals known for loyalty. They require regular "
        "walks and are often trained to assist people with disabilities.",
        encoding="utf-8",
    )
    return d


def test_ingest_and_retrieve_finds_relevant_document(tmp_path):
    pipeline = Pipeline(embedder="hashing", vectorstore="numpy", llm="none")
    d = _sample_dir(tmp_path)

    n_chunks = pipeline.ingest(str(d))
    assert n_chunks > 0

    results = pipeline.retrieve("Which animal is good at assisting people?", top_k=1)
    assert len(results) == 1
    assert "dog" in results[0][1].lower()


def test_query_without_llm_returns_context():
    pipeline = Pipeline(embedder="hashing", vectorstore="numpy", llm="none")
    pipeline.vectorstore.add(
        pipeline.embedder.embed(["The refund window is 30 days."]),
        ["The refund window is 30 days."],
        ["policy.txt"],
    )

    answer = pipeline.query("What is the refund window?")
    assert "30 days" in answer


def test_save_and_load_preserves_config_and_data(tmp_path):
    pipeline = Pipeline(embedder="hashing", vectorstore="numpy", llm="none")
    d = _sample_dir(tmp_path)
    pipeline.ingest(str(d))

    # Nested, not-yet-existing directory -- save() must create it, not assume
    # it exists (this is the real-world case for a fresh CLI ingest).
    index_path = str(tmp_path / "nested" / "does_not_exist_yet" / "index")
    pipeline.save(index_path)

    reloaded = Pipeline.load(index_path, llm="none")
    results = reloaded.retrieve("independent pet", top_k=1)
    assert "cat" in results[0][1].lower()
