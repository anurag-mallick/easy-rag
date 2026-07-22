import os

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


def test_ingesting_more_documents_into_a_loaded_index_appends(tmp_path):
    d = _sample_dir(tmp_path)
    index_path = str(tmp_path / "index")

    pipeline = Pipeline(embedder="hashing", vectorstore="numpy", llm="none")
    pipeline.ingest(str(d))
    pipeline.save(index_path)

    # Simulate a second `easyrag ingest` run against the same --index: load
    # the existing index and ingest more documents into it.
    extra = tmp_path / "more_docs"
    extra.mkdir()
    (extra / "birds.txt").write_text(
        "Birds are warm-blooded egg-laying vertebrates known for flight.",
        encoding="utf-8",
    )

    reloaded = Pipeline.load(index_path, llm="none")
    before = len(reloaded.vectorstore)
    reloaded.ingest(str(extra))
    reloaded.save(index_path)

    final = Pipeline.load(index_path, llm="none")
    assert len(final.vectorstore) > before
    # The hashing embedder scores on literal token overlap, not true semantic
    # similarity, so query with words that actually appear in the target doc.
    results = final.retrieve("warm-blooded egg-laying vertebrates known for flight", top_k=1)
    assert "bird" in results[0][1].lower()


def test_ingest_twice_on_unchanged_folder_adds_nothing_new(tmp_path):
    pipeline = Pipeline(embedder="hashing", vectorstore="numpy", llm="none")
    d = _sample_dir(tmp_path)

    first = pipeline.ingest(str(d))
    assert first > 0

    second = pipeline.ingest(str(d))
    assert second == 0
    # No duplicate chunks should have been added to the store either.
    assert len(pipeline.vectorstore) == first


def test_dropping_a_new_file_into_an_already_ingested_folder_adds_only_that_file(tmp_path):
    pipeline = Pipeline(embedder="hashing", vectorstore="numpy", llm="none")
    d = _sample_dir(tmp_path)
    pipeline.ingest(str(d))
    before = len(pipeline.vectorstore)

    (d / "fish.txt").write_text(
        "Fish are aquatic vertebrates that breathe through gills.", encoding="utf-8"
    )
    added = pipeline.ingest(str(d))

    assert added > 0
    assert len(pipeline.vectorstore) == before + added
    results = pipeline.retrieve("aquatic vertebrates that breathe through gills", top_k=1)
    assert "fish" in results[0][1].lower()


def test_modifying_an_ingested_file_replaces_its_chunks_not_duplicates_them(tmp_path):
    pipeline = Pipeline(embedder="hashing", vectorstore="numpy", llm="none")
    d = _sample_dir(tmp_path)
    pipeline.ingest(str(d))
    before_total = len(pipeline.vectorstore)
    before_cats_chunks = sum(1 for s in pipeline.vectorstore._sources if "cats.txt" in s)

    cats_file = d / "cats.txt"
    cats_file.write_text(
        "Cats communicate through meowing, purring, and body language.",
        encoding="utf-8",
    )
    # Ensure the mtime actually changes on filesystems with coarse timestamp
    # resolution, so the manifest fingerprint is guaranteed to differ.
    new_time = os.path.getmtime(cats_file) + 5
    os.utime(cats_file, (new_time, new_time))

    added = pipeline.ingest(str(d))

    assert added > 0
    results = pipeline.retrieve("meowing purring body language", top_k=1)
    assert "meowing" in results[0][1].lower()
    # The stale "independent and low-maintenance" sentence must be gone --
    # proof the old chunks were replaced, not just added alongside the new.
    assert not any("low-maintenance" in c.lower() for c in pipeline.vectorstore._chunks)
    # Total = everything before, minus cats.txt's old chunks, plus its new ones.
    assert len(pipeline.vectorstore) == before_total - before_cats_chunks + added
