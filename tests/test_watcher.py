import pytest

from easy_rag.pipeline import Pipeline
from easy_rag.watcher import scan_once, watch


def test_scan_once_ingests_all_registered_folders(tmp_path):
    pipeline = Pipeline(embedder="hashing", vectorstore="numpy", llm="none")

    folder_a = tmp_path / "a"
    folder_b = tmp_path / "b"
    folder_a.mkdir()
    folder_b.mkdir()
    (folder_a / "one.txt").write_text("Content from folder A.", encoding="utf-8")
    (folder_b / "two.txt").write_text("Content from folder B.", encoding="utf-8")

    added = scan_once(pipeline, [str(folder_a), str(folder_b)])

    assert added == 2
    assert len(pipeline.vectorstore) == 2


def test_scan_once_only_picks_up_a_dropped_file_on_the_next_scan(tmp_path):
    pipeline = Pipeline(embedder="hashing", vectorstore="numpy", llm="none")
    folder = tmp_path / "watched"
    folder.mkdir()
    (folder / "existing.txt").write_text("Already here.", encoding="utf-8")

    first = scan_once(pipeline, [str(folder)])
    assert first == 1

    second = scan_once(pipeline, [str(folder)])
    assert second == 0  # nothing new yet

    # Simulate dropping a new file into the watched folder.
    (folder / "dropped.txt").write_text("Just arrived.", encoding="utf-8")
    third = scan_once(pipeline, [str(folder)])

    assert third == 1
    assert len(pipeline.vectorstore) == 2


def test_scan_once_reports_a_bad_folder_but_keeps_scanning_the_rest(tmp_path):
    pipeline = Pipeline(embedder="hashing", vectorstore="numpy", llm="none")
    good_folder = tmp_path / "good"
    good_folder.mkdir()
    (good_folder / "one.txt").write_text("Good content.", encoding="utf-8")
    missing_folder = str(tmp_path / "does_not_exist")

    errors = []
    added = scan_once(
        pipeline,
        [missing_folder, str(good_folder)],
        on_error=lambda folder, e: errors.append((folder, e)),
    )

    assert added == 1  # the good folder was still ingested
    assert len(errors) == 1
    assert errors[0][0] == missing_folder


def test_watch_survives_a_scan_error_and_keeps_running(tmp_path, monkeypatch):
    pipeline = Pipeline(embedder="hashing", vectorstore="numpy", llm="none")
    index_path = str(tmp_path / "index")
    missing_folder = str(tmp_path / "does_not_exist")

    call_count = {"n": 0}

    def fake_sleep(_seconds):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr("easy_rag.watcher.time.sleep", fake_sleep)

    errors = []
    watch(
        pipeline,
        [missing_folder],
        index_path,
        interval=0,
        on_error=lambda folder, e: errors.append(folder),
    )

    # Two scans ran (each hitting the same error) before the simulated
    # Ctrl+C -- proof one bad folder didn't kill the loop after its first
    # failure.
    assert errors == [missing_folder, missing_folder]


def test_watch_with_no_sources_raises_before_looping(tmp_path):
    pipeline = Pipeline(embedder="hashing", vectorstore="numpy", llm="none")
    index_path = str(tmp_path / "index")

    with pytest.raises(ValueError, match="No source folders"):
        watch(pipeline, [], index_path)


def test_watch_stops_cleanly_on_keyboard_interrupt(tmp_path, monkeypatch):
    pipeline = Pipeline(embedder="hashing", vectorstore="numpy", llm="none")
    folder = tmp_path / "watched"
    folder.mkdir()
    (folder / "a.txt").write_text("hello", encoding="utf-8")
    index_path = str(tmp_path / "index")

    def fake_sleep(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr("easy_rag.watcher.time.sleep", fake_sleep)

    scans = []
    watch(pipeline, [str(folder)], index_path, interval=0, on_scan=scans.append)

    assert scans == [1]  # exactly one scan ran before the (simulated) interrupt
