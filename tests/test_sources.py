import os

import pytest

from easy_rag.sources import add_source, load_sources, remove_source


def test_add_source_registers_absolute_path(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    index_path = str(tmp_path / "index")

    result = add_source(index_path, str(folder))

    assert result == [os.path.abspath(str(folder))]
    assert load_sources(index_path) == result


def test_add_source_is_idempotent(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    index_path = str(tmp_path / "index")

    add_source(index_path, str(folder))
    result = add_source(index_path, str(folder))

    assert result == [os.path.abspath(str(folder))]


def test_add_nonexistent_folder_raises(tmp_path):
    index_path = str(tmp_path / "index")
    with pytest.raises(NotADirectoryError):
        add_source(index_path, str(tmp_path / "does_not_exist"))


def test_remove_source(tmp_path):
    folder_a = tmp_path / "a"
    folder_b = tmp_path / "b"
    folder_a.mkdir()
    folder_b.mkdir()
    index_path = str(tmp_path / "index")

    add_source(index_path, str(folder_a))
    add_source(index_path, str(folder_b))
    result = remove_source(index_path, str(folder_a))

    assert result == [os.path.abspath(str(folder_b))]


def test_load_sources_for_unknown_index_returns_empty_list(tmp_path):
    assert load_sources(str(tmp_path / "never_saved")) == []
