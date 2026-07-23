import pytest

from easy_rag.loaders import load_documents


def test_loads_txt_and_md(tmp_path):
    (tmp_path / "a.txt").write_text("hello from txt", encoding="utf-8")
    (tmp_path / "b.md").write_text("# hello from md", encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert len(docs) == 2
    texts = {d.text for d in docs}
    assert "hello from txt" in texts
    assert "# hello from md" in texts


def test_loads_csv_as_readable_rows(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,role\nAva,Engineer\nBen,Designer\n", encoding="utf-8")

    docs = load_documents(str(csv_path))

    assert len(docs) == 1
    assert "name: Ava" in docs[0].text
    assert "role: Engineer" in docs[0].text
    assert "name: Ben" in docs[0].text


def test_unsupported_extensions_are_skipped(tmp_path):
    (tmp_path / "notes.txt").write_text("kept", encoding="utf-8")
    (tmp_path / "archive.zip").write_bytes(b"not a real zip, just bytes")

    docs = load_documents(str(tmp_path))

    assert len(docs) == 1
    assert docs[0].text == "kept"


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_documents("this/path/does/not/exist")


def test_corrupt_file_is_skipped_without_losing_other_valid_files(tmp_path):
    (tmp_path / "good.txt").write_text("this is a perfectly fine document", encoding="utf-8")
    # pypdf will raise on this -- it's not a real PDF, just garbage bytes.
    (tmp_path / "corrupt.pdf").write_bytes(b"not a real pdf, just garbage bytes")

    docs = load_documents(str(tmp_path))

    assert len(docs) == 1
    assert docs[0].text == "this is a perfectly fine document"


def test_docx_without_dependency_raises_helpful_error(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "docx":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    docx_path = tmp_path / "report.docx"
    docx_path.write_bytes(b"not a real docx, just bytes")

    with pytest.raises(ImportError, match="easy-rag\\[docx\\]"):
        load_documents(str(docx_path))
