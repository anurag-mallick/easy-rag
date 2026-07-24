import pytest

from easy_rag.chunking import split_text
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


def test_csv_rows_are_separated_by_blank_lines_for_correct_chunking(tmp_path):
    # chunking.py only treats a blank line as a paragraph break. Joining CSV
    # rows with a single newline collapses the whole file into one giant
    # paragraph; for punctuation-free "key: value" rows (no '.', '!', '?')
    # the sentence-splitter fallback finds no boundary either, so an
    # oversized CSV gets hard-wrapped by raw character count -- slicing
    # words in half wherever a chunk boundary happens to land mid-row.
    csv_path = tmp_path / "data.csv"
    rows = ["name,role,dept"] + [f"Person{i},Engineer,Team{i % 5}" for i in range(80)]
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    docs = load_documents(str(csv_path))
    text = docs[0].text
    assert "\n\n" in text  # rows are blank-line-separated paragraphs

    chunks = split_text(text, chunk_size=200, overlap=40)
    full_words = set(text.replace(",", " ").replace(":", " ").split())
    chunk_words = set(" ".join(chunks).replace(",", " ").replace(":", " ").split())
    mangled = chunk_words - full_words
    assert not mangled, f"chunking introduced truncated/mangled words: {mangled}"


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


def test_invalid_pdf_backend_raises():
    with pytest.raises(ValueError, match="Unknown pdf_backend"):
        load_documents(".", pdf_backend="not-a-real-backend")


def test_opendataloader_backend_requires_install_with_helpful_message(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "opendataloader_pdf":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    (tmp_path / "report.pdf").write_bytes(b"not a real pdf, just bytes")

    with pytest.raises(ImportError, match=r"easy-rag\[opendataloader\]"):
        load_documents(str(tmp_path), pdf_backend="opendataloader")


def _make_pdf(path, text):
    fitz = pytest.importorskip("fitz", reason="pymupdf not installed (only needed to build test PDFs)")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=14)
    doc.save(str(path))


def test_opendataloader_backend_extracts_real_pdf_text(tmp_path):
    pytest.importorskip("opendataloader_pdf", reason="opendataloader-pdf not installed")
    import shutil

    if not shutil.which("java"):
        pytest.skip("Java not installed (opendataloader-pdf requires Java 11+)")

    _make_pdf(tmp_path / "policy.pdf", "The refund window is 30 days from purchase.")

    docs = load_documents(str(tmp_path), pdf_backend="opendataloader")

    assert len(docs) == 1
    assert "30 days" in docs[0].text


def test_opendataloader_backend_skips_corrupt_pdf_without_losing_good_ones(tmp_path):
    pytest.importorskip("opendataloader_pdf", reason="opendataloader-pdf not installed")
    import shutil

    if not shutil.which("java"):
        pytest.skip("Java not installed (opendataloader-pdf requires Java 11+)")

    _make_pdf(tmp_path / "policy.pdf", "The refund window is 30 days from purchase.")
    (tmp_path / "corrupt.pdf").write_bytes(b"not a real pdf, just garbage bytes")

    docs = load_documents(str(tmp_path), pdf_backend="opendataloader")

    assert len(docs) == 1
    assert "30 days" in docs[0].text


def test_opendataloader_backend_handles_same_basename_in_different_folders(tmp_path):
    # opendataloader-pdf writes every batched file's output into one shared
    # directory named only after the input basename -- two different PDFs
    # sharing a basename would silently overwrite each other's output if
    # batched together naively. Found by testing the real tool directly.
    pytest.importorskip("opendataloader_pdf", reason="opendataloader-pdf not installed")
    import shutil

    if not shutil.which("java"):
        pytest.skip("Java not installed (opendataloader-pdf requires Java 11+)")

    (tmp_path / "dirA").mkdir()
    (tmp_path / "dirB").mkdir()
    _make_pdf(tmp_path / "dirA" / "report.pdf", "Document A content about apples.")
    _make_pdf(tmp_path / "dirB" / "report.pdf", "Document B content about bananas.")

    docs = load_documents(str(tmp_path), pdf_backend="opendataloader")

    assert len(docs) == 2
    texts = {d.text for d in docs}
    assert any("apples" in t for t in texts)
    assert any("bananas" in t for t in texts)


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
