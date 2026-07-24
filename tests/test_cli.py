from easy_rag.cli import main


def _make_docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "policy.txt").write_text("The refund window is 30 days from purchase.", encoding="utf-8")
    return d


def test_ingest_then_query_roundtrip(tmp_path, capsys):
    docs = _make_docs(tmp_path)
    index = str(tmp_path / "index")

    main(["ingest", str(docs), "--index", index])
    out = capsys.readouterr().out
    assert "Added 1 chunk" in out

    main(["query", "How long is the refund window?", "--index", index])
    out = capsys.readouterr().out
    assert "30 days" in out


def test_query_with_min_score_filters_out_irrelevant_matches(tmp_path, capsys):
    docs = _make_docs(tmp_path)
    index = str(tmp_path / "index")

    main(["ingest", str(docs), "--index", index])
    capsys.readouterr()

    main(["query", "quantum computing stock market forecast", "--index", index, "--min-score", "1.0"])
    out = capsys.readouterr().out
    assert out.strip() == "No relevant context was found for this question."


def test_ingest_twice_without_force_adds_nothing_new(tmp_path, capsys):
    docs = _make_docs(tmp_path)
    index = str(tmp_path / "index")

    main(["ingest", str(docs), "--index", index])
    capsys.readouterr()

    main(["ingest", str(docs), "--index", index])
    out = capsys.readouterr().out
    assert "Added 0 chunk" in out


def test_ingest_with_force_reingests_unchanged_files(tmp_path, capsys):
    docs = _make_docs(tmp_path)
    index = str(tmp_path / "index")

    main(["ingest", str(docs), "--index", index])
    capsys.readouterr()

    main(["ingest", str(docs), "--index", index, "--force"])
    out = capsys.readouterr().out
    assert "Re-ingested 1 chunk" in out


def test_ingest_with_no_path_and_no_sources_errors_helpfully(tmp_path, capsys):
    index = str(tmp_path / "index")

    try:
        main(["ingest", "--index", index])
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code == 1
    err = capsys.readouterr().err
    assert "No path given and no source folders registered" in err


def test_sources_add_list_remove_roundtrip(tmp_path, capsys):
    docs = _make_docs(tmp_path)
    index = str(tmp_path / "index")

    main(["sources", "add", str(docs), "--index", index])
    capsys.readouterr()

    main(["sources", "list", "--index", index])
    out = capsys.readouterr().out
    assert str(docs.resolve()) in out

    main(["sources", "remove", str(docs), "--index", index])
    capsys.readouterr()

    main(["sources", "list", "--index", index])
    out = capsys.readouterr().out
    assert "No source folders registered" in out


def test_ingest_with_no_path_uses_registered_sources(tmp_path, capsys):
    docs = _make_docs(tmp_path)
    index = str(tmp_path / "index")

    main(["sources", "add", str(docs), "--index", index])
    capsys.readouterr()

    main(["ingest", "--index", index])
    out = capsys.readouterr().out
    assert "Added 1 chunk(s) from 1 folder(s)" in out
