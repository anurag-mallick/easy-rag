import pytest

from easy_rag.llm import RetrievalOnlyGenerator


def test_retrieval_only_generator_lists_matched_passages():
    generator = RetrievalOnlyGenerator()
    contexts = [(0.9, "Refunds are processed within 5-7 business days.", "policy.txt")]

    answer = generator.generate("How long do refunds take?", contexts)

    assert "policy.txt" in answer
    assert "5-7 business days" in answer


def test_retrieval_only_generator_handles_no_matches():
    generator = RetrievalOnlyGenerator()
    answer = generator.generate("anything", [])
    assert "No relevant context" in answer


def test_openai_generator_base_url_points_at_a_local_server(monkeypatch):
    # Confirms base_url lets OpenAIGenerator talk to a locally running
    # llama-server instead of the real OpenAI API, without a real network
    # call -- llama-server's /v1/chat/completions is wire-compatible.
    pytest.importorskip("openai", reason="openai package not installed")
    from easy_rag.llm import OpenAIGenerator

    monkeypatch.setenv("OPENAI_API_KEY", "unused-placeholder")
    generator = OpenAIGenerator(model="local-model", base_url="http://localhost:8080/v1")
    assert str(generator._client.base_url).startswith("http://localhost:8080/v1")


def test_llamacpp_generator_requires_install_with_helpful_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "llama_cpp":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from easy_rag.llm import LlamaCppGenerator

    with pytest.raises(ImportError, match=r"easy-rag\[llamacpp\]"):
        LlamaCppGenerator(model_path="does_not_matter.gguf")


def test_llamacpp_generator_wiring_with_a_stubbed_model(monkeypatch, tmp_path):
    # As with the embedder, stub llama_cpp.Llama itself so this stays a
    # fast, offline unit test instead of downloading/running a real model.
    llama_cpp = pytest.importorskip("llama_cpp", reason="llama-cpp-python not installed")

    class FakeLlama:
        def __init__(self, model_path=None, n_ctx=None, verbose=None, **kw):
            self.model_path = model_path

        def create_chat_completion(self, messages, max_tokens):
            prompt = messages[0]["content"]
            assert "What is the refund window?" in prompt
            assert "30 days" in prompt  # the context was interpolated in
            return {"choices": [{"message": {"content": "The refund window is 30 days."}}]}

    monkeypatch.setattr(llama_cpp, "Llama", FakeLlama)

    from easy_rag.llm import LlamaCppGenerator

    generator = LlamaCppGenerator(model_path=str(tmp_path / "fake.gguf"))
    answer = generator.generate(
        "What is the refund window?",
        [(0.9, "The refund window is 30 days.", "policy.txt")],
    )
    assert answer == "The refund window is 30 days."
