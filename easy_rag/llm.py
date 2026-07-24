"""Generation providers for turning retrieved context into an answer.

The default requires no API key at all: it just returns the retrieved
context so the pipeline is queryable out of the box. Swap in Claude or
OpenAI once you have an API key for a real synthesized answer.
"""

DEFAULT_PROMPT_TEMPLATE = """Answer the question using only the context below. \
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

# Returned by every generator (instead of making an API call over an empty
# or near-empty prompt) when retrieve() found nothing -- either the index
# is empty, or a min_score threshold filtered everything out because
# nothing ingested was actually relevant to the question. Skipping the call
# avoids paying for/waiting on a request that would otherwise ask the model
# to answer from no context at all.
NO_CONTEXT_MESSAGE = "No relevant context was found for this question."


class Generator:
    name = "base"

    def generate(self, question, contexts):
        raise NotImplementedError


class RetrievalOnlyGenerator(Generator):
    """No LLM call: returns the retrieved chunks themselves. Always works,
    needs no API key, and is useful for testing retrieval quality in
    isolation before paying for generation."""

    name = "none"

    def generate(self, question, contexts):
        if not contexts:
            return NO_CONTEXT_MESSAGE
        lines = [f"Top matching passages for: {question!r}\n"]
        for i, (score, text, source) in enumerate(contexts, 1):
            lines.append(f"[{i}] (score={score:.3f}, source={source})\n{text}\n")
        return "\n".join(lines)


class AnthropicGenerator(Generator):
    """Generation via the Claude API. Install with: pip install easy-rag[anthropic]
    Requires the ANTHROPIC_API_KEY environment variable.
    """

    name = "anthropic"

    def __init__(self, model="claude-sonnet-5", prompt_template=DEFAULT_PROMPT_TEMPLATE):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' generator requires the anthropic package. "
                "Install it with: pip install easy-rag[anthropic]"
            ) from e
        self._client = anthropic.Anthropic()
        self._model = model
        self._template = prompt_template

    def generate(self, question, contexts):
        if not contexts:
            return NO_CONTEXT_MESSAGE
        context_text = "\n\n".join(text for _score, text, _source in contexts)
        prompt = self._template.format(context=context_text, question=question)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIGenerator(Generator):
    """Generation via the OpenAI API, or any OpenAI-wire-compatible server.
    Install with: pip install easy-rag[openai]
    Requires the OPENAI_API_KEY environment variable (any non-empty string
    if base_url points at a local server that doesn't check it).

    Passing base_url lets this same class talk to a locally running
    `llama-server` (from llama.cpp) instead of the real OpenAI API -- its
    /v1/chat/completions endpoint is wire-compatible with OpenAI's. This is
    often the path of least resistance on Windows, since llama-server is a
    prebuilt executable with no Python package to compile -- see the
    'llamacpp' generator below for the alternative that runs the model
    in-process via the llama-cpp-python bindings instead.
    """

    name = "openai"

    def __init__(self, model="gpt-4o-mini", prompt_template=DEFAULT_PROMPT_TEMPLATE, base_url=None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' generator requires the openai package. "
                "Install it with: pip install easy-rag[openai]"
            ) from e
        self._client = OpenAI(base_url=base_url) if base_url else OpenAI()
        self._model = model
        self._template = prompt_template

    def generate(self, question, contexts):
        if not contexts:
            return NO_CONTEXT_MESSAGE
        context_text = "\n\n".join(text for _score, text, _source in contexts)
        prompt = self._template.format(context=context_text, question=question)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class GeminiGenerator(Generator):
    """Generation via Google's Gemini API. Install with: pip install easy-rag[gemini]
    Requires the GEMINI_API_KEY environment variable.
    """

    name = "gemini"

    def __init__(self, model="gemini-2.5-flash", prompt_template=DEFAULT_PROMPT_TEMPLATE):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "The 'gemini' generator requires the google-genai package. "
                "Install it with: pip install easy-rag[gemini]"
            ) from e
        self._client = genai.Client()
        self._model = model
        self._template = prompt_template

    def generate(self, question, contexts):
        if not contexts:
            return NO_CONTEXT_MESSAGE
        context_text = "\n\n".join(text for _score, text, _source in contexts)
        prompt = self._template.format(context=context_text, question=question)
        response = self._client.models.generate_content(model=self._model, contents=prompt)
        return response.text


class LlamaCppGenerator(Generator):
    """Fully local generation via llama.cpp, in-process -- no server to run,
    no API key, no internet needed after the model is downloaded once.
    Install with: pip install easy-rag[llamacpp]

    With no model_path given, downloads (and caches) a small instruction-
    tuned GGUF model from Hugging Face Hub the first time it's used -- see
    the README for the exact model and its size before relying on this in
    an environment with restricted bandwidth or disk space. Pass model_path
    to a local .gguf file instead to use your own model and skip the
    download entirely.
    """

    name = "llamacpp"

    DEFAULT_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    DEFAULT_FILENAME = "*q4_k_m.gguf"

    def __init__(
        self,
        model_path=None,
        repo_id=None,
        filename=None,
        n_ctx=4096,
        max_tokens=512,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        **llama_kwargs,
    ):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "The 'llamacpp' generator requires llama-cpp-python. "
                "Install it with: pip install easy-rag[llamacpp]"
            ) from e
        if model_path:
            self._llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False, **llama_kwargs)
        else:
            self._llm = Llama.from_pretrained(
                repo_id=repo_id or self.DEFAULT_REPO,
                filename=filename or self.DEFAULT_FILENAME,
                n_ctx=n_ctx,
                verbose=False,
                **llama_kwargs,
            )
        self._max_tokens = max_tokens
        self._template = prompt_template

    def generate(self, question, contexts):
        if not contexts:
            return NO_CONTEXT_MESSAGE
        context_text = "\n\n".join(text for _score, text, _source in contexts)
        prompt = self._template.format(context=context_text, question=question)
        response = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
        )
        return response["choices"][0]["message"]["content"]


_REGISTRY = {
    "none": RetrievalOnlyGenerator,
    "anthropic": AnthropicGenerator,
    "openai": OpenAIGenerator,
    "gemini": GeminiGenerator,
    "llamacpp": LlamaCppGenerator,
}


def get_generator(name="none", **kwargs):
    """Look up a generator by name: 'none' (default, zero deps/keys),
    'anthropic' (Claude API), 'openai' (OpenAI API, or any OpenAI-compatible
    server via base_url), 'gemini' (Google's Gemini API), or 'llamacpp'
    (fully local GGUF model, no server)."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown generator '{name}'. Choose from: {sorted(_REGISTRY)}")
    return cls(**kwargs)
