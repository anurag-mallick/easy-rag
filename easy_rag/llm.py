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
            return "No relevant context was found for this question."
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
        context_text = "\n\n".join(text for _score, text, _source in contexts)
        prompt = self._template.format(context=context_text, question=question)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIGenerator(Generator):
    """Generation via the OpenAI API. Install with: pip install easy-rag[openai]
    Requires the OPENAI_API_KEY environment variable.
    """

    name = "openai"

    def __init__(self, model="gpt-4o-mini", prompt_template=DEFAULT_PROMPT_TEMPLATE):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' generator requires the openai package. "
                "Install it with: pip install easy-rag[openai]"
            ) from e
        self._client = OpenAI()
        self._model = model
        self._template = prompt_template

    def generate(self, question, contexts):
        context_text = "\n\n".join(text for _score, text, _source in contexts)
        prompt = self._template.format(context=context_text, question=question)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


_REGISTRY = {
    "none": RetrievalOnlyGenerator,
    "anthropic": AnthropicGenerator,
    "openai": OpenAIGenerator,
}


def get_generator(name="none", **kwargs):
    """Look up a generator by name: 'none' (default, zero deps/keys),
    'anthropic' (Claude API), or 'openai' (OpenAI API)."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown generator '{name}'. Choose from: {sorted(_REGISTRY)}")
    return cls(**kwargs)
