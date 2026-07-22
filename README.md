# easy-rag

Build a working Retrieval-Augmented-Generation (RAG) pipeline from a folder
of documents in a few lines of code — or one CLI command. No API key
required to get started.

![easyrag ingest and query in the terminal](docs/docs_terminal_demo.png)

![easyrag demo animation](docs/easyrag-demo.gif)

## What is RAG, in plain terms?

A language model like Claude or GPT only knows what it was trained on — it
has never seen your PDFs, your company's internal docs, or last week's
meeting notes. **Retrieval-Augmented Generation (RAG)** fixes this by:

1. **Retrieving** the most relevant passages from your own documents for a
   given question (using a search technique called embedding similarity,
   explained below), then
2. **Feeding those passages to a language model** as context, so it answers
   using your actual documents instead of guessing from its training data.

This is how most "chat with your documents" or "AI search over your
knowledge base" products work under the hood. Building one from scratch
normally means wiring together four separate pieces yourself: a document
loader, a text chunker, an embedding model, and a vector database. **easy-rag
does all four for you.**

### The three concepts you need to know

- **Chunking** — documents are split into smaller passages (a few hundred
  words each) because embedding models and language models both work better
  on short, focused pieces of text than on a 50-page PDF at once.
- **Embeddings** — each chunk is converted into a list of numbers (a vector)
  that captures its meaning. Chunks about similar topics end up with similar
  vectors, which is what makes semantic search possible.
- **Vector store** — a database of these vectors that can quickly find the
  ones most similar to a question's vector, i.e. the passages most relevant
  to what was asked.

## How this tool works

```
   documents/          Pipeline.ingest()                  Pipeline.query()
  ┌──────────┐    ┌──────────────────────┐            ┌───────────────────┐
  │ text/md  │    │ 1. load              │            │ 1. embed question │
  │ csv      │ →  │ 2. chunk             │  →  index  │ 2. search index   │
  │ pdf/docx │    │ 3. embed             │            │ 3. generate answer│
  │ images   │    │ 4. store in index    │            └───────────────────┘
  └──────────┘    └──────────────────────┘
```

Every stage is swappable. The defaults need **zero setup and zero API
keys** — a built-in hashing-based embedder and an in-memory vector store —
so you can try the whole pipeline the moment you install it. When you're
ready for better answer quality, swap in real embedding models or Claude/
OpenAI generation by changing one argument.

| Stage      | Default (zero setup)        | Upgrade options |
|------------|------------------------------|------------------|
| Embedder   | `hashing` (numpy only)        | `local` (sentence-transformers), `openai` |
| Vector store | `numpy` (in-memory, brute-force) | `faiss` (approximate nearest neighbor, for larger corpora) |
| Generator  | `none` (returns matched passages, no LLM call) | `anthropic` (Claude), `openai` (GPT) |

### Supported file types

| Type | Extensions | Extra needed |
|------|-----------|--------------|
| Plain text / Markdown | `.txt`, `.md`, `.markdown`, `.rst` | none |
| Spreadsheets | `.csv` | none |
| PDF | `.pdf` | `pip install easy-rag[pdf]` |
| Word documents | `.docx` | `pip install easy-rag[docx]` |
| Images (via OCR) | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff` | `pip install easy-rag[ocr]` (also needs the [Tesseract OCR engine](https://github.com/tesseract-ocr/tesseract) installed on your system) |

Mixing file types in the same folder is fine — `ingest()` picks the right
loader per file automatically and skips anything it doesn't recognize.

## Installation

```bash
pip install -e .                 # core: .txt / .md / .csv (numpy only)
pip install -e ".[pdf]"          # + PDF support
pip install -e ".[docx]"         # + Word document support
pip install -e ".[ocr]"          # + image support (also needs Tesseract OCR installed)
pip install -e ".[local]"        # + real local embeddings (sentence-transformers, faiss)
pip install -e ".[anthropic]"    # + Claude generation
pip install -e ".[openai]"       # + OpenAI embeddings/generation
pip install -e ".[all]"          # everything
```

## Quickstart

### As a library

```python
from easy_rag import Pipeline

pipeline = Pipeline()                       # zero-setup defaults
pipeline.ingest("./my_documents")            # load, chunk, embed, index
answer = pipeline.query("What is the refund policy?")
print(answer)
```

Run the included example (uses the sample docs in `examples/docs`):

```bash
python examples/quickstart.py
```

### As a CLI

```bash
easyrag ingest ./my_documents --index .easyrag/index
easyrag query "What is the refund policy?" --index .easyrag/index
```

## Adding your own documents ("training" the pipeline on your data)

There is an important thing to understand up front: **RAG does not train or
fine-tune a model.** The language model itself never changes. "Teaching" the
pipeline about your documents just means indexing them so they can be
searched — a process called **ingestion**, not training. This is why it is
fast (seconds to minutes, not hours) and why you can add new documents at
any time without retraining anything.

Here is the whole process, end to end:

1. **Put your files in a folder.** Any mix of `.txt`, `.md`, `.csv`, `.pdf`,
   `.docx`, or images — subfolders are searched too.
2. **Ingest that folder** — this is the "training" step, and it is one line:
   ```bash
   easyrag ingest ./my_documents --index .easyrag/index
   ```
   Under the hood this loads each file, splits it into chunks, converts each
   chunk into a vector, and saves everything to `.easyrag/index`.
3. **Query it** — also one line:
   ```bash
   easyrag query "your question here" --index .easyrag/index
   ```
4. **Adding more documents later?** Ingest the new folder into the same
   `--index` path; new chunks are appended to the existing index, so you
   never have to reprocess documents you already ingested.

That's the entire loop. There is no separate training run, no GPU required
for the default setup, and no waiting — the example in this repo indexes and
becomes queryable in well under a second.

### Upgrading to real embeddings and Claude

```python
pipeline = Pipeline(embedder="local", vectorstore="faiss", llm="anthropic")
```

```bash
easyrag ingest ./my_documents --embedder local --vectorstore faiss
easyrag query "What is the refund policy?" --llm anthropic
```

`anthropic`/`openai` providers read their API key from the standard
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` environment variables.

## Architecture

```
easy_rag/
  loaders.py       load text/md/csv/pdf/docx/image files into Document objects
  chunking.py       split text into overlapping chunks at paragraph/sentence boundaries
  embeddings.py      Embedder implementations: hashing (default), local, openai
  vectorstore.py      VectorStore implementations: numpy (default), faiss
  llm.py                Generator implementations: none (default), anthropic, openai
  pipeline.py            Pipeline: wires the above together, plus save()/load()
  cli.py                  `easyrag ingest` / `easyrag query`
```

Every provider (embedder, vector store, generator) implements a small,
consistent interface, so adding a new one — a different embedding API, a
managed vector database — means writing one new class, not touching the
pipeline itself.

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

All tests run offline in under a second — no API keys or model downloads
needed, since they exercise the zero-dependency default providers.

## License

MIT — see [LICENSE](LICENSE).
