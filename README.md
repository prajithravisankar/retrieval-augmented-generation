# RAG Search Engine

A movie search engine built to learn how retrieval augmented generation works. It lets you search a movie dataset using keyword search, semantic search, hybrid search, and image search, then use an LLM to generate answers from the results.

## What it does

- **Keyword search**: classic BM25 style search over movie descriptions
- **Semantic search**: search using text embeddings to match by meaning
- **Hybrid search**: combines keyword and semantic search, with weighted and reciprocal rank fusion methods
- **Reranking**: reorders results using a cross encoder or an LLM
- **Query enhancement**: spell correction, query expansion, and query rewriting to improve search results
- **Multimodal search**: search for movies using an image instead of text
- **RAG (augmented generation)**: ask questions and get an LLM generated answer based on search results, with support for summaries and citations
- **Evaluation**: measures search quality with precision, recall, and F1 scores against a golden dataset

## Project structure

- `cli/` command line tools, one per feature (keyword search, semantic search, hybrid search, multimodal search, augmented generation, evaluation)
- `cli/lib/` the underlying logic for each feature
- `data/` movie dataset, golden test set, and sample files
- `cache/` cached embeddings and indexes so search is fast after the first run

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

You will need a Gemini API key for the LLM features. Add it to a `.env` file:

```
GEMINI_API_KEY=your_key_here
```

## Usage

Each CLI can be run with uv, for example:

```bash
uv run python cli/semantic_search_cli.py search "a movie about space travel"
uv run python cli/hybrid_search_cli.py rrf-search "a movie about space travel"
uv run python cli/augmented_generation_cli.py rag "what movies are about space travel?"
uv run python cli/multimodal_search_cli.py image_search path/to/image.jpg
uv run python cli/evaluation_cli.py --limit 5
```

Run any CLI with `--help` to see its full list of commands.
