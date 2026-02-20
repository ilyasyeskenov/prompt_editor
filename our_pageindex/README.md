# PageIndex (standalone module)

Build document trees from PDFs and chat over them **without the PageIndex API**. Trees can be saved to Supabase and reused.

## Features

- **Tree creation**: PDF → ToC detection → hierarchical tree (node_id, title, text/summary, nodes).
- **Chat**: Tree-based RAG (LLM selects relevant nodes → context → answer). No external API.
- **Storage**: Optional Supabase backend to save/load trees by `doc_id`.

## Setup

1. **Env** (for tree building and chat):
   - `OPENAI_API_KEY` or `CHATGPT_API_KEY` for LLM calls.
   - For Supabase: `SUPABASE_URL` and `SUPABASE_ANON_KEY`.

2. **Supabase table** (optional): run `pageindex_trees_table.sql` in the SQL Editor to create `pageindex_trees`.

3. **Dependencies**: `openai`, `tiktoken`, `PyPDF2`, `pymupdf`, `pyyaml`, `python-dotenv`. For storage: `supabase`.

## Usage

```python
from our_pageindex import (
    build_tree_from_pdf,
    get_tree,
    chat,
    get_tree_then_chat,
    SupabaseTreeStorage,
    InMemoryTreeStorage,
)

# Build tree from PDF and save to Supabase
storage = SupabaseTreeStorage()
tree = build_tree_from_pdf("doc.pdf", doc_id="my-doc", storage=storage)

# Load tree and get an answer
tree = get_tree("my-doc", storage=storage)
answer = chat(ai_client, "What is the deadline?", tree=tree)

# One-liner: load from storage and chat
answer = get_tree_then_chat(
    ai_client,
    "What is the deadline?",
    get_tree=lambda: get_tree("my-doc", storage=storage),
)
```

`ai_client` must be OpenAI-compatible: `.client.chat.completions.create(model=..., messages=..., temperature=0)` and `.model` (e.g. the app’s `AIClient`).

## Config

Defaults live in `config.yaml` (model, ToC page limit, node size, whether to add text/summaries/node_id). Override via kwargs to `build_tree_from_pdf`.

## Reuse

This folder is a **standalone module**: no dependency on the rest of the app. Use it from the tender checker, another app, or scripts by adding the repo root to `PYTHONPATH` or installing as a package.
