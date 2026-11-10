# Graph RAG Navigator

[![CI](https://github.com/txltedxgod/graph-rag-navigator/actions/workflows/ci.yml/badge.svg)](https://github.com/txltedxgod/graph-rag-navigator/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)

Knowledge Graph-enhanced RAG engine in Python 3.11+ combining entity linking, diamond-safe multi-hop traversal, PageRank centrality, shortest-path reasoning, and FastAPI REST endpoints.

## Features
- **Diamond & Cycle-Safe Traversal** (`multi_hop_search`): Path-scoped cycle detection preserves parallel DAG branches without skipping valid multi-hop routes.
- **GraphRAG Subgraph Querying** (`subgraph_query`): Links query entities via lexical + PageRank signals, expands bidirectional k-hop neighborhoods, and extracts connecting shortest paths.
- **REST API**: `/health`, `POST /api/v1/triples`, `POST /api/v1/traverse`, `POST /api/v1/query`, `GET /api/v1/path`, `GET /api/v1/stats`.

## Testing
```bash
ruff check .
pytest -v tests/
```
