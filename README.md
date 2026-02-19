# Text → Knowledge Graph → Semantic Search

Build a knowledge graph from unstructured text using **CogDB** and **OpenAI**, then query it with graph traversal, semantic search, and hybrid queries.

## Pipeline

```
Source Text
    ↓  (OpenAI gpt-5)
Entity Extraction        → typed entities (CelestialBody, Moon, Mission, …)
    ↓  (OpenAI gpt-5)
Relationship Extraction  → typed triples (Europa MOON_OF Jupiter, …)
    ↓
Entity Resolution        → normalize aliases, deduplicate
    ↓  (OpenAI text-embedding-3-small)
Embedding Generation     → vector per entity
    ↓
CogDB Graph + Vectors    → triples + embeddings loaded into CogDB
    ↓
Queries                  → traversal, semantic, and hybrid
```

## Quick Start

```bash
# 1. Set your OpenAI API key
echo 'OPENAI_API_KEY=sk-...' > .env

# 2. Run (creates venv, installs deps, runs the demo)
./run.sh
```

On the first run, the pipeline calls OpenAI to extract entities and relationships from `planetary-habitability.txt` and caches the results in `kg_data.json`. Subsequent runs skip extraction and go straight to graph construction and queries.

Use `./run.sh --clean` to wipe the venv and database and start fresh.

## What the Queries Demonstrate

| Section | Technique | Example |
|---|---|---|
| **A. Graph Traversal** | Structural hops | Europa → features, missions targeting habitable bodies |
| **B. Semantic Search** | Embedding k-nearest | Entities closest to "signs of life" or "ocean world" |
| **C. Hybrid** | Traversal + semantic | BFS from Europa → filter by "water and ice" relevance |

## Requirements

- Python 3.10+
- OpenAI API key
- See [requirements.txt](requirements.txt) for Python packages
