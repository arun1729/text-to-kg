"""
CogDB Demo: Text → Knowledge Graph → Semantic Search
=====================================================
Build a knowledge graph from text using OpenAI for entity extraction
and embeddings, then query it with CogDB's graph traversal + vector search.

Usage:
    export OPENAI_API_KEY="sk-..."
    python cogdb_text_to_kg_demo.py

Requirements:
    pip install cogdb openai
"""

import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from cog.torque import Graph

load_dotenv()

# ---------------------------------------------------------------------------
# 1. SAMPLE TEXT — a few paragraphs covering different domains
# ---------------------------------------------------------------------------
SAMPLE_TEXTS = [
    """
    Albert Einstein developed the theory of General Relativity in 1915 while
    working at the University of Berlin. The theory fundamentally changed our
    understanding of gravity, describing it as the curvature of spacetime
    caused by mass and energy. Einstein received the Nobel Prize in Physics
    in 1921, though it was awarded for his work on the photoelectric effect,
    not relativity. He later moved to Princeton University in the United States
    where he spent the rest of his career.
    """,
    """
    Python was created by Guido van Rossum and first released in 1991.
    It was influenced by the ABC programming language. Python emphasizes
    code readability and supports multiple programming paradigms including
    object-oriented, procedural, and functional programming. The language
    is maintained by the Python Software Foundation. Popular Python frameworks
    include Django for web development, NumPy for scientific computing,
    and PyTorch for machine learning.
    """,
    """
    The Amazon Rainforest spans across nine countries in South America,
    with Brazil containing about 60 percent of it. The forest produces
    roughly 20 percent of the world's oxygen and is home to approximately
    10 percent of all species on Earth. Deforestation driven by agriculture
    and logging threatens the ecosystem. The Amazon River, which flows
    through the forest, is the largest river by water volume in the world
    and feeds into the Atlantic Ocean.
    """,
    """
    GraphRAG is an approach to retrieval-augmented generation that uses
    knowledge graphs instead of pure vector similarity. Microsoft Research
    published a paper on GraphRAG in 2024 showing improved performance on
    global queries compared to traditional vector-only RAG. The approach
    involves building a knowledge graph from documents, creating community
    summaries, and using graph traversal to gather context for LLM prompts.
    LlamaIndex and LangChain both support knowledge graph integration.
    """,
]

# ---------------------------------------------------------------------------
# 2. EXTRACT TRIPLES — Use OpenAI to pull (subject, predicate, object) from text
# ---------------------------------------------------------------------------
def extract_triples(client, text: str) -> list[tuple[str, str, str]]:
    """Use GPT to extract knowledge graph triples from a text passage."""
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledge graph extraction engine. "
                    "Extract factual triples from the text as JSON. "
                    "Return ONLY a JSON array of objects with keys: "
                    '"subject", "predicate", "object". '
                    "Use short, normalized entity names (e.g. 'Einstein' not "
                    "'Albert Einstein developed'). Use UPPER_SNAKE_CASE for "
                    "predicates (e.g. CREATED_BY, LOCATED_IN, PUBLISHED_IN). "
                    "Extract 8-15 triples per passage. No commentary."
                ),
            },
            {"role": "user", "content": text.strip()},
        ],
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    triples = json.loads(raw)
    # Normalize to lowercase for consistent querying
    return [(str(t["subject"]).lower(), str(t["predicate"]).lower(), str(t["object"]).lower()) for t in triples]


# ---------------------------------------------------------------------------
# 3. EMBED ENTITIES — Get OpenAI embeddings for every unique entity
# ---------------------------------------------------------------------------
def embed_entities(client, entities: list[str]) -> dict[str, list[float]]:
    """Batch-embed a list of entity names using OpenAI."""
    # text-embedding-3-small: 1536 dims, cheap, good quality
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=entities,
    )
    return {
        entity: item.embedding
        for entity, item in zip(entities, response.data)
    }


# ---------------------------------------------------------------------------
# 4. BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_graph(client) -> Graph:
    """Full pipeline: text → triples → graph + embeddings."""
    g = Graph("TextKG")

    all_triples = []
    print("=" * 60)
    print("STEP 1: Extracting triples from text with GPT-5-mini")
    print("=" * 60)

    for i, text in enumerate(SAMPLE_TEXTS):
        print(f"\n--- Passage {i+1} ---")
        triples = extract_triples(client, text)
        all_triples.extend(triples)
        for s, p, o in triples:
            print(f"  ({s}) --[{p}]--> ({o})")

    print(f"\nTotal triples extracted: {len(all_triples)}")

    # Insert triples into CogDB
    print("\n" + "=" * 60)
    print("STEP 2: Loading triples into CogDB")
    print("=" * 60)
    start = time.time()
    g.put_batch(all_triples)
    elapsed = time.time() - start
    print(f"  Inserted {len(all_triples)} triples in {elapsed:.3f}s")

    # Collect unique entities (subjects + objects)
    entities = list(set(
        [s for s, _, _ in all_triples] + [o for _, _, o in all_triples]
    ))
    print(f"  Unique entities: {len(entities)}")

    # Embed all entities
    print("\n" + "=" * 60)
    print("STEP 3: Generating OpenAI embeddings for all entities")
    print("=" * 60)
    start = time.time()
    embeddings = embed_entities(client, entities)
    elapsed = time.time() - start
    dim = len(next(iter(embeddings.values())))
    print(f"  Embedded {len(entities)} entities ({dim}-dim) in {elapsed:.3f}s")

    # Store embeddings in CogDB
    print("\n" + "=" * 60)
    print("STEP 4: Storing embeddings in CogDB")
    print("=" * 60)
    start = time.time()
    for entity, vec in embeddings.items():
        g.put_embedding(entity, vec)
    elapsed = time.time() - start
    print(f"  Stored {len(entities)} embeddings in {elapsed:.3f}s")

    return g, entities, embeddings


# ---------------------------------------------------------------------------
# 5. DEMO QUERIES — Show graph traversal + semantic search
# ---------------------------------------------------------------------------
def run_demos(g: Graph, entities: list, embeddings: dict):
    print("\n" + "=" * 60)
    print("STEP 5: Running queries")
    print("=" * 60)

    # --- Query A: Basic graph traversal ---
    print("\n--- A. Graph Traversal: What did Einstein do? ---")
    results = g.v("einstein").out().all()
    print(f"  einstein --> {results}")

    # --- Query B: Incoming edges ---
    print("\n--- B. Incoming Edges: What connects to Python? ---")
    results = g.v("python").inc().all()
    print(f"  ? --> python: {results}")

    # --- Query C: Two-hop traversal ---
    print("\n--- C. Two-Hop: einstein -> ? -> ? ---")
    results = g.v("einstein").out().out().all()
    print(f"  einstein (2 hops): {results}")

    # --- Query D: Semantic similarity (k-nearest) ---
    print("\n--- D. Semantic Search: 5 entities most similar to 'einstein' ---")
    results = g.v().k_nearest("einstein", k=5).all()
    print(f"  k_nearest: {results}")

    # --- Query E: Semantic similarity threshold ---
    print("\n--- E. Similarity > 0.5 to 'machine learning' ---")
    # 'machine learning' might not be a node, so let's find what's close
    # to a known entity
    results = g.v().sim("pytorch", ">", 0.3).all()
    print(f"  sim > 0.3 to 'pytorch': {results}")

    # --- Query F: Graph + Vector combined ---
    # "Find entities related to Einstein that are semantically similar to physics"
    print("\n--- F. Combined: Neighbors of 'einstein' ranked by similarity ---")
    neighbors = g.v("einstein").out().all()
    if neighbors and "result" in neighbors:
        neighbor_ids = [n["id"] for n in neighbors["result"]]
        print(f"  Graph neighbors: {neighbor_ids}")
        # Now find which of ALL entities are closest to "general_relativity"
        # (showing the graph narrows context, vector ranks within it)
        nearest = g.v().k_nearest("general_relativity", k=5).all()
        print(f"  Top 5 nearest to 'general_relativity': {nearest}")

    # --- Query G: Count and filter ---
    print("\n--- G. How many entities in the graph? ---")
    count = g.v().out().count()
    print(f"  Total edges: {count}")

    # --- Query H: Filter with lambda ---
    print("\n--- H. All entities starting with 'p' ---")
    results = g.v().filter(lambda v: v.startswith("p")).all()
    print(f"  p* entities: {results}")

    # --- Query I: Tagged traversal ---
    print("\n--- I. Tagged: Who created what? ---")
    results = (
        g.v()
        .tag("creator")
        .out("created")
        .tag("creation")
        .all()
    )
    print(f"  Creator->Creation: {results}")

    # --- Query J: BFS ---
    print("\n--- J. BFS from 'python' (depth=2) ---")
    results = g.v("python").bfs(max_depth=2).all()
    print(f"  BFS: {results}")


# ---------------------------------------------------------------------------
# 6. STATS
# ---------------------------------------------------------------------------
def print_stats(g: Graph):
    print("\n" + "=" * 60)
    print("GRAPH STATS")
    print("=" * 60)

    total_edges = g.v().out().count()
    print(f"  Total edges (outgoing): {total_edges}")

    # Check graph files on disk
    import glob
    files = glob.glob("cog_home/TextKG/**/*", recursive=True)
    total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
    print(f"  Files on disk: {len([f for f in files if os.path.isfile(f)])}")
    print(f"  Total disk size: {total_size / 1024:.1f} KB")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY in .env file or environment")
        print("  echo 'OPENAI_API_KEY=sk-...' > .env")
        exit(1)

    client = OpenAI(api_key=api_key)

    from importlib.metadata import version
    print("\n🧠 CogDB Demo: Text → Knowledge Graph → Semantic Search")
    print(f"   CogDB version: {version('cogdb')}\n")

    g, entities, embeddings = build_graph(client)
    run_demos(g, entities, embeddings)
    print_stats(g)

    g.sync()
    # Close graph to flush data to disk
    g.close()

    print("\n" + "=" * 60)
    print("DONE. Graph persisted to ./cog_home/TextKG/")
    print("Re-open anytime with: Graph('TextKG')")
    print("=" * 60)

    # open the graph again and do a count
    g2 = Graph("TextKG")
    count = g2.v().out().count()
    print(f"\nRe-opened graph, total edges: {count}")
    g2.close()