"""Knowledge Graph queries — traversal, semantic search, and hybrid.

Opens an existing CogDB graph (built by etl.py) and runs demo queries.
"""

import os

from cog.torque import Graph
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def fmt_result(result) -> str:
    """Format a CogDB result dict into readable lines."""
    if not result or "result" not in result:
        return "  (no results)"
    lines = []
    for item in result["result"]:
        if isinstance(item, dict):
            parts = [f"{k}={v}" for k, v in item.items() if k != "id" or len(item) == 1]
            if "id" in item and len(item) > 1:
                lines.append(f"  {item['id']:30s} | {', '.join(parts)}")
            else:
                lines.append(f"  {item.get('id', str(item))}")
        else:
            lines.append(f"  {item}")
    return "\n".join(lines)


def summarize_result(result) -> list[str]:
    """Extract entity names from a CogDB result."""
    if not result or "result" not in result:
        return []
    return [
        item["id"] if isinstance(item, dict) else str(item)
        for item in result["result"]
    ]


def print_query(number: int, question: str, torque_str: str, result):
    """Print a query with question, Torque expression, and result."""
    print(f"\n{'─' * 60}")
    print(f"  Q{number}: {question}")
    print(f"  Torque:  {torque_str}")
    print(f"{'─' * 60}")
    print(fmt_result(result))


def embed_query(client, g: Graph, text: str) -> str:
    """Embed a query string and store it in the graph for k_nearest lookups."""
    tag = f"__query__{text}"
    if g.get_embedding(tag) is None:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text],
        )
        g.put_embedding(tag, response.data[0].embedding)
    return tag


def run_queries(g: Graph, client):
    """Showcase graph traversal, semantic search, and hybrid queries."""
    print("\n" + "═" * 60)
    print("  KNOWLEDGE GRAPH QUERIES")
    print("═" * 60)

    # ── A. Graph Traversal ──

    print("\n  ── A. Graph Traversal ──\n")

    print_query(
        1, "What is Europa connected to in the graph?",
        'g.v("europa").out().all()',
        g.v("europa").out().all(),
    )

    print_query(
        2, "Which entities mention or relate to water?",
        'g.v("water").inc().all()',
        g.v("water").inc().all(),
    )

    print_query(
        3, "Where is Jezero Crater located, and what might that body harbor?",
        'g.v("jezero crater").out("located_in").out("may_harbor").all()',
        g.v("jezero crater").out("located_in").out("may_harbor").all(),
    )

    print_query(
        4, "Which missions target bodies that may harbor life?",
        'g.v("life").inc("may_harbor").inc("targets").all()',
        g.v("life").inc("may_harbor").inc("targets").all(),
    )

    # ── B. Semantic Search ──

    print("\n  ── B. Semantic Search ──\n")

    tag5 = embed_query(client, g, "signs of life")
    print_query(
        5, 'Nearest entities to "signs of life"',
        'g.v().k_nearest("signs of life", k=5).all()',
        g.v().k_nearest(tag5, k=5).all(),
    )

    tag6 = embed_query(client, g, "ocean world")
    print_query(
        6, 'Nearest entities to "ocean world"',
        'g.v().k_nearest("ocean world", k=5).all()',
        g.v().k_nearest(tag6, k=5).all(),
    )

    # ── C. Hybrid: Graph Traversal + Semantic Filtering ──

    print("\n  ── C. Hybrid: Graph Traversal + Semantic Filtering ──\n")

    tag7 = embed_query(client, g, "water and ice")
    print_query(
        7, 'Explore two hops from Europa, then find the most relevant to "water and ice"',
        'g.v("europa").bfs(max_depth=2).k_nearest("water and ice", k=3).all()',
        g.v("europa").bfs(max_depth=2).k_nearest(tag7, k=3).all(),
    )

    tag8 = embed_query(client, g, "extraterrestrial biology")
    print_query(
        8, 'Which bodies may harbor life, ranked by relevance to "extraterrestrial biology"?',
        'g.v("life").inc("may_harbor").k_nearest("extraterrestrial biology", k=5).all()',
        g.v("life").inc("may_harbor").k_nearest(tag8, k=5).all(),
    )

    tag9 = embed_query(client, g, "ocean")
    print_query(
        9, 'What is Enceladus connected to that is most related to "ocean"?',
        'g.v("enceladus").out().k_nearest("ocean", k=3).all()',
        g.v("enceladus").out().k_nearest(tag9, k=3).all(),
    )

    tag10 = embed_query(client, g, "habitable world")
    print_query(
        10, 'What do NASA spacecraft target, ranked by "habitable world"?',
        'g.v("nasa").inc("explored_by").out("targets").k_nearest("habitable world", k=5).all()',
        g.v("nasa").inc("explored_by").out("targets").k_nearest(tag10, k=5).all(),
    )

    tag11 = embed_query(client, g, "icy moon")
    print_query(
        11, 'Find entities similar to "icy moon", then see what they may harbor',
        'g.v().k_nearest("icy moon", k=3).out("may_harbor").all()',
        g.v().k_nearest(tag11, k=3).out("may_harbor").all(),
    )

    tag12 = embed_query(client, g, "robotic spacecraft")
    print_query(
        12, 'Which missions target habitable bodies, ranked by "robotic spacecraft"?',
        'g.v("life").inc("may_harbor").inc("targets").k_nearest("robotic spacecraft", k=3).all()',
        g.v("life").inc("may_harbor").inc("targets").k_nearest(tag12, k=3).all(),
    )

    tag13a = embed_query(client, g, "space exploration mission")
    print_query(
        13, 'Find "space exploration" entities, follow their edges, then filter for "habitable world"',
        'g.v().k_nearest("space exploration", k=10).out().k_nearest("habitable world", k=5).all()',
        g.v().k_nearest(tag13a, k=10).out().k_nearest(tag10, k=5).all(),
    )

    tag14a = embed_query(client, g, "deep space probe")
    tag14b = embed_query(client, g, "frozen world")
    print_query(
        14, 'Find "deep space probe" entities, see what they target, then filter for "frozen world"',
        'g.v().k_nearest("deep space probe", k=5).out("targets").k_nearest("frozen world", k=3).all()',
        g.v().k_nearest(tag14a, k=5).out("targets").k_nearest(tag14b, k=3).all(),
    )

    total_edges = g.v().out().count()
    print(f"\n{'─' * 60}")
    print(f"  Graph stats: {total_edges} edges")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY in .env file or environment")
        print("  echo 'OPENAI_API_KEY=sk-...' > .env")
        exit(1)

    client = OpenAI(api_key=api_key)

    g = Graph("AstrobiologyKG")

    run_queries(g, client)
    g.close()
