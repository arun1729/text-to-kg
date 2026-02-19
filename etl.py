"""Text → Ontology → Knowledge Graph ETL pipeline.

Pipeline:
  1. Define ontology (entity types + relationship types)
  2. Extract entities with type labels
  3. Extract relationships guided by ontology
  4. Resolve / normalize entities
  5. Generate embeddings
  6. Load into CogDB
"""

import glob
import json
import os
import re
import time
from collections import Counter

from cog.torque import Graph
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kg_data.json")
TEXT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planetary-habitability.txt")

ONTOLOGY = {
    "entity_types": {
        "CelestialBody": "A planet, dwarf planet, or star (e.g. Mars, Jupiter, Sun, Pluto)",
        "Moon": "A natural satellite of a planet (e.g. Europa, Titan, Enceladus)",
        "Mission": "A space mission or program (e.g. Cassini, Perseverance, Viking)",
        "Spacecraft": "A specific vehicle, rover, lander, or telescope (e.g. Hubble, JWST, Curiosity rover)",
        "SpaceAgency": "An organization that operates space missions (e.g. NASA, ESA, JAXA)",
        "Scientist": "A person who contributed to the field (e.g. Carl Sagan, Frank Drake)",
        "Chemical": "A chemical element, compound, or molecule (e.g. water, methane, phosphine, CO2)",
        "Feature": "A geological or physical feature (e.g. subsurface ocean, ice shell, hydrothermal vent)",
        "Concept": "A scientific concept, theory, or phenomenon (e.g. habitable zone, biosignature, tidal heating)",
        "Region": "A named area or location on a body (e.g. Jezero Crater, Arcadia Planitia)",
        "Instrument": "A scientific instrument on a spacecraft (e.g. mass spectrometer, drill, spectrograph)",
        "Event": "A specific event or discovery (e.g. 2020 phosphine detection, Enceladus plume discovery)",
    },
    "relationship_types": {
        "ORBITS": "CelestialBody/Moon ORBITS CelestialBody (e.g. Europa ORBITS Jupiter)",
        "MOON_OF": "Moon MOON_OF CelestialBody (e.g. Titan MOON_OF Saturn)",
        "HAS_FEATURE": "CelestialBody/Moon HAS_FEATURE Feature (e.g. Europa HAS_FEATURE subsurface ocean)",
        "HAS_ATMOSPHERE": "CelestialBody/Moon HAS_ATMOSPHERE Chemical (e.g. Titan HAS_ATMOSPHERE methane)",
        "EXPLORED_BY": "CelestialBody/Moon EXPLORED_BY Mission/Spacecraft (e.g. Mars EXPLORED_BY Perseverance)",
        "OPERATED_BY": "Mission/Spacecraft OPERATED_BY SpaceAgency (e.g. Cassini OPERATED_BY NASA)",
        "LAUNCHED_IN": "Mission/Spacecraft LAUNCHED_IN year (e.g. JWST LAUNCHED_IN 2021)",
        "CARRIES": "Mission/Spacecraft CARRIES Instrument (e.g. Perseverance CARRIES drill)",
        "DISCOVERED": "Mission/Scientist DISCOVERED Event/Feature (e.g. Cassini DISCOVERED Enceladus plumes)",
        "INDICATES": "Feature/Chemical INDICATES Concept (e.g. methane INDICATES possible biosignature)",
        "REQUIRES": "Concept REQUIRES Chemical/Feature (e.g. life REQUIRES liquid water)",
        "LOCATED_IN": "Feature/Region LOCATED_IN CelestialBody/Moon (e.g. Jezero Crater LOCATED_IN Mars)",
        "MAY_HARBOR": "CelestialBody/Moon MAY_HARBOR Concept (e.g. Europa MAY_HARBOR microbial life)",
        "CONTAINS": "CelestialBody/Moon/Feature CONTAINS Chemical (e.g. Enceladus plumes CONTAINS water vapor)",
        "SUCCESSOR_OF": "Mission SUCCESSOR_OF Mission (e.g. Perseverance SUCCESSOR_OF Curiosity)",
        "TARGETS": "Mission/Spacecraft TARGETS CelestialBody/Moon (e.g. Europa Clipper TARGETS Europa)",
        "EVIDENCE_FOR": "Feature/Event EVIDENCE_FOR Concept (e.g. subsurface ocean EVIDENCE_FOR habitability)",
        "PROPOSED_BY": "Concept PROPOSED_BY Scientist (e.g. Drake Equation PROPOSED_BY Frank Drake)",
        "PART_OF": "Moon/Region PART_OF CelestialBody (e.g. Olympus Mons PART_OF Mars)",
    },
}

ALIASES = {
    "james webb space telescope": "jwst",
    "james webb": "jwst",
    "webb telescope": "jwst",
    "webb": "jwst",
    "jst": "jwst",
    "perseverance rover": "perseverance",
    "curiosity rover": "curiosity",
    "opportunity rover": "opportunity",
    "spirit rover": "spirit",
    "europa clipper mission": "europa clipper",
    "cassini-huygens": "cassini",
    "cassini spacecraft": "cassini",
    "huygens probe": "huygens",
    "dragonfly mission": "dragonfly",
    "h2o": "water",
    "liquid water": "water",
    "ch4": "methane",
    "co2": "carbon dioxide",
    "carbon dioxide (co2)": "carbon dioxide",
    "nh3": "ammonia",
    "h2": "hydrogen",
    "nacl": "sodium chloride",
    "european space agency": "esa",
    "national aeronautics and space administration": "nasa",
    "japan aerospace exploration agency": "jaxa",
}


def ontology_prompt_block() -> str:
    """Format the ontology as a prompt-friendly string."""
    lines = ["## Entity Types"]
    for etype, desc in ONTOLOGY["entity_types"].items():
        lines.append(f"- {etype}: {desc}")
    lines.append("\n## Relationship Types (SUBJECT --[REL]--> OBJECT)")
    for rel, desc in ONTOLOGY["relationship_types"].items():
        lines.append(f"- {rel}: {desc}")
    return "\n".join(lines)


def extract_entities(client, text: str) -> list[dict]:
    """Extract typed entities from text, guided by the ontology."""
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledge graph entity extractor.\n\n"
                    "Given a text about planetary science and astrobiology, extract "
                    "all named entities. For each entity return:\n"
                    '  {"name": "<short canonical name>", "type": "<EntityType>"}\n\n'
                    "Use these entity types ONLY:\n"
                    + "\n".join(f"- {k}: {v}" for k, v in ONTOLOGY["entity_types"].items())
                    + "\n\nRules:\n"
                    "- Use short canonical names: 'Europa' not 'Europa, a moon of Jupiter'\n"
                    "- Normalize: 'JWST' and 'James Webb Space Telescope' → 'JWST'\n"
                    "- One entry per unique entity\n"
                    "- Return ONLY a JSON array. No commentary.\n"
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def extract_relationships(client, text: str, entities: list[dict]) -> list[dict]:
    """Extract typed relationships between known entities."""
    entity_list = "\n".join(f"- {e['name']} ({e['type']})" for e in entities)

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledge graph relationship extractor.\n\n"
                    "Given a text and a list of known entities, extract relationships "
                    "between them. Each relationship MUST be a JSON object with exactly "
                    "three keys: \"subject\", \"predicate\", and \"object\".\n\n"
                    "Example output:\n"
                    '[{"subject": "Europa", "predicate": "MOON_OF", "object": "Jupiter"},\n'
                    ' {"subject": "Europa", "predicate": "HAS_FEATURE", "object": "subsurface ocean"},\n'
                    ' {"subject": "Cassini", "predicate": "OPERATED_BY", "object": "NASA"}]\n\n'
                    "Use these relationship types ONLY:\n"
                    + "\n".join(f"- {k}: {v}" for k, v in ONTOLOGY["relationship_types"].items())
                    + "\n\nRules:\n"
                    "- ONLY use entities from the provided entity list\n"
                    "- ONLY use relationship types from the list above\n"
                    "- Every object MUST have exactly the keys: subject, predicate, object\n"
                    "- Extract as many relationships as the text supports\n"
                    "- Each triple must be factually grounded in the text\n"
                    "- Return ONLY a JSON array. No commentary.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Known Entities\n{entity_list}\n\n"
                    f"## Source Text\n{text}"
                ),
            },
        ],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    parsed = json.loads(raw)
    if not parsed:
        return parsed

    valid_rel_keys = set(ONTOLOGY["relationship_types"].keys())
    normalized = []
    for r in parsed:
        if "predicate" in r and "object" in r:
            normalized.append(r)
        else:
            for key in r:
                upper_key = key.strip().upper().replace(" ", "_").replace("-", "_")
                if upper_key in valid_rel_keys and key != "subject":
                    normalized.append({
                        "subject": r.get("subject", ""),
                        "predicate": upper_key,
                        "object": r[key],
                    })
    return normalized


def normalize_name(name: str) -> str:
    """Normalize an entity name to a canonical form."""
    name = name.strip().lower()
    name = re.sub(r"\s+", " ", name)
    return name


def resolve_entities(
    entities: list[dict],
    relationships: list[dict],
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Normalize entity names and deduplicate. Returns resolved entities and triples."""
    canonical = {}
    for e in entities:
        raw = e["name"]
        norm = normalize_name(raw)
        canon = ALIASES.get(norm, norm)
        canonical[raw] = canon
        canonical[norm] = canon

    seen = {}
    for e in entities:
        canon = canonical.get(e["name"], normalize_name(e["name"]))
        if canon not in seen:
            seen[canon] = e["type"]

    resolved_entities = [{"name": k, "type": v} for k, v in seen.items()]

    triples = []
    valid_rels = set(ONTOLOGY["relationship_types"].keys())
    for r in relationships:
        subj_raw = r.get("subject") or r.get("source") or r.get("head")
        pred_raw = r.get("predicate") or r.get("relation") or r.get("relationship") or r.get("type") or r.get("rel")
        obj_raw = r.get("object") or r.get("target") or r.get("tail")

        if not all([subj_raw, pred_raw, obj_raw]):
            continue

        subj = canonical.get(subj_raw, normalize_name(subj_raw))
        pred = pred_raw.strip().upper().replace(" ", "_").replace("-", "_")
        obj = canonical.get(obj_raw, normalize_name(obj_raw))

        if subj_raw not in canonical:
            subj = canonical.get(normalize_name(subj_raw), normalize_name(subj_raw))
        if obj_raw not in canonical:
            obj = canonical.get(normalize_name(obj_raw), normalize_name(obj_raw))

        if pred not in valid_rels:
            continue
        if subj == obj:
            continue

        triples.append((subj, pred.lower(), obj))

    triples = list(set(triples))
    return resolved_entities, triples


def embed_entities(client, entities: list[dict]) -> dict[str, list[float]]:
    """Embed entities using their name + type for richer semantics."""
    texts = [f"{e['name']} ({e['type']})" for e in entities]
    names = [e["name"] for e in entities]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return {
        name: item.embedding
        for name, item in zip(names, response.data)
    }


def save_data(
    entities: list[dict],
    triples: list[tuple[str, str, str]],
    embeddings: dict[str, list[float]],
):
    """Save extracted data to JSON."""
    data = {
        "entities": entities,
        "triples": [[s, p, o] for s, p, o in triples],
        "embeddings": embeddings,
    }
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)
    size_kb = os.path.getsize(DATA_FILE) / 1024
    print(f"Saved {len(entities)} entities, {len(triples)} triples, "
          f"{len(embeddings)} embeddings ({size_kb:.1f} KB)")


def load_data() -> tuple[list[dict], list[tuple[str, str, str]], dict[str, list[float]]]:
    """Load previously extracted data from JSON."""
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    entities = data.get("entities", [])
    triples = [(s, p, o) for s, p, o in data["triples"]]
    embeddings = data.get("embeddings", {})
    print(f"Loaded {len(entities)} entities, {len(triples)} triples, "
          f"{len(embeddings)} embeddings")
    return entities, triples, embeddings


def load_text(path: str) -> str:
    """Read the full source text file."""
    with open(path, "r") as f:
        return f.read().strip()


def build_graph(
    triples: list[tuple[str, str, str]],
    embeddings: dict[str, list[float]],
) -> Graph:
    """Load triples and embeddings into CogDB."""
    g = Graph("AstrobiologyKG")
    g.drop()
    g = Graph("AstrobiologyKG")

    start = time.time()
    g.put_batch(triples)
    print(f"Loaded {len(triples)} triples into CogDB in {time.time() - start:.3f}s")

    start = time.time()
    for entity, vec in embeddings.items():
        g.put_embedding(entity, vec)
    print(f"Stored {len(embeddings)} embeddings in {time.time() - start:.3f}s")

    return g


def extract_and_save(client) -> tuple[list[dict], list[tuple[str, str, str]], dict[str, list[float]]]:
    """Full pipeline: text → entities → relationships → resolve → embed → save."""
    text = load_text(TEXT_FILE)
    print(f"Source text: {len(text)} chars\n")

    print("Step 1: Extracting entities...")
    start = time.time()
    raw_entities = extract_entities(client, text)
    print(f"  Found {len(raw_entities)} entities in {time.time() - start:.1f}s")
    type_counts = Counter(e["type"] for e in raw_entities)
    for etype, count in sorted(type_counts.items()):
        print(f"    {etype}: {count}")

    print("\nStep 2: Extracting relationships...")
    start = time.time()
    raw_relationships = extract_relationships(client, text, raw_entities)
    print(f"  Found {len(raw_relationships)} relationships in {time.time() - start:.1f}s")

    print("\nStep 3: Resolving entities...")
    entities, triples = resolve_entities(raw_entities, raw_relationships)
    print(f"  {len(raw_entities)} raw → {len(entities)} resolved entities")
    print(f"  {len(raw_relationships)} raw → {len(triples)} valid triples")

    print("\n  Sample triples:")
    for s, p, o in triples[:20]:
        print(f"    ({s}) ──[{p}]──▶ ({o})")
    if len(triples) > 20:
        print(f"    ... and {len(triples) - 20} more")

    print(f"\nStep 4: Generating embeddings for {len(entities)} entities...")
    start = time.time()
    embeddings = embed_entities(client, entities)
    dim = len(next(iter(embeddings.values())))
    print(f"  {len(embeddings)} embeddings ({dim}-dim) in {time.time() - start:.1f}s")

    save_data(entities, triples, embeddings)
    return entities, triples, embeddings


def print_stats(g: Graph):
    """Print graph size statistics."""
    total_edges = g.v().out().count()
    files = [
        f for f in glob.glob("cog_home/AstrobiologyKG/**/*", recursive=True)
        if os.path.isfile(f)
    ]
    total_size = sum(os.path.getsize(f) for f in files)
    print(f"\nStats: {total_edges} edges, {len(files)} files, "
          f"{total_size / 1024:.1f} KB on disk")


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY in .env file or environment")
        print("  echo 'OPENAI_API_KEY=sk-...' > .env")
        exit(1)

    client = OpenAI(api_key=api_key)

    from importlib.metadata import version
    print(f"cogdb {version('cogdb')}\n")

    if os.path.isfile(DATA_FILE):
        print(f"Data file found: {DATA_FILE}")
        entities, triples, embeddings = load_data()
    else:
        print(f"No data file found — running extraction pipeline\n")
        entities, triples, embeddings = extract_and_save(client)

    g = build_graph(triples, embeddings)
    print_stats(g)

    g.sync()
    g.close()
    print("\nGraph persisted to ./cog_home/AstrobiologyKG/")
