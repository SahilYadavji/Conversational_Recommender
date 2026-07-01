"""
search.py

Semantic search engine for the SHL Assessment Recommender.

Responsibilities
----------------
- Load the SentenceTransformer embedding model.
- Load the FAISS vector index.
- Load assessment metadata.
- Encode user queries.
- Perform semantic similarity search.
- Return ranked assessment results.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    FAISS_PATH,
    METADATA_PATH,
    TOP_K,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Load Resources (executed once when module is imported)
# ---------------------------------------------------------------------

logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
model = SentenceTransformer(EMBEDDING_MODEL)

logger.info("Loading FAISS index...")
index = faiss.read_index(str(Path(FAISS_PATH)))

logger.info("Loading metadata...")

with open(METADATA_PATH, "rb") as file:
    metadata: list[dict[str, Any]] = pickle.load(file)

logger.info(
    "Loaded %d assessments into memory.",
    len(metadata),
)


# ---------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------

def _encode_query(query: str) -> np.ndarray:
    """
    Encode a query into an embedding vector.
    """

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embedding.astype(np.float32)


def _normalize_score(distance: float) -> float:
    """
    Convert FAISS similarity score into a float.

    Higher score = better match.
    """

    return round(float(distance), 4)


# ---------------------------------------------------------------------
# Public Search API
# ---------------------------------------------------------------------

def search_assessments(
    query: str,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """
    Search the FAISS index for relevant SHL assessments.

    Args:
        query:
            User query.

        top_k:
            Maximum number of results.

    Returns:
        Ranked assessment list.
    """

    if not query.strip():
        return []

    logger.info("Searching for: %s", query)

    embedding = _encode_query(query)

    distances, indices = index.search(
        embedding,
        top_k,
    )

    seen = set()
    results: list[dict[str, Any]] = []

    for idx, score in zip(indices[0], distances[0]):

        if idx == -1:
            continue

        if idx >= len(metadata):
            continue

        assessment = metadata[idx].copy()

        name = assessment.get("name")

        # Remove duplicate assessments
        if name in seen:
            continue

        seen.add(name)

        assessment["score"] = _normalize_score(score)

        results.append(assessment)

    # Highest score first
    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    logger.info(
        "Returning %d assessment(s).",
        len(results),
    )

    return results


# ---------------------------------------------------------------------
# CLI Test Utility
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("SHL Assessment Semantic Search")
    print("Type 'exit' to quit.")
    print("=" * 70)

    while True:

        query = input("\nSearch > ").strip()

        if query.lower() == "exit":
            break

        results = search_assessments(query)

        if not results:
            print("\nNo matching assessments found.")
            continue

        print()

        for i, item in enumerate(results, start=1):

            print("=" * 80)
            print(f"{i}. {item.get('name', 'Unknown')}")
            print(f"Score : {item.get('score')}")
            print(f"URL   : {item.get('link', '-')}")
            print(f"Type  : {', '.join(item.get('keys', []))}")