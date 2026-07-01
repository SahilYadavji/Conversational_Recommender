"""
vector_store.py

Builds the FAISS vector database for the SHL Assessment Recommender.

Responsibilities
----------------
- Load SHL assessment catalog
- Convert assessments into searchable documents
- Generate embeddings
- Normalize embeddings for cosine similarity
- Build a FAISS index
- Save metadata and index

Run:

    python vector_store.py
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CATALOG_PATH,
    EMBEDDING_MODEL,
    FAISS_PATH,
    METADATA_PATH,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------------------

logger.info("Loading embedding model: %s", EMBEDDING_MODEL)

model = SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_catalog() -> list[dict[str, Any]]:
    """
    Load SHL assessment catalog.
    """

    catalog_file = Path(CATALOG_PATH)

    if not catalog_file.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_file}")

    with catalog_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        catalog = json.load(file)

    logger.info("Loaded %d assessments.", len(catalog))

    return catalog


def build_document(item: dict[str, Any]) -> str:
    """
    Convert one assessment into a semantic document.
    """

    return f"""
Assessment Name:
{item.get("name", "")}

Description:
{item.get("description", "")}

Job Levels:
{", ".join(item.get("job_levels", []))}

Assessment Types:
{", ".join(item.get("keys", []))}

Languages:
{", ".join(item.get("languages", []))}

Remote Testing:
{item.get("remote", "")}

Adaptive Testing:
{item.get("adaptive", "")}
""".strip()


def generate_embeddings(documents: list[str]) -> np.ndarray:
    """
    Generate normalized embeddings.
    """

    logger.info("Generating embeddings...")

    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype(np.float32)


def save_metadata(metadata: list[dict[str, Any]]) -> None:
    """
    Save metadata alongside the FAISS index.
    """

    with open(METADATA_PATH, "wb") as file:
        pickle.dump(metadata, file)

    logger.info("Metadata saved.")


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a cosine similarity FAISS index.
    """

    dimension = embeddings.shape[1]

    logger.info("Embedding dimension: %d", dimension)

    # Cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    logger.info("Indexed %d vectors.", index.ntotal)

    return index


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    catalog = load_catalog()

    documents = []
    metadata = []

    for assessment in catalog:

        documents.append(build_document(assessment))

        metadata.append(assessment)

    embeddings = generate_embeddings(documents)

    index = build_index(embeddings)

    faiss.write_index(index, FAISS_PATH)

    logger.info("FAISS index saved to %s", FAISS_PATH)

    save_metadata(metadata)

    logger.info("=" * 60)
    logger.info("Vector store successfully created.")
    logger.info("Assessments : %d", len(metadata))
    logger.info("Vectors     : %d", index.ntotal)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()