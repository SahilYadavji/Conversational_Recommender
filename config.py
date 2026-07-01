"""
config.py

Centralized configuration for the SHL Assessment Recommender.

Responsibilities
----------------
- Load environment variables
- Validate required configuration
- Manage project paths
- Configure model settings
- Configure retrieval settings
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

CATALOG_PATH = DATA_DIR / "catalog.json"
FAISS_PATH = EMBEDDINGS_DIR / "faiss.index"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.pkl"

# ---------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

print("=" * 50)
print("DEBUG")
print("GEMINI_API_KEY exists:", "GEMINI_API_KEY" in os.environ)
print("GEMINI_API_KEY value:", os.getenv("GEMINI_API_KEY"))
print("Length:", len(os.getenv("GEMINI_API_KEY", "")))
print("=" * 50)

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is missing.\n"
        "Please add it to your .env file:\n\n"
        "GEMINI_API_KEY=your_api_key"
    )

# ---------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# ---------------------------------------------------------------------
# Retrieval Configuration
# ---------------------------------------------------------------------

TOP_K = int(os.getenv("TOP_K", "10"))

# Minimum similarity score to keep a result.
# Tune this after evaluating retrieval quality.
MIN_SIMILARITY_SCORE = float(
    os.getenv("MIN_SIMILARITY_SCORE", "0.25")
)

# ---------------------------------------------------------------------
# Gemini Configuration
# ---------------------------------------------------------------------

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)

GEMINI_TEMPERATURE = float(
    os.getenv("GEMINI_TEMPERATURE", "0.2")
)

MAX_OUTPUT_TOKENS = int(
    os.getenv("MAX_OUTPUT_TOKENS", "700")
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Validate Required Files
# ---------------------------------------------------------------------

required_files = [
    CATALOG_PATH,
]

for file in required_files:
    if not file.exists():
        logger.warning("Missing file: %s", file)

# Ensure directories exist
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Configuration loaded successfully.")