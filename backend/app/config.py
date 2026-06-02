from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
IS_SERVERLESS = os.getenv("VERCEL") == "1"
DATA_DIR = Path(os.getenv("MOLSPACE_DATA_DIR", "/tmp/molecule-space-data" if IS_SERVERLESS else str(ROOT_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT_DIR / ".env")


COLLECTION_NAME = os.getenv("MOLSPACE_COLLECTION", "molspace_molecules")
STRUCTURE_DIM = 2048
BIOACTIVITY_DIM = 384
DEFAULT_MAX_RECORDS = int(os.getenv("MOLSPACE_MAX_RECORDS", "1000"))
AUTO_BUILD = os.getenv("MOLSPACE_AUTO_BUILD", "0") == "1"
DISABLE_CHEMBL = os.getenv("MOLSPACE_DISABLE_CHEMBL", "0") == "1"

RAW_CHEMBL_PATH = DATA_DIR / "raw_chembl_activities.json"
PROCESSED_PATH = DATA_DIR / "processed_molecules.json"
LOCAL_QDRANT_PATH = Path(os.getenv("QDRANT_LOCAL_PATH", "/tmp/molecule-space-qdrant" if IS_SERVERLESS else str(BACKEND_DIR / ".qdrant")))
