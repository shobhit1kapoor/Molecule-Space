from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .config import BIOACTIVITY_DIM


_vectorizer = HashingVectorizer(
    n_features=BIOACTIVITY_DIM,
    alternate_sign=False,
    norm=None,
    analyzer="word",
    ngram_range=(1, 2),
)


def bioactivity_text(payload: dict[str, Any]) -> str:
    parts = [
        payload.get("name") or payload.get("molecule_id") or "unnamed molecule",
        payload.get("target_name") or "unknown target",
        payload.get("target_class") or "unknown class",
        payload.get("bioactivity_type") or "bioactivity",
        payload.get("assay_description") or "",
        payload.get("organism") or "",
    ]
    pchembl = payload.get("pchembl_value")
    if pchembl is not None:
        parts.append(f"potency pchembl {pchembl}")
    return " ".join(str(part) for part in parts if part)


def text_vector(text: str) -> list[float]:
    arr = _vectorizer.transform([text]).toarray().astype(np.float32)[0]
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    a = np.array(left, dtype=np.float32)
    b = np.array(right, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0 or math.isnan(denom):
        return 0.0
    return round(float(np.dot(a, b) / denom), 4)

