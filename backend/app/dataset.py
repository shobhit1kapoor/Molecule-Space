from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import requests
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

from . import config
from .chemistry import (
    canonicalize_smiles,
    compute_descriptors,
    sparse_fingerprint,
    structure_vector,
    toxicity_flag,
)
from .vectorize import bioactivity_text, text_vector


ANCHORS: list[dict[str, Any]] = [
    # Recognizable anchor molecules make the demo predictable and let users
    # search by familiar names while still mixing in ChEMBL-derived records.
    {
        "molecule_id": "CHEMBL25",
        "name": "Aspirin",
        "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "target_name": "Cyclooxygenase prostaglandin pathway",
        "target_class": "enzyme",
        "bioactivity_type": "IC50",
        "pchembl_value": 6.3,
        "assay_description": "anti-inflammatory cyclooxygenase inhibition",
        "organism": "Homo sapiens",
        "max_phase": 4,
        "source": "anchor",
    },
    {
        "molecule_id": "CHEMBL521",
        "name": "Ibuprofen",
        "canonical_smiles": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "target_name": "Cyclooxygenase prostaglandin pathway",
        "target_class": "enzyme",
        "bioactivity_type": "IC50",
        "pchembl_value": 6.1,
        "assay_description": "anti-inflammatory cyclooxygenase inhibition",
        "organism": "Homo sapiens",
        "max_phase": 4,
        "source": "anchor",
    },
    {
        "molecule_id": "CHEMBL154",
        "name": "Naproxen",
        "canonical_smiles": "COc1ccc2cc([C@H](C)C(=O)O)ccc2c1",
        "target_name": "Cyclooxygenase prostaglandin pathway",
        "target_class": "enzyme",
        "bioactivity_type": "IC50",
        "pchembl_value": 6.5,
        "assay_description": "anti-inflammatory analgesic NSAID",
        "organism": "Homo sapiens",
        "max_phase": 4,
        "source": "anchor",
    },
    {
        "molecule_id": "CHEMBL112",
        "name": "Acetaminophen",
        "canonical_smiles": "CC(=O)Nc1ccc(O)cc1",
        "target_name": "Prostaglandin pathway analgesic target neighborhood",
        "target_class": "enzyme",
        "bioactivity_type": "EC50",
        "pchembl_value": 4.8,
        "assay_description": "analgesic antipyretic anti-inflammatory neighborhood",
        "organism": "Homo sapiens",
        "max_phase": 4,
        "source": "anchor",
    },
    {
        "molecule_id": "CHEMBL113",
        "name": "Caffeine",
        "canonical_smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "target_name": "Adenosine receptor",
        "target_class": "gpcr",
        "bioactivity_type": "Ki",
        "pchembl_value": 5.4,
        "assay_description": "adenosine receptor antagonist stimulant",
        "organism": "Homo sapiens",
        "max_phase": 4,
        "source": "anchor",
    },
    {
        "molecule_id": "CHEMBL1431",
        "name": "Metformin",
        "canonical_smiles": "CN(C)C(=N)NC(=N)N",
        "target_name": "AMP-activated protein kinase metabolic pathway",
        "target_class": "enzyme",
        "bioactivity_type": "activity",
        "pchembl_value": 4.5,
        "assay_description": "metabolic diabetes pathway",
        "organism": "Homo sapiens",
        "max_phase": 4,
        "source": "anchor",
    },
    {
        "molecule_id": "CHEMBL1464",
        "name": "Warfarin",
        "canonical_smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
        "target_name": "Vitamin K epoxide reductase",
        "target_class": "enzyme",
        "bioactivity_type": "IC50",
        "pchembl_value": 7.0,
        "assay_description": "anticoagulant vitamin K reductase inhibitor",
        "organism": "Homo sapiens",
        "max_phase": 4,
        "source": "anchor",
    },
    {
        "molecule_id": "CHEMBL1487",
        "name": "Atorvastatin-like anchor",
        "canonical_smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(C(=O)O)cn1Cc1ccc(F)cc1",
        "target_name": "HMG-CoA reductase",
        "target_class": "enzyme",
        "bioactivity_type": "IC50",
        "pchembl_value": 8.0,
        "assay_description": "cholesterol biosynthesis reductase inhibitor",
        "organism": "Homo sapiens",
        "max_phase": 4,
        "source": "anchor",
    },
    {
        "molecule_id": "CHEMBL139",
        "name": "Diclofenac",
        "canonical_smiles": "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
        "target_name": "Cyclooxygenase prostaglandin pathway",
        "target_class": "enzyme",
        "bioactivity_type": "IC50",
        "pchembl_value": 6.9,
        "assay_description": "anti-inflammatory NSAID",
        "organism": "Homo sapiens",
        "max_phase": 4,
        "source": "anchor",
    },
    {
        "molecule_id": "CHEMBL27",
        "name": "Morphine",
        "canonical_smiles": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
        "target_name": "Mu opioid receptor",
        "target_class": "gpcr",
        "bioactivity_type": "Ki",
        "pchembl_value": 8.2,
        "assay_description": "opioid receptor agonist analgesic",
        "organism": "Homo sapiens",
        "max_phase": 4,
        "source": "anchor",
    },
]


def infer_target_class(target_name: str) -> str:
    text = (target_name or "").lower()
    if any(word in text for word in ["channel", "sodium", "potassium", "calcium", "chloride"]):
        return "ion channel"
    if any(word in text for word in ["transporter", "pump"]):
        return "transporter"
    if any(word in text for word in ["adrenergic", "dopamine", "serotonin", "adenosine", "opioid", "histamine", "muscarinic", "receptor"]):
        return "gpcr"
    if any(word in text for word in ["kinase", "protease", "reductase", "cyclooxygenase", "heparanase", "synthase", "enzyme", "phosphatase"]):
        return "enzyme"
    if any(word in text for word in ["estrogen", "androgen", "glucocorticoid", "nuclear"]):
        return "nuclear receptor"
    return "other"


def fetch_chembl_activities(max_records: int) -> list[dict[str, Any]]:
    """Fetch a compact ChEMBL activity slice for the hackathon-sized index."""
    if config.DISABLE_CHEMBL:
        return []
    url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    page_size = 200
    activities: list[dict[str, Any]] = []
    seen_activity_ids: set[int] = set()
    try:
        for offset in range(0, max(max_records * 3, page_size), page_size):
            params = {
                "limit": page_size,
                "offset": offset,
                "pchembl_value__isnull": "false",
                "standard_units": "nM",
                "target_organism": "Homo sapiens",
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            page = response.json().get("activities", [])
            if not page:
                break
            for activity in page:
                activity_id = int(activity.get("activity_id") or 0)
                if activity_id in seen_activity_ids:
                    continue
                seen_activity_ids.add(activity_id)
                activities.append(activity)
            if len(activities) >= max_records * 2:
                break
        config.RAW_CHEMBL_PATH.write_text(json.dumps(activities, indent=2), encoding="utf-8")
    except Exception:
        if not activities:
            return []

    records: list[dict[str, Any]] = []
    seen_smiles: set[str] = set()
    for row in activities:
        smiles = row.get("canonical_smiles")
        if not smiles:
            continue
        try:
            canonical = canonicalize_smiles(smiles)
        except Exception:
            continue
        if canonical in seen_smiles:
            continue
        seen_smiles.add(canonical)
        target_name = row.get("target_pref_name") or "Unknown target"
        records.append(
            {
                "molecule_id": row.get("molecule_chembl_id") or f"CHEMBL_ACTIVITY_{len(records)}",
                "name": row.get("molecule_pref_name") or row.get("molecule_chembl_id") or f"ChEMBL molecule {len(records) + 1}",
                "canonical_smiles": canonical,
                "target_name": target_name,
                "target_class": infer_target_class(target_name),
                "bioactivity_type": row.get("standard_type") or row.get("type") or "activity",
                "standard_value": _safe_float(row.get("standard_value")),
                "standard_units": row.get("standard_units"),
                "pchembl_value": _safe_float(row.get("pchembl_value")),
                "assay_description": row.get("assay_description") or "",
                "organism": row.get("target_organism") or "Homo sapiens",
                "document_year": row.get("document_year"),
                "max_phase": 0,
                "source": "chembl_activity",
            }
        )
        if len(records) >= max_records:
            break
    return records


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_or_build_records(max_records: int, force: bool = False) -> list[dict[str, Any]]:
    """Load cached processed molecules or rebuild descriptors/vectors from source data."""
    if config.PROCESSED_PATH.exists() and not force:
        return json.loads(config.PROCESSED_PATH.read_text(encoding="utf-8"))

    raw = _merge_anchor_records(fetch_chembl_activities(max_records), max_records)
    processed = _process_records(raw[:max_records])
    config.PROCESSED_PATH.write_text(json.dumps(processed, indent=2), encoding="utf-8")
    return processed


def _merge_anchor_records(fetched: list[dict[str, Any]], max_records: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ANCHORS + fetched:
        try:
            canonical = canonicalize_smiles(row["canonical_smiles"])
        except Exception:
            continue
        if canonical in seen:
            continue
        copy = dict(row)
        copy["canonical_smiles"] = canonical
        merged.append(copy)
        seen.add(canonical)
        if len(merged) >= max_records:
            return merged
    return _expand_fallback_if_needed(merged, max_records)


def _expand_fallback_if_needed(records: list[dict[str, Any]], max_records: int) -> list[dict[str, Any]]:
    # If ChEMBL is unavailable, generate deterministic demo analogs so the
    # application remains usable offline and during live judging.
    if len(records) >= min(max_records, 40):
        return records
    rng = random.Random(42)
    templates = records[:]
    suffixes = ["analog", "screen", "neighborhood", "fragment", "outlier"]
    for index in range(len(records), min(max_records, 120)):
        base = dict(rng.choice(templates))
        base["molecule_id"] = f"DEMO{index:04d}"
        base["name"] = f"{base['name']} {rng.choice(suffixes)} {index}"
        base["source"] = "demo_fallback"
        base["pchembl_value"] = round(float(base.get("pchembl_value") or 5.0) + rng.uniform(-1.2, 1.2), 2)
        records.append(base)
    return records


def _process_records(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    structure_vectors: list[list[float]] = []
    for index, row in enumerate(raw_records, start=1):
        try:
            # This is where raw source rows become Qdrant-ready molecules:
            # descriptors, dense vectors, sparse fingerprint bits, and payload.
            canonical = canonicalize_smiles(row["canonical_smiles"])
            descriptors = compute_descriptors(canonical)
            dense_structure = structure_vector(canonical)
            sparse_indices, sparse_values = sparse_fingerprint(canonical)
        except Exception:
            continue
        payload = {
            "point_id": index,
            "molecule_id": row.get("molecule_id") or f"MOL{index}",
            "name": row.get("name") or row.get("molecule_id") or f"Molecule {index}",
            "canonical_smiles": canonical,
            "target_name": row.get("target_name") or "Unknown target",
            "target_class": row.get("target_class") or infer_target_class(row.get("target_name") or ""),
            "bioactivity_type": row.get("bioactivity_type") or "activity",
            "standard_value": row.get("standard_value"),
            "standard_units": row.get("standard_units"),
            "pchembl_value": row.get("pchembl_value"),
            "assay_description": row.get("assay_description") or "",
            "organism": row.get("organism") or "Homo sapiens",
            "document_year": row.get("document_year"),
            "max_phase": row.get("max_phase", 0) or 0,
            "source": row.get("source") or "unknown",
            **descriptors,
        }
        payload["toxicity_flag"] = toxicity_flag(descriptors)
        payload["bioactivity_text"] = bioactivity_text(payload)
        payload["structure_vector"] = dense_structure
        payload["bioactivity_vector"] = text_vector(payload["bioactivity_text"])
        payload["sparse_indices"] = sparse_indices
        payload["sparse_values"] = sparse_values
        payloads.append(payload)
        structure_vectors.append(dense_structure)

    coords = _compute_map_coordinates(structure_vectors)
    crowdedness = _compute_crowdedness(coords)
    for payload, (x, y), density in zip(payloads, coords, crowdedness, strict=False):
        payload["umap_x"] = round(float(x), 4)
        payload["umap_y"] = round(float(y), 4)
        payload["crowdedness_score"] = round(float(density["score"]), 3)
        payload["crowdedness_label"] = density["label"]
    return payloads


def _compute_map_coordinates(vectors: list[list[float]]) -> list[tuple[float, float]]:
    # The map is a lightweight 2D projection of structure vectors for visual
    # exploration. It is for navigation, not a scientific UMAP claim.
    if not vectors:
        return []
    arr = np.array(vectors, dtype=np.float32)
    if len(vectors) == 1:
        return [(0.0, 0.0)]
    n_components = min(2, len(vectors) - 1)
    reduced = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(arr)
    if n_components == 1:
        reduced = np.column_stack([reduced[:, 0], np.zeros(len(vectors))])
    scaled = MinMaxScaler(feature_range=(-100, 100)).fit_transform(reduced)
    return [(float(x), float(y)) for x, y in scaled]


def _compute_crowdedness(coords: list[tuple[float, float]]) -> list[dict[str, Any]]:
    # Crowdedness estimates local map density so candidates can be labeled as
    # known-region, border-region, or sparse-region.
    if not coords:
        return []
    arr = np.array(coords, dtype=np.float32)
    densities: list[float] = []
    for point in arr:
        distances = np.linalg.norm(arr - point, axis=1)
        kth = min(12, max(1, len(distances) - 1))
        densities.append(float(np.mean(np.sort(distances)[1 : kth + 1])))
    low = float(np.percentile(densities, 33))
    high = float(np.percentile(densities, 66))
    results = []
    for density in densities:
        if density <= low:
            label = "known-region"
        elif density <= high:
            label = "border-region"
        else:
            label = "sparse-region"
        results.append({"score": 1.0 / (1.0 + density), "label": label})
    return results
