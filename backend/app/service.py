from __future__ import annotations

import csv
import io
import json
import math
from typing import Any

import numpy as np
from fastapi import HTTPException
from qdrant_client import models

from . import config
from .chemistry import (
    compute_descriptors,
    resolve_name_or_smiles,
    sparse_fingerprint,
    structure_vector,
    tanimoto_similarity,
)
from .dataset import load_or_build_records
from .qdrant_store import (
    collection_exists,
    count_points,
    get_client,
    query_dense,
    query_hybrid,
    recreate_collection,
    scroll_payloads,
    upsert_molecules,
)
from .schemas import DiscoveryRequest, IndexStatus, SearchFilters, SearchRequest
from .vectorize import bioactivity_text, cosine_similarity, text_vector


def build_index(max_records: int = config.DEFAULT_MAX_RECORDS, force: bool = False) -> dict[str, Any]:
    client, mode = get_client()
    if force or not collection_exists(client):
        recreate_collection(client)
    records = load_or_build_records(max_records=max_records, force=force)
    upsert_molecules(client, records)
    return {
        "collection": config.COLLECTION_NAME,
        "mode": mode,
        "count": count_points(client),
        "vectors": ["structure", "bioactivity"],
        "sparse_vectors": ["fingerprint_sparse"],
        "source_cache": str(config.PROCESSED_PATH),
    }


def status() -> IndexStatus:
    client, mode = get_client()
    exists = collection_exists(client)
    return IndexStatus(
        collection=config.COLLECTION_NAME,
        exists=exists,
        count=count_points(client) if exists else 0,
        mode=mode,
        processed_cache=config.PROCESSED_PATH.exists(),
    )


def ensure_index_ready() -> None:
    current = status()
    if current.exists and current.count > 0:
        return
    build_index(max_records=min(config.DEFAULT_MAX_RECORDS, 1000), force=True)


def resolve_molecule(query: str) -> dict[str, Any]:
    ensure_index_ready()
    payloads = _all_payloads()
    text = query.strip().lower()
    for payload in payloads:
        if text in {
            str(payload.get("molecule_id", "")).lower(),
            str(payload.get("name", "")).lower(),
        }:
            return {"kind": "indexed", "payload": payload, "descriptors": _descriptor_subset(payload)}
    for payload in payloads:
        if text and text in str(payload.get("name", "")).lower():
            return {"kind": "indexed", "payload": payload, "descriptors": _descriptor_subset(payload)}

    try:
        canonical = resolve_name_or_smiles(query)
        descriptors = compute_descriptors(canonical)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "kind": "query",
        "payload": {
            "molecule_id": "QUERY",
            "point_id": -1,
            "name": query,
            "canonical_smiles": canonical,
            "target_name": "Query molecule",
            "target_class": "unknown",
            "toxicity_flag": "unknown",
            "crowdedness_label": "query",
            "umap_x": 0,
            "umap_y": 0,
            **descriptors,
        },
        "descriptors": descriptors,
    }


def search(request: SearchRequest) -> dict[str, Any]:
    ensure_index_ready()
    resolved = resolve_molecule(request.query)
    query_payload = resolved["payload"]
    query_smiles = query_payload["canonical_smiles"]
    query_structure = structure_vector(query_smiles)
    sparse_indices, sparse_values = sparse_fingerprint(query_smiles)
    query_bioactivity = _bio_vector_for_query(query_payload)

    client, qdrant_mode = get_client()
    candidate_limit = max(request.limit * 6, 80)
    exclude = [int(query_payload["point_id"])] if int(query_payload.get("point_id", -1)) > 0 else []
    if request.mode == "structure":
        hits = query_dense(client, query_structure, "structure", request.filters, candidate_limit, offset_ids=exclude)
        qdrant_query = "named-vector structure"
    elif request.mode == "bioactivity":
        hits = query_dense(client, query_bioactivity, "bioactivity", request.filters, candidate_limit, offset_ids=exclude)
        qdrant_query = "named-vector bioactivity"
    else:
        hits = query_hybrid(client, query_structure, query_bioactivity, sparse_indices, sparse_values, request.filters, candidate_limit)
        hits = [hit for hit in hits if int(hit.id) not in exclude]
        qdrant_query = "hybrid RRF over structure, bioactivity, fingerprint_sparse"

    results = _rank_hits(hits, query_payload, query_structure, query_bioactivity, request.filters, request.mode)
    limited = results[: request.limit]
    return {
        "query": {
            "name": query_payload.get("name"),
            "molecule_id": query_payload.get("molecule_id"),
            "canonical_smiles": query_smiles,
            "descriptors": resolved["descriptors"],
        },
        "mode": request.mode,
        "results": limited,
        "qdrant": _qdrant_panel(
            qdrant_mode=qdrant_mode,
            query=qdrant_query,
            filters=request.filters,
            candidates=len(hits),
            returned=len(limited),
        ),
    }


def discover(request: DiscoveryRequest) -> dict[str, Any]:
    ensure_index_ready()
    resolved = resolve_molecule(request.query)
    query_payload = resolved["payload"]
    query_smiles = query_payload["canonical_smiles"]
    base_structure = np.array(structure_vector(query_smiles), dtype=np.float32)
    base_bioactivity = np.array(_bio_vector_for_query(query_payload), dtype=np.float32)

    id_map = _payload_map()
    positives = [id_map[mol_id] for mol_id in request.positive_ids if mol_id in id_map]
    negatives = [id_map[mol_id] for mol_id in request.negative_ids if mol_id in id_map]

    steered_structure = _steer_vector(base_structure, positives, negatives, "structure").tolist()
    steered_bioactivity = _steer_vector(base_bioactivity, positives, negatives, "bioactivity").tolist()
    sparse_indices, sparse_values = sparse_fingerprint(query_smiles)
    client, qdrant_mode = get_client()

    hits = _try_context_discovery(client, positives, negatives, request.filters, request.limit * 6)
    qdrant_query = "Qdrant discovery/context query"
    if not hits:
        hits = query_hybrid(client, steered_structure, steered_bioactivity, sparse_indices, sparse_values, request.filters, request.limit * 6)
        qdrant_query = "steered hybrid fallback over Qdrant named vectors"

    exclude_ids = {int(query_payload.get("point_id", -1))}
    hits = [hit for hit in hits if int(hit.id) not in exclude_ids]
    ranked = _rank_hits(hits, query_payload, base_structure.tolist(), base_bioactivity.tolist(), request.filters, "hybrid")
    limited = ranked[: request.limit]
    return {
        "query": {
            "name": query_payload.get("name"),
            "molecule_id": query_payload.get("molecule_id"),
            "canonical_smiles": query_smiles,
            "positive_ids": request.positive_ids,
            "negative_ids": request.negative_ids,
        },
        "mode": "discovery",
        "results": limited,
        "qdrant": _qdrant_panel(
            qdrant_mode=qdrant_mode,
            query=qdrant_query,
            filters=request.filters,
            candidates=len(hits),
            returned=len(limited),
        )
        | {"positive_examples": request.positive_ids, "negative_examples": request.negative_ids},
    }


def _try_context_discovery(client: Any, positives: list[dict[str, Any]], negatives: list[dict[str, Any]], filters: SearchFilters, limit: int) -> list[Any]:
    if not positives:
        return []
    try:
        positive_ids = [int(item["point_id"]) for item in positives]
        negative_ids = [int(item["point_id"]) for item in negatives]
        context = [
            models.ContextExamplePair(positive=positive_id, negative=negative_id)
            for positive_id, negative_id in zip(positive_ids, negative_ids or positive_ids, strict=False)
        ]
        response = client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=models.DiscoverQuery(discover=models.DiscoverInput(context=context, target=positive_ids[0])),
            query_filter=__import__("app.qdrant_store", fromlist=["build_filter"]).build_filter(filters),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return list(response.points)
    except Exception:
        return []


def compare(left_id: str, right_id: str) -> dict[str, Any]:
    ensure_index_ready()
    payloads = _payload_map()
    if left_id not in payloads or right_id not in payloads:
        raise HTTPException(status_code=404, detail="One or both molecules were not found")
    left = payloads[left_id]
    right = payloads[right_id]
    left_bio = _bio_vector_for_query(left)
    right_bio = _bio_vector_for_query(right)
    descriptor_fields = ["molecular_weight", "logp", "tpsa", "hbd", "hba", "rotatable_bonds", "qed", "lipinski_violations"]
    deltas = {field: round(float(right[field]) - float(left[field]), 3) for field in descriptor_fields}
    map_distance = math.dist([left["umap_x"], left["umap_y"]], [right["umap_x"], right["umap_y"]])
    return {
        "left": _molecule_summary(left),
        "right": _molecule_summary(right),
        "structure_similarity": tanimoto_similarity(left["canonical_smiles"], right["canonical_smiles"]),
        "bioactivity_similarity": cosine_similarity(left_bio, right_bio),
        "descriptor_deltas_right_minus_left": deltas,
        "shared_target_class": left.get("target_class") == right.get("target_class"),
        "shared_target_name": left.get("target_name") == right.get("target_name"),
        "map_distance": round(map_distance, 3),
    }


def map_points() -> list[dict[str, Any]]:
    ensure_index_ready()
    payloads = _all_payloads()
    return [
        {
            "molecule_id": payload["molecule_id"],
            "name": payload["name"],
            "x": payload["umap_x"],
            "y": payload["umap_y"],
            "target_class": payload["target_class"],
            "qed": payload["qed"],
            "toxicity_flag": payload["toxicity_flag"],
            "crowdedness_label": payload.get("crowdedness_label"),
            "max_phase": payload.get("max_phase", 0),
        }
        for payload in payloads
    ]


def qdrant_summary() -> dict[str, Any]:
    current = status()
    return {
        "collection": current.collection,
        "exists": current.exists,
        "count": current.count,
        "mode": current.mode,
        "named_vectors": [
            {"name": "structure", "size": config.STRUCTURE_DIM, "meaning": "RDKit Morgan fingerprint"},
            {"name": "bioactivity", "size": config.BIOACTIVITY_DIM, "meaning": "target/activity text embedding"},
        ],
        "sparse_vectors": [{"name": "fingerprint_sparse", "meaning": "active Morgan fingerprint bits"}],
        "payload_indexes": ["target_class", "toxicity_flag", "qed", "molecular_weight", "logp", "lipinski_violations"],
    }


def export_shortlist(molecule_ids: list[str], export_format: str) -> tuple[str, str]:
    payloads = _payload_map()
    selected = [payloads[molecule_id] for molecule_id in molecule_ids if molecule_id in payloads]
    if export_format == "json":
        return "application/json", json.dumps(selected, indent=2)
    output = io.StringIO()
    fields = [
        "molecule_id",
        "name",
        "canonical_smiles",
        "target_name",
        "target_class",
        "molecular_weight",
        "logp",
        "qed",
        "lipinski_violations",
        "toxicity_flag",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in selected:
        writer.writerow({field: row.get(field) for field in fields})
    return "text/csv", output.getvalue()


def _rank_hits(
    hits: list[Any],
    query_payload: dict[str, Any],
    query_structure: list[float],
    query_bioactivity: list[float],
    filters: SearchFilters,
    mode: str,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    query_target = query_payload.get("target_class")
    for hit in hits:
        payload = hit.payload or {}
        structure_sim = tanimoto_similarity(query_payload["canonical_smiles"], payload["canonical_smiles"])
        bio_sim = cosine_similarity(query_bioactivity, _bio_vector_for_query(payload))
        qdrant_score = round(float(hit.score or 0), 4)
        toxicity_penalty = {"low": 0.0, "medium": 0.35, "high": 0.9}.get(payload.get("toxicity_flag"), 0.2)
        same_target_penalty = 0.0
        if mode == "analog" or filters.different_target_class:
            same_target_penalty = 0.18 if payload.get("target_class") == query_target else -0.06
        if mode == "structure":
            final_score = 0.70 * structure_sim + 0.15 * bio_sim + 0.15 * float(payload.get("qed", 0))
        elif mode == "bioactivity":
            final_score = 0.15 * structure_sim + 0.70 * bio_sim + 0.15 * float(payload.get("qed", 0))
        else:
            final_score = 0.45 * structure_sim + 0.35 * bio_sim + 0.15 * float(payload.get("qed", 0)) - 0.05 * toxicity_penalty
        final_score -= same_target_penalty
        conflict = _conflict_badge(structure_sim, bio_sim)
        ranked.append(
            {
                "molecule": _molecule_summary(payload),
                "final_score": round(max(final_score, 0.0), 4),
                "qdrant_score": qdrant_score,
                "structure_similarity": structure_sim,
                "bioactivity_similarity": bio_sim,
                "conflict_badge": conflict,
                "why": _why(payload, structure_sim, bio_sim, conflict),
            }
        )
    ranked.sort(key=lambda item: item["final_score"], reverse=True)
    return ranked


def _conflict_badge(structure_sim: float, bio_sim: float) -> str | None:
    if structure_sim >= 0.35 and bio_sim < 0.22:
        return "structure-close bioactivity-distant"
    if bio_sim >= 0.48 and structure_sim < 0.2:
        return "bioactivity-close structure-novel"
    return None


def _why(payload: dict[str, Any], structure_sim: float, bio_sim: float, conflict: str | None) -> list[str]:
    reasons = [
        f"{structure_sim:.2f} RDKit Tanimoto structure similarity",
        f"{bio_sim:.2f} bioactivity-vector similarity",
        f"QED {float(payload.get('qed', 0)):.2f} with {payload.get('lipinski_violations', 0)} Lipinski violations",
        f"{payload.get('target_class', 'unknown')} target neighborhood",
        f"{payload.get('crowdedness_label', 'map-region')} on the chemical-space map",
    ]
    if payload.get("toxicity_flag") != "low":
        reasons.append(f"{payload.get('toxicity_flag')} heuristic toxicity flag")
    if conflict:
        reasons.append(f"flagged as {conflict}")
    return reasons


def _qdrant_panel(qdrant_mode: str, query: str, filters: SearchFilters, candidates: int, returned: int) -> dict[str, Any]:
    return {
        "engine": "Qdrant",
        "mode": qdrant_mode,
        "collection": config.COLLECTION_NAME,
        "named_vectors": ["structure", "bioactivity"],
        "sparse_vectors": ["fingerprint_sparse"],
        "query": query,
        "filters": filters.model_dump(),
        "candidates_retrieved": candidates,
        "final_reranked_results": returned,
        "rerank_formula": "0.45*structure + 0.35*bioactivity + 0.15*QED - toxicity/target penalties",
    }


def _steer_vector(base: np.ndarray, positives: list[dict[str, Any]], negatives: list[dict[str, Any]], vector_key: str) -> np.ndarray:
    vector_field = f"{vector_key}_vector"
    pos_vectors = [np.array(item[vector_field], dtype=np.float32) for item in positives if vector_field in item]
    neg_vectors = [np.array(item[vector_field], dtype=np.float32) for item in negatives if vector_field in item]
    result = base.copy()
    if pos_vectors:
        result += 0.75 * np.mean(pos_vectors, axis=0)
    if neg_vectors:
        result -= 0.55 * np.mean(neg_vectors, axis=0)
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
    return result


def _bio_vector_for_query(payload: dict[str, Any]) -> list[float]:
    if "bioactivity_vector" in payload:
        return payload["bioactivity_vector"]
    return text_vector(bioactivity_text(payload))


def _descriptor_subset(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in ["molecular_weight", "logp", "tpsa", "hbd", "hba", "rotatable_bonds", "qed", "lipinski_violations"]
    }


def _molecule_summary(payload: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "molecule_id",
        "point_id",
        "name",
        "canonical_smiles",
        "target_name",
        "target_class",
        "toxicity_flag",
        "crowdedness_label",
        "molecular_weight",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "qed",
        "lipinski_violations",
        "umap_x",
        "umap_y",
        "max_phase",
    ]
    return {field: payload.get(field) for field in fields}


def _all_payloads() -> list[dict[str, Any]]:
    if config.PROCESSED_PATH.exists():
        return json.loads(config.PROCESSED_PATH.read_text(encoding="utf-8"))
    client, _mode = get_client()
    return scroll_payloads(client)


def _payload_map() -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for payload in _all_payloads():
        mapping[str(payload.get("molecule_id"))] = payload
        mapping[str(payload.get("name", "")).lower()] = payload
    return mapping

