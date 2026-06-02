from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qdrant_client import QdrantClient, models

from . import config
from .schemas import SearchFilters


def get_client() -> tuple[QdrantClient, str]:
    url = __import__("os").environ.get("QDRANT_URL")
    api_key = __import__("os").environ.get("QDRANT_API_KEY")
    if url:
        return QdrantClient(url=url, api_key=api_key or None, timeout=60), "cloud"
    config.LOCAL_QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(config.LOCAL_QDRANT_PATH)), "local"


def collection_exists(client: QdrantClient) -> bool:
    try:
        return bool(client.collection_exists(config.COLLECTION_NAME))
    except Exception:
        return False


def recreate_collection(client: QdrantClient) -> None:
    if collection_exists(client):
        client.delete_collection(config.COLLECTION_NAME)
    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config={
            "structure": models.VectorParams(size=config.STRUCTURE_DIM, distance=models.Distance.COSINE),
            "bioactivity": models.VectorParams(size=config.BIOACTIVITY_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "fingerprint_sparse": models.SparseVectorParams(),
        },
    )
    _create_payload_indexes(client)


def _create_payload_indexes(client: QdrantClient) -> None:
    index_specs = [
        ("target_class", models.PayloadSchemaType.KEYWORD),
        ("toxicity_flag", models.PayloadSchemaType.KEYWORD),
        ("qed", models.PayloadSchemaType.FLOAT),
        ("molecular_weight", models.PayloadSchemaType.FLOAT),
        ("logp", models.PayloadSchemaType.FLOAT),
        ("lipinski_violations", models.PayloadSchemaType.INTEGER),
    ]
    for field, schema in index_specs:
        try:
            client.create_payload_index(config.COLLECTION_NAME, field_name=field, field_schema=schema)
        except Exception:
            pass


def upsert_molecules(client: QdrantClient, records: list[dict[str, Any]], batch_size: int = 64) -> None:
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        points = []
        for record in batch:
            payload = public_payload(record)
            points.append(
                models.PointStruct(
                    id=int(record["point_id"]),
                    vector={
                        "structure": record["structure_vector"],
                        "bioactivity": record["bioactivity_vector"],
                        "fingerprint_sparse": models.SparseVector(
                            indices=record["sparse_indices"],
                            values=record["sparse_values"],
                        ),
                    },
                    payload=payload,
                )
            )
        client.upsert(collection_name=config.COLLECTION_NAME, points=points, wait=True)


def public_payload(record: dict[str, Any]) -> dict[str, Any]:
    hidden = {"structure_vector", "bioactivity_vector", "sparse_indices", "sparse_values"}
    return {key: value for key, value in record.items() if key not in hidden}


def build_filter(filters: SearchFilters | None) -> models.Filter | None:
    if filters is None:
        return None
    must: list[models.Condition] = []
    must_not: list[models.Condition] = []
    if filters.molecular_weight_min is not None or filters.molecular_weight_max is not None:
        must.append(
            models.FieldCondition(
                key="molecular_weight",
                range=models.Range(gte=filters.molecular_weight_min, lte=filters.molecular_weight_max),
            )
        )
    if filters.logp_min is not None or filters.logp_max is not None:
        must.append(models.FieldCondition(key="logp", range=models.Range(gte=filters.logp_min, lte=filters.logp_max)))
    if filters.qed_min is not None:
        must.append(models.FieldCondition(key="qed", range=models.Range(gte=filters.qed_min)))
    if filters.lipinski_max is not None:
        must.append(models.FieldCondition(key="lipinski_violations", range=models.Range(lte=filters.lipinski_max)))
    if filters.target_class:
        must.append(models.FieldCondition(key="target_class", match=models.MatchValue(value=filters.target_class)))
    if filters.exclude_high_toxicity:
        must_not.append(models.FieldCondition(key="toxicity_flag", match=models.MatchValue(value="high")))
    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def query_dense(
    client: QdrantClient,
    vector: list[float],
    using: str,
    filters: SearchFilters | None,
    limit: int,
    offset_ids: Iterable[int] | None = None,
) -> list[Any]:
    query_filter = build_filter(filters)
    must_not_ids = list(offset_ids or [])
    if must_not_ids:
        id_condition = models.HasIdCondition(has_id=must_not_ids)
        if query_filter is None:
            query_filter = models.Filter(must_not=[id_condition])
        else:
            query_filter.must_not = list(query_filter.must_not or []) + [id_condition]
    response = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=vector,
        using=using,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return list(response.points)


def query_hybrid(
    client: QdrantClient,
    structure: list[float],
    bioactivity: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    filters: SearchFilters | None,
    limit: int,
) -> list[Any]:
    query_filter = build_filter(filters)
    try:
        response = client.query_points(
            collection_name=config.COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=structure, using="structure", limit=limit * 4, filter=query_filter),
                models.Prefetch(query=bioactivity, using="bioactivity", limit=limit * 4, filter=query_filter),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                    using="fingerprint_sparse",
                    limit=limit * 4,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return list(response.points)
    except Exception:
        structure_hits = query_dense(client, structure, "structure", filters, limit * 3)
        bio_hits = query_dense(client, bioactivity, "bioactivity", filters, limit * 3)
        return reciprocal_rank_fusion([structure_hits, bio_hits], limit)


def reciprocal_rank_fusion(hit_lists: list[list[Any]], limit: int, k: int = 60) -> list[Any]:
    scored: dict[int, tuple[float, Any]] = {}
    for hits in hit_lists:
        for rank, hit in enumerate(hits, start=1):
            existing_score, _existing_hit = scored.get(int(hit.id), (0.0, hit))
            scored[int(hit.id)] = (existing_score + 1.0 / (k + rank), hit)
    fused = sorted(scored.values(), key=lambda item: item[0], reverse=True)
    results = []
    for score, hit in fused[:limit]:
        hit.score = score
        results.append(hit)
    return results


def scroll_payloads(client: QdrantClient, limit: int = 10000) -> list[dict[str, Any]]:
    next_offset = None
    payloads: list[dict[str, Any]] = []
    while True:
        points, next_offset = client.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=min(512, limit - len(payloads)),
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        payloads.extend([point.payload for point in points if point.payload])
        if next_offset is None or len(payloads) >= limit:
            break
    return payloads


def count_points(client: QdrantClient) -> int:
    if not collection_exists(client):
        return 0
    try:
        return int(client.count(config.COLLECTION_NAME, exact=True).count)
    except Exception:
        return len(scroll_payloads(client))

