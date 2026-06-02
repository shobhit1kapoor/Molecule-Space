from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


SearchMode = Literal["structure", "bioactivity", "hybrid", "analog"]


class BuildIndexRequest(BaseModel):
    max_records: int = Field(default=1000, ge=10, le=5000)
    force: bool = False


class ResolveRequest(BaseModel):
    query: str


class SearchFilters(BaseModel):
    molecular_weight_min: float | None = None
    molecular_weight_max: float | None = 500
    logp_min: float | None = None
    logp_max: float | None = 5
    qed_min: float | None = 0.35
    lipinski_max: int | None = 1
    target_class: str | None = None
    exclude_high_toxicity: bool = True
    different_target_class: bool = False


class SearchRequest(BaseModel):
    query: str = "aspirin"
    mode: SearchMode = "hybrid"
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=20, ge=1, le=100)


class DiscoveryRequest(BaseModel):
    query: str = "aspirin"
    positive_ids: list[str] = Field(default_factory=list)
    negative_ids: list[str] = Field(default_factory=list)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=20, ge=1, le=100)


class CompareRequest(BaseModel):
    left_id: str
    right_id: str


class ShortlistExportRequest(BaseModel):
    molecule_ids: list[str]
    format: Literal["json", "csv"] = "json"


class MoleculeSummary(BaseModel):
    molecule_id: str
    point_id: int
    name: str
    canonical_smiles: str
    target_name: str
    target_class: str
    toxicity_flag: str
    crowdedness_label: str | None = None
    molecular_weight: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    qed: float
    lipinski_violations: int
    umap_x: float
    umap_y: float
    max_phase: int | None = None


class SearchResult(BaseModel):
    molecule: MoleculeSummary
    final_score: float
    qdrant_score: float
    structure_similarity: float
    bioactivity_similarity: float
    conflict_badge: str | None = None
    why: list[str]


class SearchResponse(BaseModel):
    query: dict[str, Any]
    mode: str
    results: list[SearchResult]
    qdrant: dict[str, Any]


class IndexStatus(BaseModel):
    collection: str
    exists: bool
    count: int
    mode: str
    processed_cache: bool

