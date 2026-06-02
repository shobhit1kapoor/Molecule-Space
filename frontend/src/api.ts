import type { CompareResponse, IndexStatus, MapPoint, SearchFilters, SearchMode, SearchResponse } from "./types/molspace";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // Shared JSON request helper keeps endpoint calls typed and consistent.
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json() as Promise<T>;
}

export function getStatus() {
  return request<IndexStatus>("/index/status");
}

export function buildIndex(maxRecords = 1000, force = true) {
  return request<Record<string, unknown>>("/index/build", {
    method: "POST",
    body: JSON.stringify({ max_records: maxRecords, force }),
  });
}

export function searchMolecules(query: string, mode: SearchMode, filters: SearchFilters, limit = 24) {
  // Search mode controls which Qdrant vectors are used on the backend.
  return request<SearchResponse>("/search", {
    method: "POST",
    body: JSON.stringify({ query, mode, filters, limit }),
  });
}

export function discoverMolecules(query: string, positiveIds: string[], negativeIds: string[], filters: SearchFilters, limit = 24) {
  // Positive and negative examples power the discovery steering workflow.
  return request<SearchResponse>("/discover", {
    method: "POST",
    body: JSON.stringify({ query, positive_ids: positiveIds, negative_ids: negativeIds, filters, limit }),
  });
}

export function getMapPoints() {
  return request<MapPoint[]>("/map/points");
}

export function compareMolecules(leftId: string, rightId: string) {
  return request<CompareResponse>("/compare", {
    method: "POST",
    body: JSON.stringify({ left_id: leftId, right_id: rightId }),
  });
}

export async function exportShortlist(ids: string[], format: "json" | "csv") {
  const response = await fetch(`${API_BASE}/shortlist/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ molecule_ids: ids, format }),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.blob();
}
