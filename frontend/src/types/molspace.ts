export type SearchMode = "structure" | "bioactivity" | "hybrid" | "analog";

export interface SearchFilters {
  molecular_weight_min?: number | null;
  molecular_weight_max?: number | null;
  logp_min?: number | null;
  logp_max?: number | null;
  qed_min?: number | null;
  lipinski_max?: number | null;
  target_class?: string | null;
  exclude_high_toxicity: boolean;
  different_target_class: boolean;
}

export interface MoleculeSummary {
  molecule_id: string;
  point_id: number;
  name: string;
  canonical_smiles: string;
  target_name: string;
  target_class: string;
  toxicity_flag: string;
  crowdedness_label?: string;
  molecular_weight: number;
  logp: number;
  tpsa: number;
  hbd: number;
  hba: number;
  rotatable_bonds: number;
  qed: number;
  lipinski_violations: number;
  umap_x: number;
  umap_y: number;
  max_phase?: number;
}

export interface SearchResult {
  molecule: MoleculeSummary;
  final_score: number;
  qdrant_score: number;
  structure_similarity: number;
  bioactivity_similarity: number;
  conflict_badge?: string | null;
  why: string[];
}

export interface SearchResponse {
  query: {
    name: string;
    molecule_id: string;
    canonical_smiles: string;
    descriptors: Record<string, number>;
    positive_ids?: string[];
    negative_ids?: string[];
  };
  mode: string;
  results: SearchResult[];
  qdrant: Record<string, unknown>;
}

export interface MapPoint {
  molecule_id: string;
  name: string;
  x: number;
  y: number;
  target_class: string;
  qed: number;
  toxicity_flag: string;
  crowdedness_label?: string;
  max_phase?: number;
}

export interface IndexStatus {
  collection: string;
  exists: boolean;
  count: number;
  mode: string;
  processed_cache: boolean;
}

export interface CompareResponse {
  left: MoleculeSummary;
  right: MoleculeSummary;
  structure_similarity: number;
  bioactivity_similarity: number;
  descriptor_deltas_right_minus_left: Record<string, number>;
  shared_target_class: boolean;
  shared_target_name: boolean;
  map_distance: number;
}

