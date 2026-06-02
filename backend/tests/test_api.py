from __future__ import annotations

import os
from pathlib import Path

os.environ["MOLSPACE_DISABLE_CHEMBL"] = "1"
os.environ["MOLSPACE_MAX_RECORDS"] = "40"

from fastapi.testclient import TestClient

from app import config
from app.chemistry import compute_descriptors, resolve_name_or_smiles
from app.main import app


def test_aspirin_descriptors_are_present() -> None:
    smiles = resolve_name_or_smiles("aspirin")
    descriptors = compute_descriptors(smiles)
    assert descriptors["molecular_weight"] > 100
    assert descriptors["qed"] > 0
    assert descriptors["lipinski_violations"] == 0


def test_invalid_smiles_rejected() -> None:
    client = TestClient(app)
    response = client.post("/molecule/resolve", json={"query": "not-a-real-smiles"})
    assert response.status_code == 400


def test_build_search_discover_compare(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.setattr(config, "LOCAL_QDRANT_PATH", tmp_path / "qdrant")
    monkeypatch.setattr(config, "PROCESSED_PATH", tmp_path / "processed_molecules.json")
    monkeypatch.setattr(config, "RAW_CHEMBL_PATH", tmp_path / "raw_chembl_activities.json")

    client = TestClient(app)
    build = client.post("/index/build", json={"max_records": 40, "force": True})
    assert build.status_code == 200
    assert build.json()["count"] >= 10

    summary = client.get("/qdrant/summary").json()
    assert "structure" in [item["name"] for item in summary["named_vectors"]]
    assert "fingerprint_sparse" in [item["name"] for item in summary["sparse_vectors"]]

    search = client.post("/search", json={"query": "aspirin", "mode": "hybrid", "limit": 8})
    assert search.status_code == 200
    data = search.json()
    assert data["results"]
    assert data["qdrant"]["collection"] == "molspace_molecules"

    first_id = data["results"][0]["molecule"]["molecule_id"]
    discovery = client.post(
        "/discover",
        json={"query": "aspirin", "positive_ids": [first_id], "negative_ids": [], "limit": 5},
    )
    assert discovery.status_code == 200
    assert discovery.json()["results"]

    compare = client.post("/compare", json={"left_id": "CHEMBL25", "right_id": "CHEMBL521"})
    assert compare.status_code == 200
    assert "structure_similarity" in compare.json()
