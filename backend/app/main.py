from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .chemistry import molecule_svg
from .schemas import (
    BuildIndexRequest,
    CompareRequest,
    DiscoveryRequest,
    ResolveRequest,
    SearchRequest,
    ShortlistExportRequest,
)
from .service import (
    build_index,
    compare,
    discover,
    export_shortlist,
    map_points,
    qdrant_summary,
    resolve_molecule,
    search,
    status,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if config.AUTO_BUILD:
        current = status()
        if not current.exists or current.count == 0:
            build_index(max_records=config.DEFAULT_MAX_RECORDS, force=True)
    yield


app = FastAPI(
    title="Molecule Space API",
    description="Qdrant-powered molecular discovery navigator",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "molecule-space"}


@app.post("/index/build")
def index_build(request: BuildIndexRequest) -> dict[str, object]:
    return build_index(max_records=request.max_records, force=request.force)


@app.get("/index/status")
def index_status() -> dict[str, object]:
    return status().model_dump()


@app.post("/molecule/resolve")
def molecule_resolve(request: ResolveRequest) -> dict[str, object]:
    return resolve_molecule(request.query)


@app.get("/molecule/{molecule_id}")
def molecule_detail(molecule_id: str) -> dict[str, object]:
    return resolve_molecule(molecule_id)


@app.get("/molecule/{molecule_id}/structure.svg")
def molecule_structure_svg(molecule_id: str) -> Response:
    resolved = resolve_molecule(molecule_id)
    svg = molecule_svg(resolved["payload"]["canonical_smiles"])
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/search")
def molecule_search(request: SearchRequest) -> dict[str, object]:
    return search(request)


@app.post("/discover")
def molecule_discover(request: DiscoveryRequest) -> dict[str, object]:
    return discover(request)


@app.post("/compare")
def molecule_compare(request: CompareRequest) -> dict[str, object]:
    return compare(request.left_id, request.right_id)


@app.get("/map/points")
def chemical_map_points() -> list[dict[str, object]]:
    return map_points()


@app.get("/qdrant/summary")
def qdrant_engine_summary() -> dict[str, object]:
    return qdrant_summary()


@app.post("/shortlist/export")
def shortlist_export(request: ShortlistExportRequest) -> Response:
    media_type, content = export_shortlist(request.molecule_ids, request.format)
    extension = "json" if request.format == "json" else "csv"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="molecule-space-shortlist.{extension}"'},
    )
