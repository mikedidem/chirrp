"""
CHIRRP backend — Hydro-AI participatory water-resources planning.

Two layers, per the framework paper:
  * Semantic layer: LLM chat-to-model translation + RAG policy assistant
  * Physical layer: PINN groundwater surrogate with MODFLOW validation
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from chirrp.backend.controller import (
    participant_controller,
    pinn_controller,
    rag_controller,
    session_controller,
)

# When deployed as a single container, the built Angular app is copied here and
# served by FastAPI (one origin → no CORS, relative apiUrl). Unset locally, so
# local dev (UI on :4200, API on :8000) is unaffected.
STATIC_DIR = os.environ.get("STATIC_DIR")
# Extra browser origins allowed via env (comma-separated), for split deploys.
_EXTRA_ORIGINS = [o.strip() for o in os.environ.get("EXTRA_CORS_ORIGINS", "").split(",") if o.strip()]

app = FastAPI(
    title="CHIRRP — Hydro-AI Participatory Planning",
    description=(
        "LLM semantic layer + physics-informed neural network surrogate "
        "for participatory groundwater scenario planning."
    ),
)

origins = [
    "http://localhost:4200",   # Angular dev server
    "http://127.0.0.1:4200",
    "http://127.0.0.1:8000",
] + _EXTRA_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pinn_controller.router)
app.include_router(rag_controller.router)
app.include_router(session_controller.router)
app.include_router(participant_controller.router)


@app.on_event("startup")
def warm_engine():
    """Load the PINN once so the first request is already millisecond-fast."""
    pinn_controller.get_engine()


@app.get("/healthz")
def healthz():
    return {"status": "API is running"}


@app.get("/")
def root():
    if STATIC_DIR and os.path.isfile(os.path.join(STATIC_DIR, "index.html")):
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
    return {"status": "API is running"}


# SPA + asset serving for the single-container deploy. Registered LAST so it
# never shadows the API routers above. Only active when STATIC_DIR is set.
if STATIC_DIR and os.path.isdir(STATIC_DIR):
    @app.get("/{full_path:path}")
    def spa(full_path: str):
        # 1) an actual built file (e.g. main-<hash>.js, styles.css, favicon)
        f = os.path.join(STATIC_DIR, full_path)
        if full_path and os.path.isfile(f):
            return FileResponse(f)
        # 2) a prerendered route (e.g. studio/index.html)
        route_index = os.path.join(STATIC_DIR, full_path, "index.html")
        if full_path and os.path.isfile(route_index):
            return FileResponse(route_index)
        # 3) SPA fallback → client router takes over
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
