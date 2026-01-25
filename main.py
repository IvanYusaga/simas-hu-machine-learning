import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from search2 import search, reload_documents

app = FastAPI(title="SIMAS-HU Search API")


def get_allowed_origins():
    _LOCAL_ORIGINS = [
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    _env_origins = os.getenv("CORS_ALLOW_ORIGINS", "")
    _extra_origins = [o.strip() for o in _env_origins.split(",") if o.strip()]
    return _LOCAL_ORIGINS + [o for o in _extra_origins if o not in _LOCAL_ORIGINS]


allow_origins = get_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "SIMAS-HU Search API", "status": "running"}


@app.get("/search")
def search_api(
    q: str = Query(..., description="Search query"),
    category: str = Query("magang", description="Category: magang or skripsi"),
    limit: int = Query(20, description="Maximum number of results"),
):
    """
    Search for laporan titles

    - **q**: Search query string
    - **category**: "magang" for Laporan Magang/PKL, "skripsi" for Laporan Skripsi
    - **limit**: Maximum number of results to return (default: 20)
    """
    if category not in ["magang", "skripsi"]:
        category = "magang"

    results = search(q, category=category, limit=limit)
    return {"query": q, "category": category, "total": len(results), "results": results}


@app.post("/reload")
def reload_api():
    """Reload documents from database and rebuild search index"""
    reload_documents()
    return {"status": "ok", "message": "Documents reloaded successfully"}
