#!/usr/bin/env python3
"""
FastAPI app (routes). Static UI served from ./static/index.html
"""

import os, time, shutil
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import config, core
import semi_agentic

app = FastAPI(title="Local RAG Comparator UI + Index Builder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve ./static under /static
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

# ----------------------- UI -----------------------
@app.get("/")
def index_page():
    html_path = os.path.join(config.STATIC_DIR, "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h3>UI not found</h3><p>Expected: static/index.html next to api.py.</p>", status_code=404)
    return FileResponse(html_path)

# ----------------------- API -----------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "index_loaded": core.index_loaded(),
        "index_exists": core.index_exists(),
        "meta_len": core.meta_len(),
        "paths": {"data": config.DATA_DIR, "index": config.INDEX_PATH, "meta": config.META_PATH},
        "models": {"llm": config.LLM_MODEL, "embed": config.EMB_MODEL},
    }

@app.get("/api/list")
def list_files():
    files = core.list_data_files()
    return {"count": len(files), "files": files}

@app.delete("/api/data")
def clear_data():
    if os.path.isdir(config.DATA_DIR):
        shutil.rmtree(config.DATA_DIR)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    if os.path.isdir(config.ARTIFACTS_DIR):
        shutil.rmtree(config.ARTIFACTS_DIR)
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    return {"cleared": True}

@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved = []
    for uf in files:
        name = os.path.basename(uf.filename or "")
        ext = os.path.splitext(name)[1].lower()
        if not name or ext not in config.ALLOWED_EXTS:
            continue
        dest = os.path.join(config.DATA_DIR, name)
        content = await uf.read()
        with open(dest, "wb") as out:
            out.write(content)
        saved.append({"name": name, "bytes": len(content)})
    if not saved:
        raise HTTPException(status_code=400, detail="No files saved (empty selection or unsupported extensions)")
    return {"saved": saved, "data_dir": config.DATA_DIR}

@app.post("/api/build")
def api_build(chunk_size: int = Form(1200), overlap: int = Form(200), embed_model: str = Form(config.EMB_MODEL)):
    try:
        t0 = time.time()
        stats = core.build_index_from_data(config.DATA_DIR, chunk_size, overlap, embed_model)
        t1 = time.time()
        return {"ok": True, "stats": stats, "ms": int((t1 - t0) * 1000)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
def api_query(payload: Dict[str, Any]):
    try:
        question = (payload.get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="'question' is required")
        top_k = int(payload.get("top_k") or config.TOP_K_DEFAULT)
        return semi_agentic.hybrid_rag_mcp(question, top_k)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))