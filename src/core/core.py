#!/usr/bin/env python3
# Core: config, IO, embeddings/chat, FAISS index, retrieval.

import os, json, math
from typing import Any, Dict, List

import numpy as np
from src.config import config
from src.core import ollama, documents

os.makedirs(config.DATA_DIR, exist_ok=True)
os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)

# ---------- Optional deps (fail fast with clear message) ----------
try:
    import faiss  # type: ignore
except Exception as e:
    raise SystemExit(f"FAISS not installed. pip install: pip install faiss-cpu | {e}")

try:
    from pypdf import PdfReader  # type: ignore
except Exception as e:
    raise SystemExit(f"pypdf not installed. pip install: pip install pypdf | {e}")


# ----------------------- Embedding & Chat -----------------------
def _l2_normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]

# ----------------------- Index Build / Load -----------------------
_index = None
_meta: List[Dict[str, Any]] = []

def index_exists() -> bool:
    return os.path.exists(config.INDEX_PATH) and os.path.exists(config.META_PATH)

def index_loaded() -> bool:
    return _index is not None and bool(_meta)

def meta_len() -> int:
    try:
        with open(config.META_PATH, "r", encoding="utf-8") as f:
            return len(json.load(f))
    except Exception:
        return 0

def save_index(index, meta):
    faiss.write_index(index, config.INDEX_PATH)
    with open(config.META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_index_meta():
    global _index, _meta
    if not index_exists():
        raise RuntimeError("Index not found. Build it first from the UI.")
    _index = faiss.read_index(config.INDEX_PATH)
    with open(config.META_PATH, "r", encoding="utf-8") as f:
        _meta = json.load(f)

def ensure_loaded():
    if not index_loaded():
        load_index_meta()

def build_index_from_data(data_dir: str, chunk_size: int, overlap: int, embed_model: str):
    docs = documents.load_documents_from_dir(data_dir)
    if not docs:
        raise RuntimeError(f"No supported documents found under {data_dir}")

    chunks = []
    for src, text in docs:
        if src.endswith(".csv") or "#row" in src:
            chunks.append({"source": src, "chunk_id": 0, "text": text})
        else:
            parts = documents.chunk_text(text, chunk_size, overlap)
            for i, part in enumerate(parts):
                chunks.append({"source": src, "chunk_id": i, "text": part})

    vectors = []
    for c in chunks:
        v = ollama.ollama_embed(c["text"], model=embed_model)
        v = _l2_normalize(v)
        vectors.append(v)

    xb = np.array(vectors, dtype="float32")
    dim = xb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized inner product
    index.add(xb)

    save_index(index, chunks)
    load_index_meta()  # refresh in-memory
    return {"docs": len(docs), "chunks": len(chunks), "dim": dim}

# ----------------------- Retrieval -----------------------
def retrieve(query: str, top_k: int):
    ensure_loaded()
    qv = ollama.ollama_embed(query)
    qv = _l2_normalize(qv)
    xq = np.array([qv], dtype="float32")
    scores, idxs = _index.search(xq, top_k)
    hits = []
    for i, s in zip(idxs[0], scores[0]):
        if i == -1:
            continue
        item = dict(_meta[i])
        item["score"] = float(s)
        hits.append(item)
    return hits

def build_context(hits):
    blocks = []
    for h in hits:
        src = h.get("source", "unknown")
        cidx = h.get("chunk_id", 0)
        text = h.get("text", "")
        header = f"[{os.path.basename(str(src))}#chunk{cidx}] (score={h.get('score',0):.3f})"
        blocks.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(blocks)



