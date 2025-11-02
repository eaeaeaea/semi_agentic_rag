import os

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL   = os.environ.get("LLM_MODEL",   "llama3.1:8b")
EMB_MODEL   = os.environ.get("EMBED_MODEL", "nomic-embed-text")

BASE_DIR      = os.environ.get("BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join(BASE_DIR, "artifacts"))
INDEX_PATH    = os.path.join(ARTIFACTS_DIR, "faiss.index")
META_PATH     = os.path.join(ARTIFACTS_DIR, "meta.json")
STATIC_DIR    = os.path.join(BASE_DIR, "static")

ALLOWED_EXTS = {".txt", ".md", ".markdown", ".pdf", ".csv"}

TOP_K_DEFAULT   = int(os.environ.get("TOP_K", "5"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
LLM_NUM_CTX     = int(os.environ.get("LLM_NUM_CTX", "8192"))

SYSTEM_PROMPT_RAG = (
    "You answer strictly from the provided context. "
    "If the context is insufficient, say you don't have enough information. "
    "Cite sources inline like [filename#chunkN]. Be concise."
)
SYSTEM_PROMPT_BARE = "You are a helpful assistant. Answer concisely. If unsure, say so."