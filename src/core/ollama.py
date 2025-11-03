import requests

from src.config import config

def ollama_embed(text: str, model: str = config.EMB_MODEL):
    url = f"{config.OLLAMA_HOST}/api/embeddings"
    r = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Embeddings error: {r.text}")
    data = r.json()
    emb = data.get("embedding")
    if not emb:
        raise RuntimeError("Embeddings response missing 'embedding'")
    return emb

def ollama_chat(system_prompt: str, user_prompt: str) -> str:
    url = f"{config.OLLAMA_HOST}/api/chat"
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "options": {"temperature": config.LLM_TEMPERATURE, "num_ctx": config.LLM_NUM_CTX},
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Chat error: {r.text}")
    data = r.json()
    return (data.get("message") or {}).get("content", "").strip()
