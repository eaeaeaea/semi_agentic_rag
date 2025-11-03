import csv
import os

from pypdf import PdfReader

from src.config import config


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)
        return "\n".join(texts)
    except Exception as e:
        print(f"[WARN] Failed to read PDF {path}: {e}")
        return ""

def read_csv_as_rows(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader, start=1):
            # Join all key-value pairs dynamically
            row_text = "; ".join(f"{k}:{v}" for k, v in r.items())
            src_id = f"{path}#row{i}"
            rows.append((src_id, row_text))
    return rows

def chunk_text(text: str, chunk_size: int, overlap: int):
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def load_documents_from_dir(data_dir: str):
    docs = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            p = os.path.join(root, fn)
            ext = os.path.splitext(p)[1].lower()
            if ext not in config.ALLOWED_EXTS:
                continue
            if ext == ".pdf":
                text = read_pdf(p).strip()
                if text:
                    docs.append((p, text))
            elif ext == ".csv":
                docs.extend(read_csv_as_rows(p))
            else:
                text = read_txt(p).strip()
                if text:
                    docs.append((p, text))
    return docs

def list_data_files():
    files = []
    for root, _, fns in os.walk(config.DATA_DIR):
        for fn in fns:
            p = os.path.join(root, fn)
            files.append({"path": os.path.relpath(p, config.DATA_DIR), "bytes": os.path.getsize(p)})
    return files