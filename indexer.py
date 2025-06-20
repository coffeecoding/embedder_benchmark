#!/usr/bin/env python
"""
Build local Faiss vector stores for every model in --models
using the documents in datasets/docs/<dataset_id>/.
All chunking + micro-service call parameters stay identical
across models to enable fair comparison.
"""
import argparse, os, json, pathlib, glob
from typing import List
import requests, faiss, numpy as np
import html2text
from tqdm import tqdm

CHUNK_SIZE = 512
OVERLAP    = 64
ENDPOINT   = "http://localhost:8001/embed_batch"
HEADERS    = {"content-type": "application/json"}

MODELS = [
    "text-embedding-3-large",
    "text-embedding-3-small",
    "uae-large-v1",
    "snowflake-s",
    "text-embedding-ada-002"
]

def read_docs(ds_path: str) -> List[str]:
    """Load all readable text from a dataset folder (deterministic order)."""
    docs = []
    for fp in sorted(glob.glob(os.path.join(ds_path, "**/*"), recursive=True)):
        suffix = pathlib.Path(fp).suffix.lower()
        if suffix in {".txt", ".md", ".html", ".htm"}:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    if suffix in {".html", ".htm"}:
                        docs.append(html2text.html2text(f.read()))
                    else:
                        docs.append(f.read())
    return docs

def chunk(text: str) -> List[str]:
    """Fixed-size chunking with overlap."""
    text = text.strip().replace("\n", " ")
    return [text[i : i + CHUNK_SIZE]
            for i in range(0, len(text), CHUNK_SIZE - OVERLAP)]

def embed(texts: List[str], model: str, batch: int = 32) -> np.ndarray:
    """Call the micro-service in equal-sized batches (progress shown)."""
    vecs = []
    for i in tqdm(range(0, len(texts), batch),
                  desc=f"Embedding ({model})", leave=False):
        payload = {"texts": texts[i : i + batch], "model": model}
        r = requests.post(ENDPOINT, headers=HEADERS, data=json.dumps(payload))
        r.raise_for_status()
        vecs.extend(r.json()["embeddings"])
    return np.asarray(vecs, dtype="float32")

def build_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatIP(vectors.shape[1]) # FlatIP should use cosine sim (inner product)
    index.add(vectors)
    return index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="folder name inside datasets/docs")
    args = ap.parse_args()

    ds_folder = os.path.join("datasets", "docs", args.dataset)
    assert os.path.isdir(ds_folder), f"{ds_folder} not found"
    os.makedirs("vdbs", exist_ok=True)

    print(f"Reading documents from {ds_folder} …")
    raw_docs = read_docs(ds_folder)
    print(f"  loaded {len(raw_docs)} docs")

    # Chunk with progress bar
    chunks = []
    for doc in tqdm(raw_docs, desc="Chunking documents"):
        chunks.extend(chunk(doc))
    print(f"  produced {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={OVERLAP})")

    # Build one index per model
    for model in tqdm(MODELS, desc="Models"):
        vecs  = embed(chunks, model)
        index = build_index(vecs)
        out_path = os.path.join("vdbs", f"{model}_{args.dataset}.faiss")
        faiss.write_index(index, out_path)
        print(f"  wrote {out_path}  (dim={vecs.shape[1]})")

if __name__ == "__main__":
    main()
