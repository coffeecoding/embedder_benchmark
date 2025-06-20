#!/usr/bin/env python
"""
Build local Faiss vector stores for every model in --models
using the documents in datasets/docs/<dataset_id>/.
All chunking + micro-service call parameters stay identical
across models to enable fair comparison.
"""
import argparse, os, json, pathlib, glob, textwrap
from typing import List
import requests, faiss, numpy as np
from tqdm import tqdm

CHUNK_SIZE   = 512          # bytes or characters – adapt once, reused for all
OVERLAP      = 64
ENDPOINT     = "http://localhost:8001/embed_batch"   # your micro-service
HEADERS      = {"content-type": "application/json"}

MODELS = [
    # "text-embedding-3-large",
    # "text-embedding-3-small",
    "snowflake-s"
]

def read_docs(ds_path: str) -> List[str]:
    """Load all readable text from a dataset folder."""
    docs = []
    for fp in glob.glob(os.path.join(ds_path, "**/*"), recursive=True):
        if pathlib.Path(fp).suffix.lower() in {".txt", ".md", ".html", ".htm"}:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                docs.append(f.read())
    return docs

def chunk(text: str) -> List[str]:
    """Very simple fixed-size chunking with overlap."""
    text = text.strip().replace("\n", " ")
    out = []
    for i in range(0, len(text), CHUNK_SIZE - OVERLAP):
        out.append(text[i:i + CHUNK_SIZE])
    return out

def embed(texts: List[str], model: str, batch: int = 32) -> np.ndarray:
    """Call the micro-service in equal-sized batches."""
    vecs = []
    for i in range(0, len(texts), batch):
        payload = {"texts": texts[i:i + batch], "model": model}
        r = requests.post(ENDPOINT, headers=HEADERS, data=json.dumps(payload))
        r.raise_for_status()
        vecs.extend(r.json()["embeddings"])
    return np.asarray(vecs, dtype="float32")

def build_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    """Return a simple L2 index (same dim as vectors)."""
    index = faiss.IndexFlatL2(vectors.shape[1])
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

    # Pre-chunk once → reused for every model (identical input)
    chunks = []
    for doc in raw_docs:
        chunks.extend(chunk(doc))
    print(f"  produced {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={OVERLAP})")

    for model in MODELS:
        print(f"\n=== {model} ===")
        vecs = embed(chunks, model)
        index = build_index(vecs)
        out_path = os.path.join("vdbs", f"{model}_{args.dataset}.faiss")
        faiss.write_index(index, out_path)
        print(f"  wrote {out_path}  (dim={vecs.shape[1]})")

if __name__ == "__main__":
    main()
