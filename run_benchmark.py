#!/usr/bin/env python
"""
Benchmark a list of embedding models against the baseline
'text-embedding-3-large'.

Output: result_yyyyMMdd-HHmmss.csv  (model_id, score)
"""
import argparse, csv, datetime as dt, glob, json, os, pathlib, sys
from typing import List, Dict
import faiss, numpy as np, requests
from tqdm import tqdm

# -------------------- config --------------------
ENDPOINT   = "http://localhost:8001/embed_batch"
HEADERS    = {"content-type": "application/json"}
BATCH_SIZE = 32
TOP_K      = 10                               # n-nearest neighbours
BASELINE   = "text-embedding-3-large"

MODELS = [
    BASELINE,                    # ← baseline must be first
    "text-embedding-3-small",
    "snowflake-s"
]

# -------------------- helpers -------------------
def load_index(model: str, dataset: str):
    path = os.path.join("vdbs", f"{model}_{dataset}.faiss")
    if not os.path.isfile(path):
        sys.exit(f"Vector DB not found: {path}")
    return faiss.read_index(path)

def embed(texts: List[str], model: str) -> np.ndarray:
    vecs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE),
                  desc=f"Embedding ({model})",
                  leave=False):
        payload = {"texts": texts[i:i + BATCH_SIZE], "model": model}
        r = requests.post(ENDPOINT, headers=HEADERS,
                          data=json.dumps(payload))
        r.raise_for_status()
        vecs.extend(r.json()["embeddings"])
    return np.asarray(vecs, dtype="float32")

# -------------------- metrics ----------------------
def ndcg(baseline_ids: List[int], cand_ids: List[int]) -> float:
    """Simple nDCG@k with binary relevance (1 if id ∈ baseline)."""
    gains = [1 if cid in baseline_ids else 0 for cid in cand_ids]
    dcg   = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    idcg  = sum(1 / np.log2(i + 2) for i in range(len(baseline_ids)))
    return dcg / idcg if idcg else 0.0

def norm_spearman(baseline_ids: List[int], model_ids: List[int]):
    """1-Spearman-footrule distance (0…1, 1 = identical)."""
    n = len(baseline_ids)
    pos = {v: i for i, v in enumerate(model_ids)}
    dist = sum(abs(i - pos.get(v, n)) for i, v in enumerate(baseline_ids))
    max_dist = n*n/2            # worst-case footrule
    return 1 - dist / max_dist

# -------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="folder name inside datasets/docs")
    ap.add_argument("--queries", required=True,
                    help="queries .jsonl filename")
    ap.add_argument("--top_k", type=int, default=TOP_K,
                    help="K nearest neighbours to compare")
    args = ap.parse_args()

    # --- load test queries ---
    with open(f"./datasets/queries/{args.queries}", encoding="utf-8") as f:
        queries = [json.loads(line)["query"] for line in f]
    print(f"{len(queries)} queries loaded")

    # --- load Faiss indices once ---
    indices = {m: load_index(m, args.dataset) for m in MODELS}

    # --- baseline neighbours (once) ---
    base_vecs = embed(queries, BASELINE)
    D_base, I_base = indices[BASELINE].search(base_vecs, args.top_k)

    # --- score every model ---
    results = []
    for model in tqdm(MODELS, desc="Models"):
        if model == BASELINE:
            results.append((model, 1.0))
            continue

        vecs = embed(queries, model)
        _, I_cand = indices[model].search(vecs, args.top_k)

        # per-query nDCG against baseline
        scores = [
            ndcg(list(I_base[q]), list(I_cand[q]))
            for q in range(len(queries))
        ]
        results.append((model, float(np.mean(scores))))

    # --- write CSV ---
    out_csv = f"rpt_{dt.datetime.now().strftime('%y%m%d-%H%M%S')}_{args.dataset}_top-{args.top_k}.csv"

    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["model_id", "score"])
        w.writerows(results)
    print(f"Results saved → {out_csv}")

if __name__ == "__main__":
    main()
