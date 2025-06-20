#!/usr/bin/env python
"""
Benchmark embedding models against a baseline with a selectable metric
(ndcg or spearman).

Output: rpt_<dataset>_<metric>_top-<top-k>_yyMMdd-hhmmss.csv
"""
import argparse, csv, datetime as dt, json, os, sys
from typing import List
import faiss, numpy as np, requests
from tqdm import tqdm

# -------------------- config --------------------
ENDPOINT   = "http://localhost:8001/embed_batch"
HEADERS    = {"content-type": "application/json"}
BATCH_SIZE = 32
TOP_K      = 10
BASELINE   = "text-embedding-3-large"

MODELS = [
    BASELINE,
    "text-embedding-3-small",
    "text-embedding-ada-002",
    "uae-large-v1",
    "snowflake-s",
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
        r = requests.post(ENDPOINT, headers=HEADERS, data=json.dumps(payload))
        r.raise_for_status()
        vecs.extend(r.json()["embeddings"])
    return np.asarray(vecs, dtype="float32")

# -------------------- metrics -------------------
def ndcg(baseline_ids: List[int], cand_ids: List[int]) -> float:
    gains = [1 if cid in baseline_ids else 0 for cid in cand_ids]
    dcg   = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    idcg  = sum(1 / np.log2(i + 2) for i in range(len(baseline_ids)))
    return dcg / idcg if idcg else 0.0

def spearman(baseline_ids: List[int], model_ids: List[int]) -> float:
    n   = len(baseline_ids)
    pos = {v: i for i, v in enumerate(model_ids)}
    dist = sum(abs(i - pos.get(v, n)) for i, v in enumerate(baseline_ids))
    maxd = n * n / 2
    return 1 - dist / maxd

METRICS = {"ndcg": ndcg, "spearman": spearman}

def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)

# -------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset id (docs folder)")
    ap.add_argument("--queries", required=True, help="queries .jsonl filename")
    ap.add_argument("--top_k", type=int, default=TOP_K, help="K nearest neighbours")
    ap.add_argument("--metric", choices=METRICS.keys(), default="ndcg",
                    help="ranking-similarity metric to use")
    args = ap.parse_args()
    metric_fn = METRICS[args.metric]

    # load queries
    with open(f"./datasets/queries/{args.queries}", encoding="utf-8") as f:
        queries = [json.loads(line)["query"] for line in f]
    print(f"{len(queries)} queries loaded")

    # load indices
    indices = {m: load_index(m, args.dataset) for m in MODELS}

    # baseline neighbours
    base_vecs = embed(queries, BASELINE)
    # L2 normalize vectors since we use cosine similarity
    base_vecs = normalize(base_vecs)
    _, I_base = indices[BASELINE].search(base_vecs, args.top_k)

    # score models
    results = []
    for model in tqdm(MODELS, desc="Models"):
        if model == BASELINE:
            results.append((model, 1.0))
            continue
        vecs = embed(queries, model)
        # L2 normalize vectors since we use cosine similarity
        vecs = normalize(vecs)
        _, I_cand = indices[model].search(vecs, args.top_k)
        scores = [metric_fn(list(I_base[q]), list(I_cand[q]))
                  for q in range(len(queries))]
        results.append((model, float(np.mean(scores))))

    # save CSV
    ts = dt.datetime.now().strftime("%y%m%d-%H%M%S")
    out_csv = f"./reports/rpt_{args.dataset}_{args.metric}_top-{args.top_k}_{ts}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerows([["model_id", "score"], *results])
    print(f"Results saved → {out_csv}")

if __name__ == "__main__":
    main()
