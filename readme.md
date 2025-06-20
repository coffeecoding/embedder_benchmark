# Embedding Model Benchmark

Focussed on retrieval performance of embedding models by comparing retrieved document ranking to baseline model (`text-embedding-3-large`).

**Important note**:
This repo works only in tandem with an embedder backend hosted locally.

## How to run

1. Create following directories

```
./datasets/1/
./datasets/queries/
```

2. Add some documents (json, html, txt) to the dataset folder `./datasets/1/` (or whichever index you are at for identification)

3. Activate venv and install requirements

```
venv\Scripts\activate.bat
pip install -r requirements.txt
```

4. Ensure environment variable `OPENAI_API_KEY` is set.

5. Index dataset

`python indexer.py --dataset 1`

6. Generate test queries via
`python query_creator.py --dataset <subfolder_id_e.g_1>`

7. Run benchmarker
`python run_benchmark.py --dataset 1 --queries 1_gpt-4-1-mini_1_250620-192019.jsonl`
