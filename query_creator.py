#!/usr/bin/env python
"""
Generate ~1 query per paragraph for each doc in a dataset.
Saves queries as JSON-Lines:
    <dataset_id>_<llm_model>_<prompt_idx>_yyMMdd-HHmmss.jsonl
"""
import argparse, datetime as dt, glob, json, os, pathlib, re, sys
from typing import List
from openai import OpenAI
from tqdm import tqdm                # ⇦ NEW
client = OpenAI()                     # uses OPENAI_API_KEY env-var

# ---------- reusable helpers ----------
def read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)               # docs/<id>
    ap.add_argument("--llm", default="gpt-4.1-mini")          # LLM to query
    ap.add_argument("--prompt-idx", type=int, default=0)      # choose sys-prompt
    args = ap.parse_args()

    prompt_template = SYS_PROMPTS[args.prompt_idx]

    ds_dir  = os.path.join("datasets", "docs", args.dataset)
    out_id  = f"{args.dataset}_{args.llm.replace('.','-')}_{args.prompt_idx}_" \
              f"{dt.datetime.now().strftime('%y%m%d-%H%M%S')}"
    out_path = os.path.join("datasets", "queries", f"{out_id}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # collect candidate files once so tqdm can show total
    files = [
        fp for fp in glob.glob(os.path.join(ds_dir, "**/*"), recursive=True)
        if pathlib.Path(fp).suffix.lower() in {".txt", ".md", ".html", ".htm"}
    ]

    with open(out_path, "w", encoding="utf-8") as sink:
        for fp in tqdm(files, desc="Generating queries"):
            doc = read(fp)
            system_msg = prompt_template.format(doc=doc)
            rsp = client.responses.create(model=args.llm, input=system_msg)
            try:
                queries = json.loads(rsp.output_text)
            except json.JSONDecodeError:
                sys.stderr.write(f"☠️  {fp} produced non-JSON output\n")
                continue
            for q in queries:
                sink.write(json.dumps(
                    {"doc": os.path.relpath(fp, ds_dir), "query": q},
                    ensure_ascii=False) + "\n")
    print(f"Wrote → {out_path}")

if __name__ == "__main__":
    main()
