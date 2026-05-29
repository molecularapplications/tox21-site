#!/usr/bin/env python3
"""
Fetch the Tox21 leaderboard standings from Hugging Face and write data.json.

Source: the ml-jku leaderboard reads its results from the dataset
`ml-jku/tox21-results` (the "test" split), the same one its Space loads. We read
that dataset and reshape each row into the schema index.html expects.

Two read paths, tried in order:
  A) datasets-server REST API  -> works with no auth if the dataset is public
  B) the `datasets` library     -> handles a gated dataset when HF_TOKEN is set

Output (data.json):
{
  "updated": "2026-05-29T02:30:00Z",
  "source": "ml-jku/tox21-results",
  "methods": [
    {"method": "...", "note": "...", "type": "dl", "year": 2017,
     "auc": 0.842, "targets": {"NR-AR": 0.84, ...}}
  ]
}
"""

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

RESULTS_DATASET = "ml-jku/tox21-results"
SPLIT = "test"
OUT = "data.json"

# The 12 Tox21 assay endpoints (hyphenated keys, matching the Space's config).
TASK_KEYS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

TYPE_HINTS = {
    "deeptox": "dl", "snn": "dl", "self-normaliz": "dl", "neural": "dl",
    "chemprop": "dl", "mpnn": "dl", "gin": "dl", "graph": "dl", "tabpfn": "dl",
    "transformer": "dl", "xlstm": "dl", "cnn": "dl",
    "random forest": "classical", "forest": "classical", "xgboost": "classical",
    "boost": "classical", "svm": "classical", "logistic": "classical",
    "gpt": "llm", "llm": "llm", "llama": "llm", "qwen": "llm", "mistral": "llm",
}


def _first(row, *names):
    """Return the first present, non-empty value among candidate keys."""
    for n in names:
        if n in row and row[n] not in (None, "", "nan"):
            return row[n]
    return None


def _num(v):
    try:
        f = float(v)
        return f if f == f else None  # drop NaN
    except (TypeError, ValueError):
        return None


def classify(name, row):
    pre = _first(row, "Zero-shot (y/n)", "zero_shot", "zeroshot")
    if pre and str(pre).strip().lower() in ("y", "yes", "true", "1"):
        return "llm"
    low = (name or "").lower()
    for hint, kind in TYPE_HINTS.items():
        if hint in low:
            return kind
    return "dl"


def year_of(row):
    raw = _first(row, "Date Added", "date_added", "Date", "date", "year")
    if raw is None:
        return None
    s = str(raw)
    for token in s.replace("/", "-").replace(".", "-").split("-"):
        if token.strip().isdigit() and len(token.strip()) == 4:
            return int(token.strip())
    return None


def per_task(row):
    out = {}
    for t in TASK_KEYS:
        v = _first(
            row,
            "ROC_AUC_" + t, t, "roc_auc_" + t,
            "ROC_AUC_" + t.replace("-", "_"), t.replace("-", "_"),
        )
        n = _num(v)
        if n is not None:
            out[t] = round(n, 3)
    return out


def normalize(row):
    name = _first(row, "Model", "model", "model_name", "name", "method") or "Unknown"
    avg = _num(_first(
        row, "Avg. AUC", "average_score", "avg_auc", "average_auc",
        "mean_auc", "Average AUC", "avg_score",
    ))
    note = _first(row, "Model Description", "model_description", "description")
    org = _first(row, "Organization", "organization", "org")
    if not note and org:
        note = str(org)
    out = {
        "method": str(name),
        "note": str(note) if note else "",
        "type": classify(str(name), row),
        "year": year_of(row),
        "auc": round(avg, 3) if avg is not None else None,
    }
    tgt = per_task(row)
    if tgt:
        out["targets"] = tgt
        if out["auc"] is None and len(tgt) == len(TASK_KEYS):
            out["auc"] = round(sum(tgt.values()) / len(tgt), 3)
    return out


def fetch_rows():
    # --- Path A: datasets-server REST (public, no auth, CORS-friendly) ---
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={RESULTS_DATASET}&config=default&split={SPLIT}&offset=0&length=100"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tox21-site"})
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.load(r)
        rows = [row["row"] for row in payload.get("rows", [])]
        if rows:
            print(f"Read {len(rows)} rows via datasets-server.")
            return rows
    except Exception as e:
        print(f"datasets-server path failed ({e}); trying the datasets library.")

    # --- Path B: datasets library (handles a gated dataset with HF_TOKEN) ---
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN") or None  # empty/absent secret -> anonymous read
    ds = load_dataset(RESULTS_DATASET, split=SPLIT, token=token)
    rows = list(ds)
    print(f"Read {len(rows)} rows via datasets library (token: {'yes' if token else 'no'}).")
    return rows


def main():
    rows = fetch_rows()
    if rows:
        print("First row keys:", sorted(rows[0].keys()))
    methods = [normalize(r) for r in rows]
    methods = [m for m in methods if m.get("auc") is not None]
    methods.sort(key=lambda m: m["auc"], reverse=True)

    data = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "source": RESULTS_DATASET,
        "methods": methods,
    }
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {len(methods)} methods to {OUT}.")
    if not methods:
        print("WARNING: no methods had a usable AUC — the column mapping may need a tweak.")
        print("Paste the 'First row keys' line above to me and I'll adjust normalize().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
