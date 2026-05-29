#!/usr/bin/env python3
"""
Fetch the Tox21 leaderboard from Hugging Face and write data.json for the site.

The ml-jku leaderboard is a Gradio Space backed by a public results dataset, so there
are two viable strategies. Strategy A (a results dataset) is preferred when it exists,
because it's stable and CORS-free. Strategy B (the Gradio API) is the fallback.

>>> WHICH ONE TO USE IS THE ONE THING TO CONFIRM AGAINST THE LIVE SPACE <<<
Open https://huggingface.co/spaces/ml-jku/tox21_leaderboard -> "Files" tab and look for
either a committed results file (e.g. results.json / a .parquet) or a referenced dataset
repo. Plug its id into RESULTS_DATASET below, or wire Strategy B to the right api_name.

Output schema (data.json):
{
  "updated": "2026-05-28T02:00:00Z",
  "methods": [
    {"method": "DeepTox", "note": "...", "type": "dl", "year": 2015,
     "auc": 0.846, "targets": {"NR-AR": 0.84, ...}}  # targets optional
  ]
}
"""

import json
import sys
from datetime import datetime, timezone

OUT = "data.json"

# --- Strategy A: a public results dataset (preferred). Set when confirmed. -------------
RESULTS_DATASET = None          # e.g. "ml-jku/tox21_leaderboard_results"
RESULTS_CONFIG = "default"
RESULTS_SPLIT = "train"

# --- Strategy B: call the Gradio Space directly (fallback). ----------------------------
SPACE_ID = "ml-jku/tox21_leaderboard"
GRADIO_API_NAME = None          # e.g. "/load_leaderboard" — confirm from the Space

TYPE_MAP = {  # optional: classify known methods for the UI's Type column
    "deeptox": "dl", "snn": "dl", "self-normalizing": "dl", "chemprop": "dl",
    "gin": "dl", "tabpfn": "dl", "random forest": "classical", "xgboost": "classical",
    "gpt": "llm", "llm": "llm",
}


def classify(name: str) -> str:
    n = (name or "").lower()
    for k, v in TYPE_MAP.items():
        if k in n:
            return v
    return "dl"


def from_dataset():
    """Strategy A: read a results dataset via datasets-server (no auth needed if public)."""
    import urllib.request
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={RESULTS_DATASET}&config={RESULTS_CONFIG}&split={RESULTS_SPLIT}&offset=0&length=100"
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.load(r)
    rows = [row["row"] for row in payload.get("rows", [])]
    return [normalize(r) for r in rows]


def from_gradio():
    """Strategy B: hit the Space's Gradio API. Requires: pip install gradio_client."""
    from gradio_client import Client
    client = Client(SPACE_ID)
    result = client.predict(api_name=GRADIO_API_NAME)
    # `result` shape depends on the Space; commonly a dict with "data" rows, or a dataframe.
    # Adapt this once the real response is known.
    rows = result.get("data", result) if isinstance(result, dict) else result
    return [normalize(r) for r in rows]


def normalize(row) -> dict:
    """Map a source row (dict) onto the site schema. Adjust keys to the real columns."""
    name = row.get("method") or row.get("model") or row.get("name") or "Unknown"
    auc = row.get("auc") or row.get("mean_auc") or row.get("score")
    out = {
        "method": name,
        "note": row.get("note") or row.get("description") or "",
        "type": row.get("type") or classify(name),
        "year": row.get("year"),
        "auc": float(auc) if auc is not None else None,
    }
    # per-assay AUCs, if present, enable the click-to-expand view
    targets = {k: v for k, v in row.items() if k.startswith(("NR-", "SR-"))}
    if targets:
        out["targets"] = {k: float(v) for k, v in targets.items()}
    return out


def main():
    methods = []
    if RESULTS_DATASET:
        methods = from_dataset()
    elif GRADIO_API_NAME:
        methods = from_gradio()
    else:
        print("No data source configured yet — set RESULTS_DATASET or GRADIO_API_NAME.", file=sys.stderr)
        print("Leaving data.json untouched so the site keeps showing placeholder values.", file=sys.stderr)
        return 0

    methods = [m for m in methods if m.get("auc") is not None]
    methods.sort(key=lambda m: m["auc"], reverse=True)

    with open(OUT, "w") as f:
        json.dump({"updated": datetime.now(timezone.utc).isoformat(), "methods": methods}, f, indent=2)
    print(f"Wrote {len(methods)} methods to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
