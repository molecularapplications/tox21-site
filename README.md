# Tox21 Leaderboard site

A static site that displays the reproducible Tox21 Challenge leaderboard, refreshed daily
from Hugging Face. No server to run.

```
index.html                     the whole front-end (open it in a browser right now)
data.json                      generated daily; the page reads this. ships absent → placeholder data shown
scripts/fetch_data.py          pulls the leaderboard from HF, writes data.json
.github/workflows/update.yml   runs the fetcher once a day and commits data.json
```

## Run it locally
Just open `index.html`. With no `data.json` present it shows clearly-labelled placeholder
standings so you can see the layout. Add a real `data.json` (same folder) and it switches over.

## Deploy (free, ~5 min)
1. Push this folder to a GitHub repo.
2. **Cloudflare Pages** (recommended) or **GitHub Pages**: point it at the repo root, no build command, output = `/`.
3. Add your biotech domain in the host's custom-domain settings → it gives you a DNS record (a `CNAME`) to add at your registrar.
4. The daily Action keeps `data.json` fresh; the site is otherwise pure static.

## The one open item: the data source
`scripts/fetch_data.py` is wired but parked, because the exact place the ml-jku leaderboard
stores its results needs a 30-second look:

- Open https://huggingface.co/spaces/ml-jku/tox21_leaderboard → **Files** tab.
- If there's a committed results file or a referenced dataset repo → set `RESULTS_DATASET`
  (Strategy A, preferred).
- Otherwise find the Gradio function that loads the table → set `GRADIO_API_NAME` (Strategy B).

Tell me what the Files tab / API shows and I'll finish `normalize()` so the columns map cleanly.
Until then the script no-ops and the site keeps showing placeholders — nothing breaks.
