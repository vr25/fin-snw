# Narrative Graphs: AI-Hype Stock Co-Movement

Code, data, and analysis artifacts for studying financial narrative networks
and stock-return co-movement. The manuscript and paper-only documentation
are intentionally excluded from this distribution.

## Contents

- `scripts/`: data collection, relation extraction, graph construction,
  evaluation, label validation, agreement scoring, and visualization code.
- `data/raw/`: local price data, GDELT metadata, article/post datasets, and
  classification inputs.
- `data/processed/`: relation labels, original label snapshots, sample datasets,
  and ablation results.
- `data/relabel/`: relabeling prompts, input items, and classification outputs.
- `data/validation/`: validation inputs, model outputs, and human annotation
  materials. `human/key.json` contains reference labels; do not give it to
  annotators conducting a blinded evaluation.
- `analysis/metrics/`: saved analysis and agreement metrics.
- `analysis/figures/`: generated visualizations, including PDF figure exports
  (not the manuscript).

## Setup

Run commands from the extracted repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/requirements.txt
python -m pip install numpy networkx scipy scikit-learn matplotlib trafilatura
```

The original requirements file covers collection dependencies; the additional
packages cover analysis, plotting, and article extraction.

## Working with the bundled data

The archive includes existing inputs and results; collection is not required
to inspect them. The following commands regenerate graph/evaluation outputs
and analysis metrics, overwriting their corresponding saved results:

```bash
python scripts/build_graph.py
python scripts/evaluate.py
python scripts/compute_paper_stats.py
python scripts/label_agreement.py
python scripts/make_hero_network.py
python scripts/make_trend_figure.py
```

Despite its name, `compute_paper_stats.py` computes analysis statistics; it does
not require manuscript files.

Collection scripts are also included. Live Reddit and StockTwits collection
depends on platform access; credentials are read from environment variables in
`scripts/config.py`. API-based relation extraction additionally requires the
`anthropic` package and `ANTHROPIC_API_KEY`. Without that key, extraction uses
a deterministic keyword fallback intended only for pipeline testing. Do not
rerun extraction over the bundled labels unless you intend to replace them.