# Contributing

This is a portfolio project — issues and PRs are welcome, especially around:

- Live simulator UDP ingestion (F1 24 / Assetto Corsa)
- Additional track corner name mappings
- Frontend improvements (new chart types, UI polish)
- Model improvements (steering angle reconstruction, sector-split references)

## Setup

```bash
git clone https://github.com/FK-75/f1-ai-driver-coach
cd f1-ai-driver-coach
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/train.py
```

## Code style

- Python: standard formatting, type hints preferred
- React: functional components, hooks only
- Keep docstrings on all public functions — the inline comments explain design decisions that are non-obvious

## Data

Do not commit anything to `data/` except `.gitkeep`. The FastF1 cache, trained model weights, and pickle files are all in `.gitignore` for size and licensing reasons.
