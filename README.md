# fraud-detection

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

```
fraud-detection/
├── data/          # datasets (gitignored)
├── notebooks/     # exploratory analysis
├── src/           # source modules
├── tests/         # pytest tests
└── requirements.txt
```

## Common commands

```bash
source .venv/bin/activate   # activate the environment
pytest                      # run tests
ruff check .                # lint
jupyter lab                 # exploratory notebooks
```
