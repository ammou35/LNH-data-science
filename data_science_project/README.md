## Quick start

    uv sync                      # create .venv and install everything
    cp .env.example .env         # optional: choose where raw data is cached
    uv run pytest                # offline tests, no network needed
    uv run jupyter lab           # open notebooks/01_data_acquisition.ipynb

Download the 2016-17 to 2023-24 play-by-play data (slow once, cached afterwards):

    uv run download-nhl          # options: --first 2016 --last 2023 --data-dir PATH

Raw JSON goes to `$NHL_DATA_DIR` (default `data/raw/`) and is git-ignored. **Never commit the data.**

## Layout

    ift6758/data/nhl.py          Task 1: download, cache and load play-by-play data
    ift6758/features/            Task 3: tidy data (to write)
    ift6758/visualizations/      Tasks 2, 4, 5: plots and widgets (to write)
    ift6758/llm/                 [IFT6758 only] Qwen loader and RAG code
    llm_artifacts/               [IFT6758 only] doc snapshot, retrieved excerpts, generated code
    notebooks/                   one notebook per task
    figures/nhl_rink.png         rink image for the shot plots
    tests/                       pytest suite

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage Python, dependencies and the virtual environment.
Install it by following the [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

Then, from the root of the repository:

    uv sync

This creates a `.venv/` folder with a compatible Python version (see `requires-python` in `pyproject.toml`), installs the dependencies locked in `uv.lock`, and installs the `ift6758` package in editable mode (changes to the code are picked up without reinstalling).

Run anything inside the environment with `uv run`:

    uv run jupyter lab
    uv run python my_script.py

Jupyter launched this way already uses the project environment, so there is no kernel to register.
If you use another editor, point its Python interpreter / notebook kernel to the `.venv/` environment.

### Managing dependencies

Add or remove packages with `uv` rather than `pip`, so that `pyproject.toml` and `uv.lock` stay up to date:

    uv add scikit-learn          # runtime dependency
    uv add --dev pytest          # development-only dependency
    uv remove scikit-learn

Commit both `pyproject.toml` and `uv.lock`; teammates then just run `uv sync` to get the exact same environment.

### THE BELOW SHOULD BE UPDATED IF NEEDED
## LLM setup

All LLM tasks must use [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
A loading function is provided in [`ift6758/llm/qwen.py`](./ift6758/llm/qwen.py):

```python
from ift6758.llm import load_model

model, tokenizer = load_model()
```

Running it with a GPU is recommended.
See the [model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) for how to prompt the model and the recommended generation settings.

### Running locally

The LLM dependencies (`torch`, `transformers`, ...) are an optional extra, so install them with:

    uv sync --extra llm

To check that everything works (the first run downloads ~8 GB of weights to the Hugging Face cache):

    uv run python -m ift6758.llm.qwen

### Running on Google Colab

The suggested environment is a Colab T4 GPU.
Colab notebooks run in Colab's own Python environment rather than uv.
[`notebooks/1.2_LLM.ipynb`](./notebooks/LLM%20setup.ipynb) installs the same package versions as `uv.lock` on Colab, clones your repository, and loads the model; use it as a starting point for your LLM notebooks.
