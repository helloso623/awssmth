#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
uv python install 3.12.14
uv venv --python "$PWD/.python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12" .venv-neuron
uv pip install --python .venv-neuron/bin/python \
  --index-strategy unsafe-best-match \
  --extra-index-url https://pip.repos.neuron.amazonaws.com \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r training-requirements.lock
# Only the unchanged authors' archive annotations need Tinker; no service auth is used.
uv pip install --python .venv-neuron/bin/python tinker==0.33.1
