#!/usr/bin/env bash
set -euo pipefail
# User-invoked only. No sudo, no system CUDA/driver modifications.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir --upgrade pip
# RTX 4090: CUDA 12.8 wheel, provided host driver must support it.
python -m pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir -e '.[dev,data]'
python -m pip freeze > requirements-resolved.txt
