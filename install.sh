#!/usr/bin/env bash
# One-time setup (needs internet). After this, ./start.sh works offline.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -m venv .venv
source .venv/Scripts/activate

pip install -r backend/requirements.txt
python scripts/download_models.py
(cd frontend && npm install && npm run build)
echo
echo "All set. Run ./start.sh"
