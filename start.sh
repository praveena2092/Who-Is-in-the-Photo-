#!/usr/bin/env bash
# Starts the app on http://127.0.0.1:8000 and opens the browser. Works offline.
cd "$(dirname "$0")"
source .venv/Scripts/activate
URL=http://127.0.0.1:8000
( sleep 3; (command -v open >/dev/null && open $URL) || (command -v xdg-open >/dev/null && xdg-open $URL) ) >/dev/null 2>&1 &
cd backend && python -W ignore -m uvicorn app.main:app --host 127.0.0.1 --port 8000
