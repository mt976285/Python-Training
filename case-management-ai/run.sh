#!/usr/bin/env bash
set -euo pipefail

# Create venv and install if missing
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

# Activate
. .venv/bin/activate

if [ "${1-}" = "uvicorn" ]; then
  echo "uvicorn target removed; running streamlit instead"
  streamlit run main.py
elif [ "${1-}" = "streamlit" ]; then
  streamlit run main.py
else
  streamlit run main.py
fi
