#!/bin/bash
# Double-click to open the Scholar-IR app (Jupyter + ipywidgets) in your browser.
cd "$(dirname "$0")"
echo "Starting Scholar-IR... (keep this window open; close it to stop the app)"
if python3 -m jupyter --version >/dev/null 2>&1; then
  python3 -m jupyter notebook notebooks/07_app.ipynb
elif command -v jupyter >/dev/null 2>&1; then
  jupyter notebook notebooks/07_app.ipynb
else
  echo "Jupyter not found - installing requirements once..."
  python3 -m pip install -r requirements.txt ipywidgets notebook && python3 -m jupyter notebook notebooks/07_app.ipynb
fi
