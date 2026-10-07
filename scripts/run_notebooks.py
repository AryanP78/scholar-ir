"""Execute notebooks in place (used by `make notebooks` / `make demo`). Usage: python scripts/run_notebooks.py [nb ...]"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
targets = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "notebooks").glob("0*.ipynb"))
for nb_path in targets:
    nb = nbformat.read(nb_path, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(nb_path.parent)}}).execute()
    nbformat.write(nb, nb_path)
    print("executed", nb_path)
