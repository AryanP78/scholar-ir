"""Configuration loading. All tunable numbers live in config.yaml at the repo root."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load config.yaml (or $SCHOLAR_IR_CONFIG) and resolve paths relative to the repo root."""
    path = Path(path or os.environ.get("SCHOLAR_IR_CONFIG", ROOT / "config.yaml"))
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(ROOT)
    return cfg


def p(cfg: dict[str, Any], key: str) -> Path:
    """Absolute path for cfg['paths'][key]."""
    path = Path(cfg["paths"][key])
    return path if path.is_absolute() else Path(cfg["_root"]) / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "fsspec", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Minimal .env reader (lines like `export KEY=value` or `KEY=value`). Never logs values."""
    path = Path(path or ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
