"""Seed persistent model files, then start the FastAPI service."""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BEARING_DATA_DIR", "/var/data/cwru"))
MODEL_DIR = Path(os.environ.get("BEARING_MODEL_DIR", "/var/data/models"))
RESULTS_PATH = Path(os.environ.get("BEARING_RESULTS_PATH", "/var/data/results.json"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

for source in (ROOT / "models").glob("*"):
    destination = MODEL_DIR / source.name
    if source.is_file() and not destination.exists():
        shutil.copy2(source, destination)

bundled_results = ROOT / "results" / "results.json"
if bundled_results.is_file() and not RESULTS_PATH.exists():
    shutil.copy2(bundled_results, RESULTS_PATH)

os.environ["BEARING_DATA_DIR"] = str(DATA_DIR)
os.environ["BEARING_MODEL_DIR"] = str(MODEL_DIR)
os.environ["BEARING_RESULTS_PATH"] = str(RESULTS_PATH)
sys.path.insert(0, str(ROOT / "backend"))

import uvicorn

uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
