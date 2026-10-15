"""Runtime paths and deterministic settings."""
from __future__ import annotations
import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("ASD_DATA_DIR", r"E:\College\7th Semester\Project\Data Set"))
CONFIG = json.loads((PROJECT_ROOT / "configs/project.json").read_text(encoding="utf-8"))
RANDOM_SEED = int(CONFIG["random_seed"])
TRAIN_FRACTION = float(CONFIG["split"]["train"])
VALIDATION_FRACTION = float(CONFIG["split"]["validation"])
