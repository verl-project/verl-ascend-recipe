"""Filesystem configuration for the fixed Xing4 numerical acceptance workload."""

import os
from pathlib import Path

RECIPE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("XING_MODEL_PATH", "models/Xing4.0-29B-A4B")).resolve()
WORK_DIR = Path(os.environ.get("XING_WORK_DIR", "outputs/zero_mismatch/xing4")).resolve()
RUNS_DIR = WORK_DIR / "runs"
CASE_MANIFEST = RECIPE_DIR / "case_manifest.json"
