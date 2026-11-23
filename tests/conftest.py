import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("EMO_SMOKE", None)
os.environ.pop("EMO_MODEL", None)
os.environ.setdefault("EMO_DEVICE", "cpu")
