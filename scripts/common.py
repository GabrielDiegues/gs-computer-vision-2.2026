"""Caminhos e constantes compartilhados pelos scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets" / "raw"
DATASET_DIR = ROOT / "datasets" / "rua_acesa"
DATA_YAML = DATASET_DIR / "data.yaml"
RUNS_DIR = ROOT / "runs"
MODELS_DIR = ROOT / "models"
DOCS_IMG = ROOT / "docs" / "img"

CLASSES = ["poste_aceso", "poste_apagado"]
SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Permite que os scripts reutilizem o código da API (ex.: app.scene).
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
