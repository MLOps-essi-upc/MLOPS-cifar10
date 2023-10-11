from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(Path(__file__).resolve().parent.parent)

RAW_DATA_DIR = ROOT_DIR / "data/raw"
TRAIN_DATA_DIR = ROOT_DIR / "data/train"
TEST_DATA_DIR = ROOT_DIR / "data/test"

PROCESSED_DATA_DIR = ROOT_DIR / "data/processed"

METRICS_DIR = ROOT_DIR / "metrics"
MODELS_DIR = ROOT_DIR / "models"