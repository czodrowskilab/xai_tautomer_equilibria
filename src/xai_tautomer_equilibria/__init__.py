from pathlib import Path

# project root detection
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TAUTOBASE_PATH = RAW_DIR / "tautobase_edited.csv"