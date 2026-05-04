from pathlib import Path
Path("../data/processed").mkdir(parents=True, exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

TAUTOBASE_PATH = RAW_DIR / "tautobase_edited.csv"
WIEDER_ROWS_PATH = EXTERNAL_DIR / "wieder_rows.pkl"

DDG_PATH = RAW_DIR / "ddG_values.pkl"

