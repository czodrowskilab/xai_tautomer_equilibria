from pathlib import Path

Path("../data/processed").mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
INTERIM_DIR = DATA_DIR / "interim"

# -------------------------
# Raw data
# -------------------------
TAUTOBASE_PATH = RAW_DIR / "tautobase_edited.csv"

# -------------------------
# External data
# -------------------------
WIEDER_ROWS_PATH = EXTERNAL_DIR / "wieder_rows.pkl"

# -------------------------
# Tautomer stability data
# -------------------------
DDG_PATH = RAW_DIR / "ddG_values.pkl"

# -------------------------
# Aromaticity data
# -------------------------
AROMATICITY_DIR = INTERIM_DIR / "aromaticity"

HOMA_VALUES_PATH = AROMATICITY_DIR / "homa_values.csv"
MCI_VALUES_PATH = AROMATICITY_DIR / "mci_values.csv"
NICSZZ_VALUES_PATH = AROMATICITY_DIR / "nicszz_values.csv"