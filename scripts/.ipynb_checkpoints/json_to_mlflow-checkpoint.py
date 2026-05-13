import json
import shutil
from pathlib import Path

import mlflow

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(
    "/fshpc/limmiche/projects/xai_tautomer_equilibria"
)

JSON_FILE = BASE_DIR / "optuna_summary.json"

MLFLOW_DIR = BASE_DIR / "mlruns_ui"

# =========================================================
# RESET OLD DATA
# =========================================================
if MLFLOW_DIR.exists():
    shutil.rmtree(MLFLOW_DIR)

# =========================================================
# MLFLOW
# =========================================================
mlflow.set_tracking_uri(
    f"file:{MLFLOW_DIR}"
)

mlflow.set_experiment(
    "demo"
)

# =========================================================
# LOAD DATA
# =========================================================
data = json.loads(
    JSON_FILE.read_text()
)

# =========================================================
# KEEP UI SMALL
# =========================================================
MAX_RUNS = 30

for item in data[:MAX_RUNS]:

    try:

        with mlflow.start_run(
            run_name=item["study_name"][:60]
        ):

            # -----------------------------------------
            # ONLY 2-3 IMPORTANT PARAMS
            # -----------------------------------------
            params = item.get("params", {})

            important_keys = list(params.keys())[:3]

            for k in important_keys:

                mlflow.log_param(
                    str(k),
                    str(params[k]),
                )

            # -----------------------------------------
            # ONLY MAIN METRIC
            # -----------------------------------------
            if item.get("best_value") is not None:

                mlflow.log_metric(
                    "score",
                    float(item["best_value"]),
                )

    except Exception as e:

        print(f"Skipping run: {e}")

print("Compact MLflow demo dataset created.")