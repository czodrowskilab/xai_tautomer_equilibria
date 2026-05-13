import json
import shutil
from pathlib import Path

import mlflow

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(
    "/mnt/czolab-storage-01/limmiche/projects/xai_tautomer_equilibria_mogon"
)

JSON_FILE = BASE_DIR / "optuna_summary.json"
MLFLOW_DIR = BASE_DIR / "mlruns_ui"

# =========================================================
# RESET
# =========================================================
if MLFLOW_DIR.exists():
    shutil.rmtree(MLFLOW_DIR)

mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
mlflow.set_experiment("optuna_overview")

data = json.loads(JSON_FILE.read_text())

MAX_RUNS = 480

for idx, item in enumerate(data[:MAX_RUNS]):

    try:

        study_name = item["study_name"]
        short_name = study_name.replace("_", "-")[:60]

        # =================================================
        # STRUCTURED METADATA (NEW JSON SCHEMA)
        # =================================================
        model = item.get("model", "unknown")
        dataset = item.get("dataset", "unknown")
        radius = item.get("radius", "unknown")
        bitlength = item.get("bitlength", "unknown")
        variant = item.get("variant", "unknown")
        seed = item.get("seed", "unknown")

        params = item.get("params", {})

        with mlflow.start_run(run_name=short_name):

            # =================================================
            # TAGS (🔥 NOW FULL MULTI-AXIS FILTERING)
            # =================================================
            mlflow.set_tag("model", model)
            mlflow.set_tag("dataset", dataset)
            mlflow.set_tag("radius", radius)
            mlflow.set_tag("bitlength", bitlength)
            mlflow.set_tag("variant", variant)
            mlflow.set_tag("seed", seed)
            mlflow.set_tag("study", short_name)

            # =================================================
            # PARAMS (clean + safe)
            # =================================================
            for k, v in params.items():

                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(k, str(v))

            # =================================================
            # MAIN SCORE
            # =================================================
            best_value = item.get("best_value")

            if best_value is not None:
                mlflow.log_metric("score", float(best_value))

            # =================================================
            # LEARNING CURVE
            # =================================================
            for p in item.get("learning_curve", []):

                mlflow.log_metric(
                    "optuna_value",
                    p["value"],
                    step=p["trial"],
                )

            # =================================================
            # BEST-SO-FAR CURVE
            # =================================================
            for p in item.get("best_so_far_curve", []):

                mlflow.log_metric(
                    "best_so_far",
                    p["best_so_far"],
                    step=p["trial"],
                )

            # =================================================
            # SUMMARY ARTIFACT
            # =================================================
            mlflow.log_text(
                json.dumps(
                    {
                        "study": study_name,
                        "model": model,
                        "dataset": dataset,
                        "radius": radius,
                        "bitlength": bitlength,
                        "variant": variant,
                        "seed": seed,
                        "best": best_value,
                    },
                    indent=2,
                ),
                "summary.json",
            )

    except Exception as e:
        print(f"Skipping run: {e}")

print(
    f"Created MLflow UI with {min(MAX_RUNS, len(data))} runs."
)