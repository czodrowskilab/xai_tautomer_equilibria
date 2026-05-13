import json
import os
from pathlib import Path

import optuna
from optuna.storages import RDBStorage

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(
    "/fshpc/limmiche/projects/xai_tautomer_equilibria"
)

OUTPUT_FILE = BASE_DIR / "optuna_summary.json"

# =========================================================
# STORAGE
# =========================================================
OPTUNA_STORAGE = os.getenv(
    "OPTUNA_STORAGE",
    "sqlite:///optuna_demo.db",
)

# =========================================================
# STORAGE CONFIG
# =========================================================
storage = RDBStorage(
    url=OPTUNA_STORAGE,
    engine_kwargs={
        "pool_size": 1,
        "max_overflow": 0,
        "pool_pre_ping": True,
    },
)

# =========================================================
# LOAD STUDIES
# =========================================================
studies = optuna.study.get_all_study_summaries(
    storage=storage
)

print(f"Found {len(studies)} studies")

results = []

for summary in studies:

    try:

        study = optuna.load_study(
            study_name=summary.study_name,
            storage=storage,
        )

        if len(study.trials) == 0:
            continue

        completed_trials = [
            t for t in study.trials
            if t.value is not None
        ]

        if len(completed_trials) == 0:
            continue

        best = study.best_trial

        results.append(
            {
                "study_name": summary.study_name,
                "n_trials": len(study.trials),
                "best_value": float(best.value),
                "params": best.params,
                "direction": summary.direction.name,
                "best_trial_number": best.number,
                "datetime_start": (
                    str(best.datetime_start)
                    if best.datetime_start
                    else None
                ),
                "datetime_complete": (
                    str(best.datetime_complete)
                    if best.datetime_complete
                    else None
                ),
                "user_attrs": best.user_attrs,
            }
        )

        print(
            f"Added: {summary.study_name} "
            f"({len(study.trials)} trials)"
        )

    except Exception as e:

        print(
            f"Skipping corrupted study "
            f"{summary.study_name}"
        )

        print(str(e))

# =========================================================
# SAVE
# =========================================================
with open(OUTPUT_FILE, "w") as f:

    json.dump(
        results,
        f,
        indent=2,
    )

print(
    f"Saved {len(results)} study summaries "
    f"→ {OUTPUT_FILE}"
)

# =========================================================
# CLEANUP
# =========================================================
try:
    storage.engine.dispose()
except Exception:
    pass