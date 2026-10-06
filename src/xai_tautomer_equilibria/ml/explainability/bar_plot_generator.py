#!/usr/bin/env python
"""
Aggregate feature importances across models for each
(subset, target, variant, radius, n_bits) combination.

Supports two kinds of input trees:

  --kind shap       reads: data_shap/<target>/<model_id>/<family>_shap.csv
                    value column: "Mean_SHAP_Abs"


  --kind intrinsic  reads: data_intrinsic/<target>/<model_id>/<family>-regression_coef.csv
                    value column: "Coefficient"

Writes (per group):
    <out-root>/scaled_feat_coeff_<group>/<family>__seed<N>__scaled.csv
    <out-root>/ordered_scaled_feat_coeff_<group>/<family>__seed<N>__scaled_ordered.csv
    <out-root>/<group>/feat_imp_<kind>.xlsx
    <out-root>/<group>/bar_plot_<kind>.svg

Also emits:
    <out-root>/feature_position_tracking.csv
    (split per (subset, target) with --split-positions)

Usage:
    python bar_plot_generator.py --kind shap \\
        --root ./data_shap --out-root ./_data_shap \\
        --families random_forest xgboost_regressor \\
        --seeds 43 --target-feature relative_homa_difference --top-n 10
or:
    python bar_plot_generator.py --kind intrinsic \\
        --root ./data_intrinsic --out-root ./_data_intrinsic \\
        --seeds 43 --top-n 10
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aggregate")


# =========================================================
# KIND-SPECIFIC CONFIG
# =========================================================

@dataclass(frozen=True)
class KindSpec:
    name:              str            # "shap" | "intrinsic"
    value_column:      str            # column holding importance
    filename_template: str            # how to build the per-model filename
    default_families:  tuple[str, ...]

KIND_SPECS = {
    "shap": KindSpec(
        name="shap",
        value_column="Mean_SHAP_Abs",
        filename_template="{family}_shap.csv",
        default_families=("random_forest", "xgboost_regressor"),
    ),
    "intrinsic": KindSpec(
        name="intrinsic",
        value_column="Coefficient",
        # NOTE: intrinsic script writes "<family>-regression_coef.csv"
        # for all families except random_forest, which is "RF-regression_coef.csv".
        # See resolve_filename() below.
        filename_template="{family}-regression_coef.csv",
        default_families=(
            "elastic_net_regression",
            "lasso_regression",
            "ridge_regression",
            "random_forest",
            "xgboost_regressor",
        ),
    ),
}


def resolve_filename(family: str, spec: KindSpec) -> str:
    """Per-model CSV filename inside the model_id directory."""
    if spec.name == "intrinsic" and family == "random_forest":
        return "RF-regression_coef.csv"
    return spec.filename_template.format(family=family)


# =========================================================
# GROUPING
# =========================================================

@dataclass(frozen=True)
class GroupKey:
    subset:  str
    target:  str
    variant: str
    radius:  int
    n_bits:  int

    @property
    def slug(self) -> str:
        return (
            f"{self.subset}"
            f"__{self.target}"
            f"__{self.variant}"
            f"__r{self.radius}"
            f"__b{self.n_bits}"
        )


# Matches: <family>_r<radius>_b<n_bits>_<variant>_<subset>_seed<seed>_v<version>
MODEL_ID_RE = re.compile(
    r"^(?P<family>.+?)"
    r"_r(?P<radius>\d+)"
    r"_b(?P<n_bits>\d+)"
    r"_(?P<variant>[a-z_]+?)"
    r"_(?P<subset>[a-z_]+?)"
    r"_seed(?P<seed>\d+)"
    r"_v\d+$"
)


def parse_model_id(model_id: str) -> dict | None:
    m = MODEL_ID_RE.match(model_id)
    if not m:
        return None
    g = m.groupdict()
    return {
        "family":  g["family"],
        "radius":  int(g["radius"]),
        "n_bits":  int(g["n_bits"]),
        "variant": g["variant"],
        "subset":  g["subset"],
        "seed":    int(g["seed"]),
    }


# =========================================================
# DISCOVERY
# =========================================================

def discover_files(
    root:     Path,
    spec:     KindSpec,
    families: set[str],
    seeds:    set[int] | None,
) -> list[dict]:
    records = []

    for target_dir in sorted(root.iterdir()):
        if not target_dir.is_dir():
            continue
        target = target_dir.name

        for model_dir in sorted(target_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            parsed = parse_model_id(model_dir.name)
            if parsed is None:
                continue
            if parsed["family"] not in families:
                continue
            if seeds is not None and parsed["seed"] not in seeds:
                continue

            csv_path = model_dir / resolve_filename(parsed["family"], spec)
            if not csv_path.exists():
                continue

            records.append({
                "path":   csv_path,
                "target": target,
                **parsed,
                "group": GroupKey(
                    subset=parsed["subset"],
                    target=target,
                    variant=parsed["variant"],
                    radius=parsed["radius"],
                    n_bits=parsed["n_bits"],
                ),
            })

    return records


# =========================================================
# AGGREGATION
# =========================================================

def scale_values(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Min-max scale `value_column` to [0, 1]; sort descending."""
    if "Feature" not in df.columns or value_column not in df.columns:
        raise KeyError(f"expected columns Feature, {value_column}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.copy()
    df["Scaled Coefficient"] = scaler.fit_transform(df[[value_column]])
    df = df.reindex(
        df["Scaled Coefficient"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    return df


def aggregate_group(
    records:        list[dict],
    out_root:       Path,
    spec:           KindSpec,
    aggregate:      str,             # "mean" | "median" | "sum"
    top_n:          int,
    target_feature: str,
) -> dict | None:

    group: GroupKey = records[0]["group"]
    slug = group.slug

    log.info(f"group {slug}  ({len(records)} files)")

    scaled_folder  = out_root / f"scaled_feat_coeff_{slug}"
    ordered_folder = out_root / f"ordered_scaled_feat_coeff_{slug}"
    report_folder  = out_root / slug
    scaled_folder.mkdir(parents=True, exist_ok=True)
    ordered_folder.mkdir(parents=True, exist_ok=True)
    report_folder.mkdir(parents=True, exist_ok=True)

    scaled_dfs = []
    for rec in records:
        df = pd.read_csv(rec["path"])
        df = scale_values(df, spec.value_column)
        out_name = f"{rec['family']}__seed{rec['seed']}__scaled.csv"
        df.to_csv(scaled_folder / out_name, index=False)
        scaled_dfs.append((rec, df))

    merged = pd.concat([df for _, df in scaled_dfs], ignore_index=True)

    if aggregate == "mean":
        agg = merged.groupby("Feature")["Scaled Coefficient"].mean().reset_index()
    elif aggregate == "median":
        agg = merged.groupby("Feature")["Scaled Coefficient"].median().reset_index()
    elif aggregate == "sum":
        agg = merged.groupby("Feature")["Scaled Coefficient"].sum().reset_index()
    else:
        raise ValueError(f"unknown aggregate: {aggregate}")

    agg = agg.sort_values("Scaled Coefficient", ascending=False).reset_index(drop=True)
    agg.to_excel(report_folder / f"feat_imp_{spec.name}.xlsx", index=False)

    # ---- target feature position ----------------------------------
    if target_feature in agg["Feature"].values:
        position = int(agg.index[agg["Feature"] == target_feature][0]) + 1
    else:
        position = -1

    # ---- top-N features ------------------------------------------
    top_features = list(agg.head(top_n)["Feature"])
    feature_order = {f: i for i, f in enumerate(top_features)}

    def reorder(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["Feature"].isin(top_features)].copy()
        df["__order"] = df["Feature"].map(feature_order)
        df = df.sort_values("__order").drop(columns="__order").reset_index(drop=True)
        df = df.drop(columns=[spec.value_column], errors="ignore")
        return df

    ordered_dfs = []
    for rec, df in scaled_dfs:
        odf = reorder(df)
        out_name = f"{rec['family']}__seed{rec['seed']}__scaled_ordered.csv"
        odf.to_csv(ordered_folder / out_name, index=False)
        ordered_dfs.append((rec, odf))

    # ---- bar plot --------------------------------------------------
    wide = None
    for rec, odf in ordered_dfs:
        col = f"{rec['family']}__seed{rec['seed']}"
        s = odf.set_index("Feature")["Scaled Coefficient"].rename(col)
        wide = s.to_frame() if wide is None else wide.join(s, how="outer")

    wide = wide.reindex([f for f in top_features if f in wide.index])

    families = sorted({rec["family"] for rec, _ in ordered_dfs})
    palette = sns.color_palette("colorblind", n_colors=len(families))
    family_color = dict(zip(families, palette))
    colors = [family_color[col.split("__seed")[0]] for col in wide.columns]

    fig, ax = plt.subplots(figsize=(max(10, 0.4 * len(wide.index)), 6))
    wide.plot(kind="bar", ax=ax, color=colors, width=0.8)
    ax.set_xlabel("Feature")
    ax.set_ylabel(
        "Scaled SHAP importance" if spec.name == "shap"
        else "Scaled intrinsic importance"
    )
    ax.set_title(
        f"{group.target} | {group.variant} | {group.subset} | "
        f"r={group.radius} b={group.n_bits}"
    )
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, ncol=1)
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(report_folder / f"bar_plot_{spec.name}.svg", format="svg",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    log.info(
        f"  → {report_folder.relative_to(out_root)}  "
        f"(target position: {position})"
    )

    return {
        "subset":   group.subset,
        "target":   group.target,
        "variant":  group.variant,
        "radius":   group.radius,
        "n_bits":   group.n_bits,
        "position": position,
    }


# =========================================================
# MAIN
# =========================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--kind", choices=list(KIND_SPECS.keys()),
                    default="shap",
                    help="which tree to aggregate")
    ap.add_argument("--root",     type=Path, required=True,
                    help="data_shap/ or data_intrinsic/ root")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="default: ./_<kind>")

    ap.add_argument("--families", nargs="*", default=None,
                    help="default depends on --kind")
    ap.add_argument("--seeds",    nargs="*", type=int, default=None)

    ap.add_argument("--aggregate", choices=["mean", "median", "sum"],
                    default="mean")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--target-feature", default="relative_homa_difference")
    ap.add_argument("--split-positions", action="store_true")

    args = ap.parse_args()

    spec = KIND_SPECS[args.kind]

    if args.out_root is None:
        args.out_root = Path(f"./_{args.kind}")

    if not args.root.exists():
        sys.exit(f"[ERROR] input root not found: {args.root}")

    families = set(args.families) if args.families else set(spec.default_families)
    seeds    = set(args.seeds)    if args.seeds    else None

    records = discover_files(args.root, spec,
                             families=families, seeds=seeds)
    log.info(
        f"[{spec.name}] discovered {len(records)} files "
        f"(families={sorted(families)}, "
        f"seeds={sorted(seeds) if seeds else 'all'})"
    )

    if not records:
        sys.exit("Nothing to aggregate.")

    groups: dict[GroupKey, list[dict]] = {}
    for r in records:
        groups.setdefault(r["group"], []).append(r)

    log.info(f"Formed {len(groups)} (subset, target, variant, radius, n_bits) groups")

    args.out_root.mkdir(parents=True, exist_ok=True)
    position_rows = []

    for group in sorted(groups.keys(), key=lambda g: g.slug):
        try:
            row = aggregate_group(
                groups[group],
                out_root=args.out_root,
                spec=spec,
                aggregate=args.aggregate,
                top_n=args.top_n,
                target_feature=args.target_feature,
            )
            if row is not None:
                position_rows.append(row)
        except Exception as exc:
            log.error(f"  FAILED group {group.slug}: {exc}")

    positions_csv = args.out_root / "feature_position_tracking.csv"
    with open(positions_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["subset", "target", "variant", "radius", "n_bits",
                        "position"],
        )
        w.writeheader()
        w.writerows(position_rows)
    log.info(f"Wrote {positions_csv}  ({len(position_rows)} rows)")

    if args.split_positions and position_rows:
        df = pd.DataFrame(position_rows)
        for (subset, target), sub in df.groupby(["subset", "target"]):
            out = args.out_root / f"positions__{subset}__{target}.csv"
            sub.to_csv(out, index=False)
            log.info(f"Wrote {out}")


if __name__ == "__main__":
    main()