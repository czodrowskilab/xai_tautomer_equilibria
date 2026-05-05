import mlflow
import pandas as pd


def main():

    client = mlflow.tracking.MlflowClient()

    runs = client.search_runs("tautomer_equilibria")

    results = []

    for r in runs:

        results.append({
            "run_id": r.info.run_id,
            "mse": r.data.metrics.get("mse"),
            "mae": r.data.metrics.get("mae"),
            "model": r.data.tags.get("model"),
            "seed": r.data.tags.get("seed"),
        })

    df = pd.DataFrame(results)
    df.to_csv("results_summary.csv", index=False)

    print(df.sort_values("mse", ascending=True).head(10))


if __name__ == "__main__":
    main()