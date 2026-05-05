import mlflow
import pandas as pd

def setup_mlflow(experiment_name: str):
    mlflow.set_experiment(experiment_name)


def log_model_and_metrics(model, metrics, params, dataset_meta):

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    mlflow.set_tags(dataset_meta)

    mlflow.sklearn.log_model(model, "model")




def get_best_params(experiment_name, dataset, model, seed):
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"""
            tags.dataset = '{dataset}' and
            tags.model = '{model}' and
            tags.seed = '{seed}' and
            tags.stage = 'tuning'
        """,
        order_by=["metrics.best_mse ASC"]
    )

    if runs.empty:
        return None

    best_run = runs.iloc[0]

    params = {
        k.replace("params.", ""): v
        for k, v in best_run.items()
        if k.startswith("params.")
    }

    return params