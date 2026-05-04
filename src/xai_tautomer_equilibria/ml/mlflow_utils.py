import mlflow


def setup_mlflow(experiment_name: str):
    mlflow.set_experiment(experiment_name)


def log_model_and_metrics(model, metrics, params, dataset_meta):

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    mlflow.set_tags(dataset_meta)

    mlflow.sklearn.log_model(model, "model")