import argparse
import os
import pickle

import mlflow
from hyperopt import hp, space_eval
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from mlflow.exceptions import MlflowException
from sklearn.metrics import mean_squared_error
import uuid

HPO_EXPERIMENT_NAME = "random-forest-hyperopt_01"
EXPERIMENT_NAME = f"random-forest-best-models_final_{uuid.uuid4()}"
print(EXPERIMENT_NAME)
MLFLOW_TRACKING_URI = "sqlite:///run_folder/mlflow.db"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

SPACE = {
    'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
    'random_state': 42
}


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        params = space_eval(SPACE, params)
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # evaluate model on the validation and test sets
        valid_rmse = mean_squared_error(y_valid, rf.predict(X_valid), squared=False)
        mlflow.log_metric("valid_rmse", valid_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)


def run(data_path, log_top):

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # retrieve the top_n model runs and log the models to MLflow
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=log_top,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"])[0]

    # register the best model
    # mlflow.register_model( ... )

    print("#"*30)
    print(best_run.info.run_id)
    # print("#"*30)
    # print(best_run.data)
    # print("#"*30)
    # print(best_run.to_dictionary())
    # print("#"*30)
    # # print(client.list_run_infos(experiment_id=experiment.experiment_id))
    # run_id = client.list_run_infos(experiment_id='1')[0].run_id
    # print(f"runs:/{best_run.run_id}/models")
    mlflow.register_model(
        model_uri=f"runs:/{best_run.info.run_id}/models",
        name='best_test_rmse'
    )

    model_name = 'best_test_rmse'
    latest_versions = client.get_latest_versions(name=model_name)

    for version in latest_versions:
        print(f"version: {version.version}, stage: {version.current_stage}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    parser.add_argument(
        "--top_n",
        default=1,
        type=int,
        help="the top 'top_n' models will be evaluated to decide which model to promote."
    )
    args = parser.parse_args()

    run(args.data_path, args.top_n)

