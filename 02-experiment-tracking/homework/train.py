import argparse
import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path):
    with mlflow.start_run():
        # mlflow.set_tag("developer", "ag239")
        # mlflow.log_param("data_path", data_path)
        params = {
            "max_depth": 10,
            "random_state": 0,
        }
        # for k, v in params.items():
        #     mlflow.log_param(k, v)

        # mlflow.sklearn.autolog()
        mlflow.log_param("data_path", data_path)

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_valid)

        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        # mlflow.log_metric("rmse", rmse)
        #
        # mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///run_folder/mlflow.db")
    mlflow.set_experiment("sdfdsf")

    run(args.data_path)
