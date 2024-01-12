import sys

# setting path
sys.path.append('..')

import pandas as pd
import mlflow
from mlflow.models import infer_signature

from abstract_models.runner import RFRegressorRunner
from data.source import HousePriceSource
from data.loader import load_house_price_data

if __name__ == "__main__":
    ds = HousePriceSource(load_house_price_data("house_price_data/train.csv"))

    rf = RFRegressorRunner(ds)
    result = rf.run()
    result.report()
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("House Price Prediction")

    # Start an MLflow run
    with mlflow.start_run():
        # Log the metrics
        for k, v in rf.metric.__dict__.items():
            mlflow.log_metric(k, v)

        # Infer the model signature
        signature = infer_signature(rf.X_train, rf.rf.predict(rf.X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=rf.rf,
            artifact_path="house_price",
            signature=signature,
            input_example=rf.X_train,
            registered_model_name="house_price",
        )
