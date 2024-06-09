from typing import List, Tuple
import pandas as pd
import mlflow
import pickle
import os

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "homework_3 LinReg"

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data : Tuple[DictVectorizer, LinearRegression], *args, **kwargs):

    dv, lr = data

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(lr, artifact_path = "artifacts", registered_model_name="LinRegModel")

    client = MlflowClient()
  
    os.makedirs("./artifacts", exist_ok=True)
    with open('./artifacts/homework_03_dv.pkl', 'wb') as f:
        pickle.dump(dv, f)
        client.log_artifact(run.info.run_id, './artifacts/homework_03_dv.pkl')
   
    #with mlflow.start_run():
    #   mlflow.sklearn.log_model(lr, artifact_path = "/workspaces/mlops/artifacts/", registered_model_name="LinRegModel")
    #
