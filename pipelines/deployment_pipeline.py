import json


import numpy as np
import pandas as pd

from materializer.custom_materializer import cs_materializer
from steps.clean_data import clean_data
from steps.ingest_data import ingest_df
from steps.evaluation import evaluate_model
from steps.train_model import train_model

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output



docker_setting = DockerSettings(required_integrations=[MLFLOW])




class DeploymentTriggerConfig(BaseParameters):
    min_accuracy:float = 0

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    
    return accuracy >= config.min_accuracy
    





@pipeline(enable_cache=True, settings={"docker_setting": docker_setting})
def continous_deployment_pipeline(
    data_pth: str = "data/olist_customers_dataset.csv",
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df = ingest_df(data_pth)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test ) 
    mse, r2 = evaluate_model(model, X_test, y_test)

    deployment_decision = deployment_trigger(r2)

    mlflow_model_deployer_step(
        model= model,
        deployment_decision = deployment_decision,
        workers = workers,
        timeout= timeout,
    )

