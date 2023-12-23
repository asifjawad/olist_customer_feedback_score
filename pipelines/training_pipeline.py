from zenml import pipeline

from steps.clean_data import clean_df
from steps.ingest_data import ingest_df
from steps.evaluation import evaluate_model
from steps.train_model import train_model

@pipeline
def train_pipeline(data_pth:str):
    df = ingest_df(data_pth)
    clean_df(df)
    train_model(df)
    evaluate_model(df)

