from zenml import pipeline

from steps.clean_data import clean_data
from steps.ingest_data import ingest_df
from steps.evaluation import evaluate_model
from steps.train_model import train_model
# factory pattern, singleton pattern

@pipeline(enable_cache=True)
def train_pipeline(data_pth:str):
    df = ingest_df(data_pth)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test )
    
    mse, r2 = evaluate_model(model, X_test, y_test)

