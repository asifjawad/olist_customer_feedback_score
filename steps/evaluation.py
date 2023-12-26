import logging

import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin

from typing import Tuple
from typing_extensions import Annotated

from src.evaluation import RMSE, MSE, R2


@step
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test:pd.DataFrame
                   ) -> Tuple[
                       Annotated[float, "mse"],
                       Annotated[float,"r2"]
                   ]:
    

    try:
        prediction = model.predict(X_test)
        mse_class= MSE()
        mse = mse_class.calculate_score(y_test, prediction)

        r2_class= R2()
        r2 = r2_class.calculate_score(y_test, prediction)

        rmse_class= RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)

        return mse, r2
    except Exception as e:
        logging.error(f"Error in evaluating model, Please check again")
        raise e
