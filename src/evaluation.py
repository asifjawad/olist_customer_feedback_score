from abc import ABC, abstractmethod
import logging
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score






class Evaluation(ABC):
    @abstractmethod
    def calculate_score(self, y_test: np.ndarray, y_pred: np.ndarray):
        pass


class MSE(Evaluation):
    def calculate_score(self, y_test: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info(f"Calculating MSE")
            mse = mean_squared_error(y_test, y_pred)
            logging.info(f"MSE {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error calculating MSE")
            raise e
        


class R2(Evaluation):
    def calculate_score(self, y_test: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info(f"Calculating MSE")
            r2 = r2_score(y_test, y_pred)
            logging.info(f"R2 {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error calculating R2")
            raise e
        

class RMSE(Evaluation):
    def calculate_score(self, y_test: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info(f"Calculating MSE")
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            logging.info(f"R2 {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error calculating RMSE")
            raise e