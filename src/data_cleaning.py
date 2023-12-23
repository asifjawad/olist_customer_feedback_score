import logging

import pandas as pd
import numpy as np
from typing import Union

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract Class to handle strategy and Data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataProcessingStrategy(DataStrategy):
    """ Pre-processing Data"""
def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
    try:
        data = data.drop([], axis=1)
        data = data.select_dtype(include=[np.number])
        cols_to_drop = ["", ""]
        data = data.drop(cols_to_drop, axis=1)
        return data
         


    except Exception as e:
        logging.error(f" Error in Processing Data {e}")
        raise e
    

class DataDivideStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X= data.drop([], axis= 1)
            y = data[]
            X_train, X_test, y_train, y_test = train_test_split(X,y)
            retun X_train

        except Exception as e:
            logging.error(f)
            raise e


class DataCleaning:
    def __init__(self, df, strategy):
        self.df = df
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.df)