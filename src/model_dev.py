import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):

    @abstractmethod
    def train(self):
        pass


class LinearRegressModel(Model):
     def train(self, X_train, y_train, **kwargs):
        reg = LinearRegression(**kwargs)
        reg.fit(X_train,y_train)
        logging.info("Model has been trained")
        return reg


        
