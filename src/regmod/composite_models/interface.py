"""
Node Model
"""
from abc import ABC, abstractmethod
from typing import Dict

from pandas import DataFrame

from regmod.composite_models.node import Node


class ModelInterface(ABC):
    """
    Abstract class that encode the behavior of the model interface
    """

    col_label = "label"
    col_value = "value"

    @abstractmethod
    def set_data(self, df: DataFrame):
        pass

    @abstractmethod
    def get_data(self) -> DataFrame:
        pass

    @abstractmethod
    def fit(self, **fit_options):
        pass

    @abstractmethod
    def predict(self, df: DataFrame = None) -> DataFrame:
        pass

    @abstractmethod
    def set_prior(self, priors: Dict, masks: Dict = None):
        pass

    @abstractmethod
    def get_posterior(self) -> Dict:
        pass


class NodeModel(Node, ModelInterface):
    pass
