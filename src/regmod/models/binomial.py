"""
Binomial Model
"""
from typing import List

import numpy as np
from regmod.data import Data

from .model import Model


class BinomialModel(Model):
    param_names = ("p",)
    default_param_specs = {"p": {"inv_link": "expit"}}

    def __init__(self, data: Data, **kwargs):
        if not np.all(data.obs >= 0):
            raise ValueError("Binomial model requires observations to be non-negative.")
        if len(data.col_obs) != 2:
            raise ValueError("Binomial model need 2 columns of observations, "
                             "one for number of events, one for sample size.")
        if any(np.diff(data.get_cols(data.col_obs), axis=1) < 0):
            raise ValueError("Binomial model requires number of events less or equal than sample size.")

        self.obs_1s = data.get_cols(data.col_obs[0])
        self.obs_0s = np.diff(data.get_cols(data.col_obs), axis=1).ravel()
        self.obs_sample_sizes = data.get_cols(data.col_obs[1])

        super().__init__(data, **kwargs)

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        return -(self.obs_1s*np.log(params[0]) + self.obs_0s*np.log(1.0 - params[0]))

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [-(self.obs_1s/params[0] - self.obs_0s/(1.0 - params[0]))]

    def d2nll(self, params: List[np.ndarray]) -> List[List[np.ndarray]]:
        return [[self.obs_1s/params[0]**2 + self.obs_0s/(1.0 - params[0])**2]]

    def __repr__(self) -> str:
        return f"BinomialModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"
