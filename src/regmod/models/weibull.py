"""
Weibull Model
"""
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import weibull_min

from .model import Model


class WeibullModel(Model):
    param_names = ("b", "k")
    default_param_specs = {"b": {"inv_link": "exp"}, "k": {"inv_link": "exp"}}

    def _validate_data(self, df: pd.DataFrame, fit: bool = True):
        super()._validate_data(df, fit)
        if fit and not all(df[self.y] > 0):
            raise ValueError("Weibull model requires observations to be positive.")

    def nll(self, data: dict, params: list[NDArray]) -> NDArray:
        t = data["y"]
        ln_t = np.log(t)
        return (
            params[0] * (t ** params[1])
            - (params[1] - 1) * ln_t
            - np.log(params[0])
            - np.log(params[1])
        )

    def dnll(self, data: dict, params: list[NDArray]) -> list[NDArray]:
        t = data["y"]
        ln_t = np.log(t)
        return [
            t ** params[1] - 1 / params[0],
            ln_t * params[0] * (t ** params[1]) - ln_t - 1 / params[1],
        ]

    def d2nll(self, data: dict, params: list[NDArray]) -> list[list[NDArray]]:
        t = data["y"]
        ln_t = np.log(t)
        return [
            [1 / params[0] ** 2, ln_t * (t ** params[1])],
            [
                ln_t * (t ** params[1]),
                1 / params[1] ** 2 + params[0] * (ln_t**2) * (t ** params[1]),
            ],
        ]

    def get_ui(self, params: list[NDArray], bounds: tuple[float, float]) -> NDArray:
        scale = 1 / params[0] ** (1 / params[1])
        return [
            weibull_min.ppf(bounds[0], c=params[1], scale=scale),
            weibull_min.ppf(bounds[1], c=params[1], scale=scale),
        ]
