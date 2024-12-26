"""
Binomial Model
"""

import numpy as np
from scipy.stats import binom

from regmod._typing import Callable, DataFrame, NDArray
from regmod.data import Data
from regmod.optimizer import msca_optimize

from .model import Model
from .utils import get_params, model_post_init


class BinomialModel(Model):
    param_names = ("p",)
    default_param_specs = {"p": {"inv_link": "expit"}}

    def __init__(self, data: Data, **kwargs):
        if not np.all((data.obs >= 0) & (data.obs <= 1)):
            raise ValueError(
                "Binomial model requires observations to be " "between zero and one."
            )
        super().__init__(data, **kwargs)

    def attach_df(self, df: DataFrame):
        super().attach_df(df)
        self.mat[0], self.cmat, self.cvec, self.hmat = model_post_init(
            self.mat[0],
            self.uvec,
            self.linear_umat,
            self.linear_uvec,
            self.gvec,
            self.linear_gmat,
            self.linear_gvec,
        )

    def get_lin_param(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        lin_param = mat.dot(coefs)
        if self.params[0].offset is not None:
            lin_param += self.data.get_cols(self.params[0].offset)
        return lin_param

    def hessian_from_gprior(self) -> NDArray:
        """Hessian matrix from the Gaussian prior.

        Returns
        -------
        Matrix
            Hessian matrix.

        """
        return self.hmat

    def objective(self, coefs: NDArray) -> float:
        """Objective function.
        Parameters
        ----------
        coefs : NDArray
            Given coefficients.

        Returns
        -------
        float
            Objective value.

        """
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(coefs, self.data, mat=self.mat[0])
        param = inv_link.fun(lin_param)

        weights = self.data.weights * self.data.trim_weights
        obj_param = -weights * (
            self.data.obs * np.log(param) + (1 - self.data.obs) * np.log(1 - param)
        )
        return obj_param.sum() + self.objective_from_gprior(coefs)

    def gradient(self, coefs: NDArray) -> NDArray:
        """Gradient function.

        Parameters
        ----------
        coefs : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Gradient vector.

        """
        mat = self.mat[0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(coefs, self.data, mat=self.mat[0])
        param = inv_link.fun(lin_param)
        dparam = inv_link.dfun(lin_param)

        weights = self.data.weights * self.data.trim_weights
        grad_param = weights * (
            (param - self.data.obs) / (param * (1 - param)) * dparam
        )

        return mat.T.dot(grad_param) + self.gradient_from_gprior(coefs)

    def hessian(self, coefs: NDArray) -> NDArray:
        """Hessian function.

        Parameters
        ----------
        coefs : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Hessian matrix.

        """
        mat = self.mat[0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(coefs, self.data, mat=self.mat[0])
        param = inv_link.fun(lin_param)
        dparam = inv_link.dfun(lin_param)
        d2param = inv_link.d2fun(lin_param)

        weights = self.data.weights * self.data.trim_weights
        hess_param = weights * (
            (self.data.obs / param**2 + (1 - self.data.obs) / (1 - param) ** 2)
            * dparam**2
            + (param - self.data.obs) / (param * (1 - param)) * d2param
        )

        scaled_mat = mat.scale_rows(hess_param)
        hess_mat = mat.T.dot(scaled_mat)
        hess_mat_gprior = type(hess_mat)(self.hessian_from_gprior())
        return hess_mat + hess_mat_gprior

    def jacobian2(self, coefs: NDArray) -> NDArray:
        """Jacobian function.

        Parameters
        ----------
        coefs : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Jacobian matrix.

        """
        mat = self.mat[0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(coefs, self.data, mat=self.mat[0])
        param = inv_link.fun(lin_param)
        dparam = inv_link.dfun(lin_param)
        weights = self.data.weights * self.data.trim_weights
        grad_param = weights * (
            (param - self.data.obs) / (param * (1 - param)) * dparam
        )
        jacobian = mat.T.scale_cols(grad_param)
        hess_mat_gprior = type(jacobian)(self.hessian_from_gprior())
        jacobian2 = jacobian.dot(jacobian.T) + hess_mat_gprior
        return jacobian2

    def get_pearson_residuals(self, coefs: NDArray) -> NDArray:
        pred = self.params[0].get_param(coefs, self.data, mat=self.mat[0])
        pred_sd = np.sqrt(pred * (1 - pred) / self.data.weights)

        return (self.data.obs - pred) / pred_sd

    def fit(self, optimizer: Callable = msca_optimize, **optimizer_options):
        """Fit function.

        Parameters
        ----------
        optimizer : Callable, optional
            Model solver, by default scipy_optimize.

        """
        super().fit(optimizer=optimizer, **optimizer_options)

    def nll(self, params: list[NDArray]) -> NDArray:
        return -(
            self.data.obs * np.log(params[0])
            + (1 - self.data.obs) * np.log(1.0 - params[0])
        )

    def dnll(self, params: list[NDArray]) -> list[NDArray]:
        return [-(self.data.obs / params[0] - (1 - self.data.obs) / (1.0 - params[0]))]

    def d2nll(self, params: list[NDArray]) -> list[list[NDArray]]:
        return [
            [
                self.data.obs / params[0] ** 2
                + (1 - self.data.obs) / (1.0 - params[0]) ** 2
            ]
        ]

    def get_ui(self, params: list[NDArray], bounds: tuple[float, float]) -> NDArray:
        p = params[0]
        n = self.obs_sample_sizes
        return [binom.ppf(bounds[0], n=n, p=p), binom.ppf(bounds[1], n=n, p=p)]


class CanonicalBinomialModel(BinomialModel):
    def __init__(self, data: Data, **kwargs):
        super().__init__(data, **kwargs)
        if self.params[0].inv_link.name != "expit":
            raise ValueError(
                "Canonical Binomial model requires inverse link to be expit."
            )

    def objective(self, coefs: NDArray) -> float:
        weights = self.data.weights * self.data.trim_weights
        y = self.get_lin_param(coefs)

        prior_obj = self.objective_from_gprior(coefs)
        likli_obj = weights.dot(np.log(1 + np.exp(-y)) + (1 - self.data.obs) * y)
        return prior_obj + likli_obj

    def gradient(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        z = np.exp(self.get_lin_param(coefs))

        prior_grad = self.gradient_from_gprior(coefs)
        likli_grad = mat.T.dot(weights * (z / (1 + z) - self.data.obs))
        return prior_grad + likli_grad

    def hessian(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        z = np.exp(self.get_lin_param(coefs))
        likli_hess_scale = weights * (z / ((1 + z) ** 2))

        likli_hess_right = mat.scale_rows(likli_hess_scale)
        likli_hess = mat.T.dot(likli_hess_right)

        return self.hessian_from_gprior() + likli_hess

    def jacobian2(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        z = np.exp(self.get_lin_param(coefs))
        likli_jac_scale = weights * (z / (1 + z) - self.data.obs)

        likli_jac = mat.T.scale_cols(likli_jac_scale)
        likli_jac2 = likli_jac.dot(likli_jac.T)
        return self.hessian_from_gprior() + likli_jac2


def create_binomial_model(data: Data, **kwargs) -> BinomialModel:
    params = get_params(
        params=kwargs.get("params"),
        param_specs=kwargs.get("param_specs"),
        default_param_specs=BinomialModel.default_param_specs,
    )

    if params[0].inv_link.name == "expit":
        return CanonicalBinomialModel(data, params=params)
    return BinomialModel(data, params=params)
