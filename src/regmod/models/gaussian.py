"""
Gaussian Model
"""

import numpy as np
from scipy.stats import norm

from regmod._typing import Callable, DataFrame, Matrix, NDArray
from regmod.data import Data
from regmod.optimizer import msca_optimize

from .model import Model
from .utils import get_params, model_post_init


class GaussianModel(Model):
    param_names = ("mu",)
    default_param_specs = {"mu": {"inv_link": "identity"}}

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

    def hessian_from_gprior(self) -> Matrix:
        return self.hmat

    def get_lin_param(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        lin_param = mat.dot(coefs)
        if self.params[0].offset is not None:
            lin_param += self.data.get_cols(self.params[0].offset)
        return lin_param

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
        obj_param = weights * 0.5 * (param - self.data.obs) ** 2
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
        grad_param = weights * (param - self.data.obs) * dparam

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
        hess_param = weights * (dparam**2 + (param - self.data.obs) * d2param)

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
        grad_param = weights * (param - self.data.obs) * dparam
        jacobian = mat.T.scale_cols(grad_param)
        hess_mat_gprior = type(jacobian)(self.hessian_from_gprior())
        jacobian2 = jacobian.dot(jacobian.T) + hess_mat_gprior
        return jacobian2

    def fit(self, optimizer: Callable = msca_optimize, **optimizer_options):
        """Fit function.

        Parameters
        ----------
        optimizer : Callable, optional
            Model solver, by default scipy_optimize.
        """
        super().fit(optimizer=optimizer, **optimizer_options)

    def nll(self, params: list[NDArray]) -> NDArray:
        return 0.5 * (params[0] - self.data.obs) ** 2

    def dnll(self, params: list[NDArray]) -> list[NDArray]:
        return [params[0] - self.data.obs]

    def d2nll(self, params: list[NDArray]) -> list[NDArray]:
        return [[np.ones(self.data.num_obs)]]

    def get_ui(self, params: list[NDArray], bounds: tuple[float, float]) -> NDArray:
        mean = params[0]
        sd = 1.0 / np.sqrt(self.data.weights)
        return [
            norm.ppf(bounds[0], loc=mean, scale=sd),
            norm.ppf(bounds[1], loc=mean, scale=sd),
        ]


class CanonicalGaussianModel(GaussianModel):
    def __init__(self, data: Data, **kwargs):
        super().__init__(data, **kwargs)
        if self.params[0].inv_link.name != "identity":
            raise ValueError(
                "Canonical Gaussian model requires inverse link to be identity."
            )

    def objective(self, coefs: NDArray) -> float:
        weights = self.data.weights * self.data.trim_weights
        y = self.get_lin_param(coefs)

        prior_obj = self.objective_from_gprior(coefs)
        likli_obj = 0.5 * weights.dot((y - self.data.obs) ** 2)
        return prior_obj + likli_obj

    def gradient(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        y = self.get_lin_param(coefs)

        prior_grad = self.gradient_from_gprior(coefs)
        likli_grad = mat.T.dot(weights * (y - self.data.obs))
        return prior_grad + likli_grad

    def hessian(self, coefs: NDArray) -> Matrix:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        likli_hess_scale = weights

        prior_hess = self.hessian_from_gprior()
        likli_hess_right = mat.scale_rows(likli_hess_scale)
        likli_hess = mat.T.dot(likli_hess_right)

        return prior_hess + likli_hess

    def jacobian2(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        y = self.get_lin_param(coefs)
        likli_jac_scale = weights * (y - self.data.obs)

        likli_jac = mat.T.scale_cols(likli_jac_scale)
        likli_jac2 = likli_jac.dot(likli_jac.T)
        return self.hessian_from_gprior() + likli_jac2


def create_gaussian_model(data: Data, **kwargs) -> GaussianModel:
    params = get_params(
        params=kwargs.get("params"),
        param_specs=kwargs.get("param_specs"),
        default_param_specs=GaussianModel.default_param_specs,
    )

    if params[0].inv_link.name == "identity":
        return CanonicalGaussianModel(data, params=params)
    return GaussianModel(data, params=params)
