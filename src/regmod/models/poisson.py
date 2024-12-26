"""
Poisson Model
"""

import numpy as np
from scipy.stats import poisson

from regmod._typing import Callable, DataFrame, Matrix, NDArray
from regmod.data import Data
from regmod.optimizer import msca_optimize

from .model import Model
from .utils import get_params, model_post_init


class PoissonModel(Model):
    param_names = ("lam",)
    default_param_specs = {"lam": {"inv_link": "exp"}}

    def __init__(self, data: Data, **kwargs):
        if not all(data.obs >= 0):
            raise ValueError("Poisson model requires observations to be non-negagive.")
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

    def hessian_from_gprior(self):
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
        obj_param = weights * (param - self.data.obs * np.log(param))
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
        grad_param = weights * (1 - self.data.obs / param) * dparam

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
            self.data.obs / param**2 * dparam**2 + (1 - self.data.obs / param) * d2param
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
        grad_param = weights * (1.0 - self.data.obs / param) * dparam
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
        return params[0] - self.data.obs * np.log(params[0])

    def dnll(self, params: list[NDArray]) -> list[NDArray]:
        return [1.0 - self.data.obs / params[0]]

    def d2nll(self, params: list[NDArray]) -> list[list[NDArray]]:
        return [[self.data.obs / params[0] ** 2]]

    def get_ui(self, params: list[NDArray], bounds: tuple[float, float]) -> NDArray:
        mean = params[0]
        return [poisson.ppf(bounds[0], mu=mean), poisson.ppf(bounds[1], mu=mean)]


class CanonicalPoissonModel(PoissonModel):
    def __init__(self, data: Data, **kwargs):
        super().__init__(data, **kwargs)
        if self.params[0].inv_link.name != "exp":
            raise ValueError("Canonical Poisson model requires inverse link to be exp.")

    def objective(self, coefs: NDArray) -> float:
        weights = self.data.weights * self.data.trim_weights
        y = self.get_lin_param(coefs)
        z = np.exp(y)

        prior_obj = self.objective_from_gprior(coefs)
        likli_obj = weights.dot(z - self.data.obs * y)
        return prior_obj + likli_obj

    def gradient(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        z = np.exp(self.get_lin_param(coefs))

        prior_grad = self.gradient_from_gprior(coefs)
        likli_grad = mat.T.dot(weights * (z - self.data.obs))
        return prior_grad + likli_grad

    def hessian(self, coefs: NDArray) -> Matrix:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        z = np.exp(self.get_lin_param(coefs))
        likli_hess_scale = weights * z

        prior_hess = self.hessian_from_gprior()
        likli_hess_right = mat.scale_rows(likli_hess_scale)
        likli_hess = mat.T.dot(likli_hess_right)

        return prior_hess + likli_hess

    def jacobian2(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        z = np.exp(self.get_lin_param(coefs))
        likli_jac_scale = weights * (z - self.data.obs)

        likli_jac = mat.T.scale_cols(likli_jac_scale)
        likli_jac2 = likli_jac.dot(likli_jac.T)
        return self.hessian_from_gprior() + likli_jac2


def create_poisson_model(data: Data, **kwargs) -> PoissonModel:
    params = get_params(
        params=kwargs.get("params"),
        param_specs=kwargs.get("param_specs"),
        default_param_specs=PoissonModel.default_param_specs,
    )

    if params[0].inv_link.name == "exp":
        return CanonicalPoissonModel(data, params=params)
    return PoissonModel(data, params=params)
