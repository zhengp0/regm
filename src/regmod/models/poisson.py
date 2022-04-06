"""
Poisson Model
"""
from typing import Callable, List, Tuple, Union

import numpy as np
from msca.linalg.matrix import asmatrix
from numpy import ndarray
from regmod.data import Data
from regmod.optimizer import msca_optimize
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from scipy.stats import poisson

from .model import Model


class PoissonModel(Model):
    param_names = ("lam",)
    default_param_specs = {"lam": {"inv_link": "exp"}}

    def __init__(self, data: Data, **kwargs):
        if not all(data.obs >= 0):
            raise ValueError("Poisson model requires observations to be non-negagive.")
        super().__init__(data, **kwargs)
        mat = self.mat[0]
        sparsity = (mat == 0).sum() / mat.size
        self.sparse = sparsity > 0.95
        if self.sparse:
            mat = csc_matrix(mat).astype(np.float64)
        self.mat[0] = asmatrix(mat)

        # process constraint matrix
        cmat = block_diag(np.identity(self.size), self.linear_umat)
        cvec = np.hstack([self.uvec, self.linear_uvec])
        index = ~np.isclose(cmat, 0.0).all(axis=1)
        cmat = cmat[index]
        scale = np.abs(cmat).max(axis=1)
        cmat = cmat / scale[:, np.newaxis]
        lb = cvec[0][index] / scale
        ub = cvec[1][index] / scale
        self.cmat = np.vstack([-cmat[~np.isneginf(lb)], cmat[~np.isposinf(ub)]])
        self.cvec = np.hstack([-lb[~np.isneginf(lb)], ub[~np.isposinf(ub)]])
        if self.sparse:
            self.cmat = csc_matrix(self.cmat).astype(np.float64)
        self.cmat = asmatrix(self.cmat)

    @property
    def opt_vcov(self) -> Union[None, ndarray]:
        if self.opt_coefs is None:
            return None
        inv_hessian = np.linalg.pinv(self.hessian(self.opt_coefs).to_numpy())
        jacobian2 = self.jacobian2(self.opt_coefs).to_numpy()
        vcov = inv_hessian.dot(jacobian2)
        vcov = inv_hessian.dot(vcov.T)
        return vcov

    def objective(self, coefs: ndarray) -> float:
        """Objective function.
        Parameters
        ----------
        coefs : ndarray
            Given coefficients.
        Returns
        -------
        float
            Objective value.
        """
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coefs, self.data, mat=self.mat[0]
        )

        weights = self.data.weights*self.data.trim_weights
        obj_params = (
            inv_link.fun(lin_param) - self.data.obs * lin_param
        ) * weights
        return obj_params.sum() + self.objective_from_gprior(coefs)

    def gradient(self, coefs: ndarray) -> ndarray:
        """Gradient function.

        Parameters
        ----------
        coefs : ndarray
            Given coefficients.

        Returns
        -------
        ndarray
            Gradient vector.
        """
        mat = self.mat[0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coefs, self.data, mat=self.mat[0]
        )

        weights = self.data.weights*self.data.trim_weights
        grad_params = (inv_link.dfun(lin_param) - self.data.obs) * weights

        return mat.T.dot(grad_params) + self.gradient_from_gprior(coefs)

    def hessian(self, coefs: ndarray) -> ndarray:
        """Hessian function.

        Parameters
        ----------
        coefs : ndarray
            Given coefficients.

        Returns
        -------
        ndarray
            Hessian matrix.
        """
        mat = self.mat[0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coefs, self.data, mat=self.mat[0]
        )

        weights = self.data.weights*self.data.trim_weights
        hess_params = (inv_link.d2fun(lin_param)) * weights

        scaled_mat = mat.scale_rows(hess_params)

        hess_mat = mat.T.dot(scaled_mat)
        hess_mat_gprior = type(hess_mat)(self.hessian_from_gprior())
        return hess_mat + hess_mat_gprior

    def jacobian2(self, coefs: ndarray) -> ndarray:
        """Jacobian function.

        Parameters
        ----------
        coefs : ndarray
            Given coefficients.

        Returns
        -------
        ndarray
            Jacobian matrix.
        """
        mat = self.mat[0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coefs, self.data, mat=self.mat[0]
        )
        param = inv_link.fun(lin_param)
        dparam = mat.scale_rows(inv_link.dfun(lin_param))
        grad_param = 1.0 - self.data.obs/param
        weights = self.data.weights*self.data.trim_weights
        jacobian = dparam.T.scale_cols(weights*grad_param)
        hess_mat_gprior = type(jacobian)(self.hessian_from_gprior())
        jacobian2 = jacobian.dot(jacobian.T) + hess_mat_gprior
        return jacobian2

    def fit(self,
            optimizer: Callable = msca_optimize,
            **optimizer_options):
        """Fit function.

        Parameters
        ----------
        optimizer : Callable, optional
            Model solver, by default scipy_optimize.
        """
        optimizer(self, **optimizer_options)

    def nll(self, params: List[ndarray]) -> ndarray:
        return params[0] - self.data.obs*np.log(params[0])

    def dnll(self, params: List[ndarray]) -> List[ndarray]:
        return [1.0 - self.data.obs/params[0]]

    def d2nll(self, params: List[ndarray]) -> List[List[ndarray]]:
        return [[self.data.obs/params[0]**2]]

    def get_ui(self, params: List[ndarray], bounds: Tuple[float, float]) -> ndarray:
        mean = params[0]
        return [poisson.ppf(bounds[0], mu=mean),
                poisson.ppf(bounds[1], mu=mean)]
