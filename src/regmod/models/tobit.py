"""
Tobit Model
"""
# pylint: disable=C0103
from typing import List, Optional

import jax.numpy as jnp
from jax import grad, hessian, jit, lax
from jax.numpy import DeviceArray
from jax.scipy.stats.norm import logcdf, logpdf
from numpy.typing import ArrayLike
from pandas import DataFrame

from regmod.function import SmoothFunction

from .model import Model

identity_jax = SmoothFunction(
    name="identity_jax",
    fun=jnp.array,
    inv_fun=jnp.array,
    dfun=jnp.ones_like,
    d2fun=jnp.zeros_like
)


exp_jax = SmoothFunction(
    name="exp_jax",
    fun=jnp.exp,
    inv_fun=jnp.log,
    dfun=jnp.exp,
    d2fun=jnp.exp
)


class TobitModel(Model):
    """Tobit model class.

    A tobit regression model is used for data generated by a Gaussian
    distribution with all negative elements set to 0 (i.e., a censored
    or rectified Gaussian distribution).

    """

    param_names = ("mu", "sigma")
    default_param_specs = {
        "mu": {"inv_link": 'identity'},
        "sigma": {"inv_link": 'exp'}
    }

    def __init__(self, *args, **kwargs) -> None:
        """Initialize tobit model.

        Parameters
        ----------
        data : Data
            Training data.

        Raises
        ------
        ValueError
            If negative observations present in `data`.

        Notes
        -----
        User input for parameter attribute `inv_link` is ingored and
        defaults used instead.

        """
        super().__init__(*args, **kwargs)

        # Use JAX inv_link functions
        for param in self.params:
            link_name = param.inv_link.name
            if link_name == 'identity':
                param.inv_link = identity_jax
            elif link_name == 'exp':
                param.inv_link = exp_jax
            else:
                msg = f"No JAX implementation of {link_name} inv_link."
                raise ValueError(msg)

    def _attach(self, df: DataFrame, require_y: bool = True) -> None:
        """Extract training data from data frame.

        Parameters
        ----------
        df : DataFrame
            Training data.

        """
        super()._attach(df, require_y=require_y)
        if require_y and not jnp.all(self._data["y"] >= 0):
            raise ValueError(
                "Tobit model requires non-negative observations."
            )

        self._data["mat"] = [jnp.asarray(mat) for mat in self._data["mat"]]
        self._data["uvec"] = jnp.asarray(self._data["uvec"])
        self._data["gvec"] = jnp.asarray(self._data["gvec"])
        self._data["linear_uvec"] = jnp.asarray(self._data["linear_uvec"])
        self._data["linear_gvec"] = jnp.asarray(self._data["linear_gvec"])
        self._data["linear_umat"] = jnp.asarray(self._data["linear_umat"])
        self._data["linear_gmat"] = jnp.asarray(self._data["linear_gmat"])

        # Data structures for objective
        self._data["offset"] = [
            jnp.asarray(df[param.offset])
            if param.offset is not None else jnp.zeros(df.shape[0])
            for param in self.params
        ]
        self._data["y"] = jnp.asarray(self._data["y"])
        self._data["weights"] = jnp.asarray(self.trim_weights*self._data["weights"])

    def objective(self, coefs: ArrayLike) -> float:
        """Get negative log likelihood wrt coefficients.

        Parameters
        ----------
        coefs : array_like
            Model coefficients.

        Returns
        -------
        float
            Negative log likelihood.

        """
        coef_list = [coefs[index] for index in self.indices]
        link_list = [param.inv_link.name == 'exp_jax' for param in self.params]
        return _objective(coef_list, link_list, self._data["mat"], self._data["offset"],
                          self._data["y"], self._data["weights"], self._data["gvec"],
                          self._data["linear_gvec"], self._data["linear_gmat"])

    def gradient(self, coefs: ArrayLike) -> DeviceArray:
        """Get gradient of negative log likelihood wrt coefficients.

        Parameters
        ----------
        coefs : array_like
            Model coefficients.

        Returns
        -------
        DeviceArray
            Gradient of negative log likelihood.

        """
        coef_list = [coefs[index] for index in self.indices]
        link_list = [param.inv_link.name == 'exp_jax' for param in self.params]
        temp = _gradient(coef_list, link_list, self._data["mat"], self._data["offset"],
                         self._data["y"], self._data["weights"], self._data["gvec"],
                         self._data["linear_gvec"], self._data["linear_gmat"])
        return jnp.concatenate(temp)

    def hessian(self, coefs: ArrayLike) -> DeviceArray:
        """Get hessian of negative log likelihood wrt coefficients.

        Parameters
        ----------
        coefs : array_like
            Model coefficients.

        Returns
        -------
        DeviceArray
            Hessian of negative log likelihood.

        """
        coef_list = [coefs[index] for index in self.indices]
        link_list = [param.inv_link.name == 'exp_jax' for param in self.params]
        temp = _hessian(coef_list, link_list, self._data["mat"], self._data["offset"],
                        self._data["y"], self._data["weights"], self._data["gvec"],
                        self._data["linear_gvec"], self._data["linear_gmat"])
        hess = jnp.concatenate([
            jnp.concatenate(temp[0], axis=1),
            jnp.concatenate(temp[1], axis=1)
        ], axis=0)
        return hess

    def nll(self, params: List[ArrayLike]) -> DeviceArray:
        """Get terms of negative log likelihood wrt parameters.

        Parameters
        ----------
        params : list[array_like]
            Model parameters.

        Returns
        -------
        DeviceArray
            Terms of negative log likelihood.

        """
        return _nll(self._data["y"], params)

    def dnll(self, params: List[ArrayLike]) -> List[DeviceArray]:
        """Get derivative of negative log likelihood wrt parameters.

        Parameters
        ----------
        params : list[array_like]
            Model parameters.

        Returns
        -------
        list[DeviceArray]
            Derivatives of negative log likelihood.

        """
        return _dnll(self._data["y"], params)

    def get_vcov(self, coefs: ArrayLike) -> DeviceArray:
        """Get variance-covariance matrix.

        Parameters
        ----------
        coefs : array_like
            Model coefficients.

        Returns
        -------
        DeviceArray
            Variance-covariance matrix.

        Notes
        -----
        Currently does not warn for singular hessian or jacobian,
        unlike other RegMod models.

        """
        H = self.hessian(coefs)
        J = self.jacobian2(coefs)
        inv_H = jnp.linalg.inv(H)
        return inv_H.dot(J.dot(inv_H.T))

    def predict(self, df: DataFrame = None) -> DataFrame:
        """Predict mu, sigma, and censored mu.

        Parameters
        ----------
        df : DataFrame, optional
            Prediction data. If None, use training data.

        Returns
        -------
        DataFrame
            Data frame with predicted parameters.

        """
        df = super().predict(df)
        mu = jnp.asarray(df["mu"])
        df["mu_censored"] = jnp.where(mu > 0, mu, 0)
        return df


@jit
def _objective(coef_list: List[ArrayLike], link_list: List[bool],
               mat: List[ArrayLike], offset: List[ArrayLike], y: ArrayLike,
               weights: ArrayLike, gvec: ArrayLike, linear_gvec: ArrayLike,
               linear_gmat: ArrayLike) -> float:
    """Get negative log likelihood wrt coefficients.

    Parameters
    ----------
    coefs : list of array_like
        Model coefficients for each parameter.
    link_list : list of bool
        True if inv_link is exp_jax for each parameter.
    mat : list of array_like
        Design matrices for each parameter.
    offset : list of array_like
        Offset vector for each parameter.
    y : array_like
        Vector of observations.
    weights : array_like
        Vector of weights for each observation.
    gvec : array_like
        Direct Gaussian prior vector.
    linear_gvec : array_like
        Linear Gaussian prior vector.
    linear_gmat : array_like
        Linear Gaussian prior design matrix.

    Returns
    -------
    float
        Negative log likelihood.

    """
    # Get objective from parameters
    param_list = []
    for ii in range(len(coef_list)):
        param_list.append(
            lax.cond(
                link_list[ii],
                lambda x: jnp.exp(x),
                lambda x: x,
                mat[ii].dot(coef_list[ii]) + offset[ii]
            )
        )
    nll_terms = _nll(y, param_list)
    obj_param = jnp.sum(weights*nll_terms)

    # Get objective from prior
    coefs = jnp.concatenate(coef_list)
    obj_prior = jnp.sum((coefs - gvec[0])**2/gvec[1]**2)/2
    obj_prior = lax.cond(
        linear_gvec.size > 0,
        lambda x: x + 0.5*jnp.sum((linear_gmat.dot(coefs) -
                                   linear_gvec[0])**2/linear_gvec[1]**2),
        lambda x: x,
        obj_prior
    )

    return obj_param + obj_prior


@jit
def _nll(y: ArrayLike, params: List[ArrayLike]) -> DeviceArray:
    """Get terms of negative log likelihood wrt parameters.

    Parameters
    ----------
    y : array_like
        Observations.
    params : list[array_like]
        Model parameters.

    Returns
    -------
    DeviceArray
        Terms of negative log likelihood.

    """
    mu = params[0]
    sigma = params[1]
    pos_term = jnp.log(sigma) - logpdf((y - mu)/sigma)
    npos_term = -logcdf(-mu/sigma)
    return jnp.where(y > 0, pos_term, npos_term)


_gradient = jit(grad(_objective))
_hessian = jit(hessian(_objective))
_dnll = jit(grad(lambda y, params: jnp.sum(_nll(y, params))))
