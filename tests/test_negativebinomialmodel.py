"""
Test Negative Binomial Model
"""
import numpy as np
import pandas as pd
import pytest

from regmod.function import fun_dict
from regmod.models import NegativeBinomialModel
from regmod.prior import (GaussianPrior, SplineGaussianPrior,
                          SplineUniformPrior, UniformPrior)
from regmod.utils import SplineSpecs
from regmod.variable import SplineVariable, Variable

# pylint:disable=redefined-outer-name


@pytest.fixture
def data():
    num_obs = 5
    df = pd.DataFrame({
        "obs": np.random.rand(num_obs)*10,
        "cov0": np.random.randn(num_obs),
        "cov1": np.random.randn(num_obs)
    })
    return df


@pytest.fixture
def wrong_data():
    num_obs = 5
    df = pd.DataFrame({
        "obs": -np.random.rand(num_obs),
        "cov0": np.random.randn(num_obs),
        "cov1": np.random.randn(num_obs)
    })
    return df


@pytest.fixture
def gprior():
    return GaussianPrior(mean=0.0, sd=1.0)


@pytest.fixture
def uprior():
    return UniformPrior(lb=0.0, ub=1.0)


@pytest.fixture
def spline_specs():
    return SplineSpecs(knots=np.linspace(0.0, 1.0, 5),
                       degree=3,
                       knots_type="rel_domain")


@pytest.fixture
def spline_gprior():
    return SplineGaussianPrior(mean=0.0, sd=1.0, order=1)


@pytest.fixture
def spline_uprior():
    return SplineUniformPrior(lb=0.0, ub=np.inf, order=1)


@pytest.fixture
def var_cov0(gprior, uprior):
    return Variable(name="cov0",
                    priors=[gprior, uprior])


@pytest.fixture
def var_cov1(spline_gprior, spline_uprior, spline_specs):
    return SplineVariable(name="cov1",
                          spline_specs=spline_specs,
                          priors=[spline_gprior, spline_uprior])


@pytest.fixture
def model(data, var_cov0, var_cov1):
    model = NegativeBinomialModel(
        y="obs",
        param_specs={"n": {"variables": [var_cov0]},
                     "p": {"variables": [var_cov1]}}
    )
    model._attach(data)
    return model


def test_model_size(model, var_cov0, var_cov1):
    assert model.size == var_cov0.size + var_cov1.size


def test_uvec(model):
    assert model._data["uvec"].shape == (2, model.size)


def test_gvec(model):
    assert model._data["gvec"].shape == (2, model.size)


def test_linear_uprior(model):
    assert model._data["linear_uvec"].shape[1] == model._data["linear_umat"].shape[0]
    assert model._data["linear_umat"].shape[1] == model.size


def test_linear_gprior(model):
    assert model._data["linear_gvec"].shape[1] == model._data["linear_gmat"].shape[0]
    assert model._data["linear_gmat"].shape[1] == model.size


def test_model_objective(model):
    coefs = np.random.randn(model.size)
    my_obj = model.objective(coefs)
    assert np.isscalar(my_obj)


@pytest.mark.parametrize("inv_link", ["expit", "exp"])
def test_model_gradient(model, inv_link):
    model.params[0].inv_link = fun_dict[inv_link]
    coefs = np.random.randn(model.size)
    coefs_c = coefs + 0j
    my_grad = model.gradient(coefs)
    tr_grad = np.zeros(model.size)
    for i in range(model.size):
        coefs_c[i] += 1e-16j
        tr_grad[i] = model.objective(coefs_c).imag/1e-16
        coefs_c[i] -= 1e-16j
    assert np.allclose(my_grad, tr_grad)


@pytest.mark.parametrize("inv_link", ["expit", "exp"])
def test_model_hessian(model, inv_link):
    model.params[0].inv_link = fun_dict[inv_link]
    coefs = np.random.randn(model.size)
    coefs_c = coefs + 0j
    my_hess = model.hessian(coefs)
    tr_hess = np.zeros((model.size, model.size))
    for i in range(model.size):
        for j in range(model.size):
            coefs_c[j] += 1e-16j
            tr_hess[i][j] = model.gradient(coefs_c).imag[i]/1e-16
            coefs_c[j] -= 1e-16j

    assert np.allclose(my_hess, tr_hess)


def test_wrong_data(wrong_data, var_cov0, var_cov1):
    with pytest.raises(ValueError):
        model = NegativeBinomialModel(
            y="obs",
            param_specs={"n": {"variables": [var_cov0]},
                         "p": {"variables": [var_cov1]}}
        )
        model._attach(wrong_data)


def test_get_ui(model):
    params = [np.full(5, 10), np.full(5, 0.5)]
    bounds = (0.025, 0.975)
    ui = model.get_ui(params, bounds)
    assert np.allclose(ui[0], 3.0)
    assert np.allclose(ui[1], 20.0)


def test_model_no_variables():
    num_obs = 5
    df = pd.DataFrame({
        "obs": np.random.rand(num_obs)*10,
        "offset": np.ones(num_obs),
    })
    model = NegativeBinomialModel(
        y="obs",
        param_specs={"n": {"offset": "offset"}, "p": {"offset": "offset"}}
    )
    model._attach(df)
    coefs = np.array([])
    grad = model.gradient(coefs)
    hessian = model.hessian(coefs)
    assert grad.size == 0
    assert hessian.size == 0

    model.fit(df)
    assert model.opt_result == "no parameter to fit"


def test_model_one_variable():
    num_obs = 5
    df = pd.DataFrame({
        "obs": np.random.rand(num_obs)*10,
        "offset": np.ones(num_obs),
    })
    model = NegativeBinomialModel(
        y="obs",
        param_specs={
            "n": {"offset": "offset"},
            "p": {"variables": [Variable("intercept")]},
        }
    )
    model.fit(df)
    assert model.opt_coefs.size == 1
