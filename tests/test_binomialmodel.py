"""
Test Binomial Model
"""
import numpy as np
import pandas as pd
import pytest

from regmod.function import fun_dict
from regmod.models import BinomialModel
from regmod.prior import (
    GaussianPrior,
    SplineGaussianPrior,
    SplineUniformPrior,
    UniformPrior,
)
from regmod.utils import SplineSpecs
from regmod.variable import SplineVariable, Variable

# pylint:disable=redefined-outer-name


@pytest.fixture
def df():
    num_obs = 5
    obs = np.random.rand(num_obs)
    sample_size = np.random.poisson(lam=5, size=num_obs)
    df = pd.DataFrame(
        {
            "obs": obs,
            "sample_size": sample_size,
            "cov0": np.random.randn(num_obs),
            "cov1": np.random.randn(num_obs),
        }
    )
    return df


@pytest.fixture
def wrong_df():
    num_obs = 5
    df = pd.DataFrame(
        {
            "obs": np.random.rand(num_obs) * 10,
            "cov0": np.random.randn(num_obs),
            "cov1": np.random.randn(num_obs),
        }
    )
    return df


@pytest.fixture
def gprior():
    return GaussianPrior(mean=0.0, sd=1.0)


@pytest.fixture
def uprior():
    return UniformPrior(lb=0.0, ub=1.0)


@pytest.fixture
def spline_specs():
    return SplineSpecs(
        knots=np.linspace(0.0, 1.0, 5), degree=3, knots_type="rel_domain"
    )


@pytest.fixture
def spline_gprior():
    return SplineGaussianPrior(mean=0.0, sd=1.0, order=1)


@pytest.fixture
def spline_uprior():
    return SplineUniformPrior(lb=0.0, ub=np.inf, order=1)


@pytest.fixture
def var_cov0(gprior, uprior):
    return Variable(name="cov0", priors=[gprior, uprior])


@pytest.fixture
def var_cov1(spline_gprior, spline_uprior, spline_specs):
    return SplineVariable(
        name="cov1", spline_specs=spline_specs, priors=[spline_gprior, spline_uprior]
    )


@pytest.fixture
def model(var_cov0, var_cov1):
    model = BinomialModel(
        y="obs", param_specs={"p": {"variables": [var_cov0, var_cov1]}}
    )
    return model


def test_model_size(model, var_cov0, var_cov1):
    assert model.size == var_cov0.size + var_cov1.size


def test_uvec(model, df):
    data = model._parse(df)
    assert data["uvec"].shape == (2, model.size)


def test_gvec(model, df):
    data = model._parse(df)
    assert data["gvec"].shape == (2, model.size)


def test_linear_uprior(model, df):
    data = model._parse(df)
    assert data["linear_uvec"].shape[1] == data["linear_umat"].shape[0]
    assert data["linear_umat"].shape[1] == model.size


def test_linear_gprior(model, df):
    data = model._parse(df)
    assert data["linear_gvec"].shape[1] == data["linear_gmat"].shape[0]
    assert data["linear_gmat"].shape[1] == model.size


def test_model_objective(model, df):
    data = model._parse(df)
    coef = np.random.randn(model.size)
    my_obj = model.objective(data, coef)
    assert np.isscalar(my_obj)


@pytest.mark.parametrize("inv_link", ["expit"])
def test_model_gradient(model, df, inv_link):
    data = model._parse(df)
    model.params[0].inv_link = fun_dict[inv_link]
    coef = np.random.randn(model.size)
    coef_c = coef + 0j
    my_grad = model.gradient(data, coef)
    tr_grad = np.zeros(model.size)
    for i in range(model.size):
        coef_c[i] += 1e-16j
        tr_grad[i] = model.objective(data, coef_c).imag / 1e-16
        coef_c[i] -= 1e-16j
    assert np.allclose(my_grad, tr_grad)


@pytest.mark.parametrize("inv_link", ["expit"])
def test_model_hessian(model, df, inv_link):
    data = model._parse(df)
    model.params[0].inv_link = fun_dict[inv_link]
    coef = np.random.randn(model.size)
    coef_c = coef + 0j
    my_hess = model.hessian(data, coef)
    tr_hess = np.zeros((model.size, model.size))
    for i in range(model.size):
        for j in range(model.size):
            coef_c[j] += 1e-16j
            tr_hess[i][j] = model.gradient(data, coef_c).imag[i] / 1e-16
            coef_c[j] -= 1e-16j

    assert np.allclose(my_hess, tr_hess)


def test_wrong_df(wrong_df, var_cov0, var_cov1):
    with pytest.raises(ValueError):
        model = BinomialModel(
            y="obs", param_specs={"p": {"variables": [var_cov0, var_cov1]}}
        )
        model._parse(wrong_df)


def test_get_ui(model):
    model.y_sample_sizes = np.array([100] * 5)
    params = [np.full(5, 0.5)]
    bounds = [0.025, 0.975]
    ui = model.get_ui(params, bounds)
    assert np.allclose(ui[0], 40)
    assert np.allclose(ui[1], 60)


def test_model_no_variables():
    num_obs = 5
    obs = np.random.rand(num_obs)
    sample_size = np.random.poisson(lam=5, size=num_obs)
    df = pd.DataFrame(
        {
            "obs": obs,
            "sample_size": sample_size,
            "offset": np.ones(num_obs),
        }
    )
    model = BinomialModel(y="obs", param_specs={"p": {"offset": "offset"}})
    data = model._parse(df)
    coef = np.array([])
    grad = model.gradient(data, coef)
    hessian = model.hessian(data, coef)
    assert grad.size == 0
    assert hessian.size == 0

    model.fit(df)
    assert model.result == "no parameter to fit"
