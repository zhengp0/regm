"""
Test Gaussian Model
"""
import numpy as np
import pandas as pd
import pytest

from regmod.function import fun_dict
from regmod.models import GaussianModel
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
    df = pd.DataFrame(
        {
            "obs": np.random.randn(num_obs),
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
    model = GaussianModel(
        y="obs", param_specs={"mu": {"variables": [var_cov0, var_cov1]}}
    )
    return model


def test_model_result(model):
    assert model.result is None
    assert model.coef is None
    assert model.vcov is None


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
    assert my_obj > 0.0


@pytest.mark.parametrize("inv_link", ["identity", "exp"])
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


@pytest.mark.parametrize("inv_link", ["identity", "exp"])
def test_model_hessian(model, df, inv_link):
    data = model._parse(df)
    model.params[0].inv_link = fun_dict[inv_link]
    coef = np.random.randn(model.size)
    coef_c = coef + 0j
    my_hess = model.hessian(data, coef).to_numpy()
    tr_hess = np.zeros((model.size, model.size))
    for i in range(model.size):
        for j in range(model.size):
            coef_c[j] += 1e-16j
            tr_hess[i][j] = model.gradient(data, coef_c).imag[i] / 1e-16
            coef_c[j] -= 1e-16j

    assert np.allclose(my_hess, tr_hess)


def test_model_get_ui(model, df):
    params = [np.zeros(5)]
    bounds = (0.025, 0.975)
    data = model._parse(df)
    ui = model.get_ui(data, params, bounds)
    assert np.allclose(ui[0], -1.95996)
    assert np.allclose(ui[1], 1.95996)


def test_model_jacobian2(model, df):
    data = model._parse(df)
    beta = np.zeros(model.size)
    jacobian2 = model.jacobian2(data, beta).to_numpy()

    mat = data["mat"][0].to_numpy()
    param = model.get_params(data, beta)[0]
    residual = (data["y"] - param) * np.sqrt(data["weights"])
    jacobian = mat.T * residual
    true_jacobian2 = jacobian.dot(jacobian.T) + model.hessian_from_gprior(data)

    assert np.allclose(jacobian2, true_jacobian2)


def test_model_no_variables():
    num_obs = 5
    df = pd.DataFrame(
        {
            "obs": np.random.randn(num_obs),
            "offset": np.ones(num_obs),
        }
    )
    model = GaussianModel(y="obs", param_specs={"mu": {"offset": "offset"}})
    data = model._parse(df)
    coef = np.array([])
    grad = model.gradient(data, coef)
    hessian = model.hessian(data, coef)
    assert grad.size == 0
    assert hessian.size == 0

    model.fit(df)
    assert model.result == "no parameter to fit"
