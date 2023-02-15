"""
Test optimizer module
"""
import numpy as np
import pandas as pd
import pytest

from regmod.models import GaussianModel
from regmod.optimizer import scipy_optimize
from regmod.utils import SplineSpecs
from regmod.variable import SplineVariable, Variable


@pytest.mark.parametrize("seed", [123, 456, 789])
def test_scipy_optimizer(seed):
    np.random.seed(seed)
    num_obs = 20
    df = pd.DataFrame(
        {
            "obs": np.random.randn(num_obs),
            "cov0": np.random.randn(num_obs),
            "cov1": np.random.randn(num_obs),
        }
    )
    spline_specs = SplineSpecs(
        knots=np.linspace(0.0, 1.0, 5), degree=3, knots_type="rel_domain"
    )

    var_cov0 = Variable(name="cov0")
    var_cov1 = SplineVariable(name="cov1", spline_specs=spline_specs)

    model = GaussianModel(
        y="obs", param_specs={"mu": {"variables": [var_cov0, var_cov1]}}
    )
    data = model._parse(df)

    coef = scipy_optimize(model, data)

    mat = data["mat"][0].to_numpy()
    tr_coef = np.linalg.solve(
        (mat.T * data["weights"]).dot(mat),
        (mat.T * data["weights"]).dot(data["y"]),
    )

    assert np.allclose(coef, tr_coef)
