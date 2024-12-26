import numpy as np
from msca.linalg.matrix import asmatrix
from scipy.sparse import csc_matrix

from regmod._typing import Matrix, NDArray
from regmod.parameter import Parameter


def model_post_init(
    mat: NDArray,
    uvec: NDArray,
    linear_umat: NDArray,
    linear_uvec: NDArray,
    gvec: NDArray,
    linear_gmat: NDArray,
    linear_gvec: NDArray,
) -> tuple[Matrix, Matrix, NDArray, Matrix]:
    # design matrix
    issparse = mat.size == 0 or ((mat == 0).sum() / mat.size) > 0.95
    if issparse:
        mat = csc_matrix(mat).astype(np.float64)
    mat = asmatrix(mat)

    # constraints
    cmat = np.vstack([np.identity(mat.shape[1]), linear_umat])
    cvec = np.hstack([uvec, linear_uvec])

    index = ~np.isclose(cmat, 0.0).all(axis=1)
    cmat = cmat[index]
    cvec = cvec[:, index]

    if cmat.size > 0:
        scale = np.abs(cmat).max(axis=1)
        cmat = cmat / scale[:, np.newaxis]
        cvec = cvec / scale

    cmat = np.vstack([-cmat[~np.isneginf(cvec[0])], cmat[~np.isposinf(cvec[1])]])
    cvec = np.hstack([-cvec[0][~np.isneginf(cvec[0])], cvec[1][~np.isposinf(cvec[1])]])
    if issparse:
        cmat = csc_matrix(cmat).astype(np.float64)
    cmat = asmatrix(cmat)

    gmat = np.vstack([np.identity(mat.shape[1]), linear_gmat])
    gvec = np.hstack([gvec, linear_gvec])

    if issparse:
        gmat = csc_matrix(gmat).astype(np.float64)
    gmat = asmatrix(gmat)

    hmat = gmat.T.scale_cols(1.0 / gvec[1] ** 2).dot(gmat)
    return mat, cmat, cvec, hmat


def get_params(
    params: list[Parameter] | None = None,
    param_specs: dict[str, dict] | None = None,
    default_param_specs: dict[str, dict] | None = None,
) -> list[Parameter]:
    if params is None and param_specs is None:
        raise ValueError("Please provide `params` or `param_specs`")

    if params is not None:
        return params

    default_param_specs = default_param_specs or {}
    param_specs = {
        key: {**default_param_specs.get(key, {}), **value}
        for key, value in param_specs.items()
    }

    params = [Parameter(key, **value) for key, value in param_specs.items()]
    return params
