"""This module tests a particular code path in the FlexCodeModel.fit function.
Each of the test cases specifies a flexcode.FlexCodeModel that is parameterized
with a `regression_params` variable that defines a dictionary with a single float
or integer.

For example, in `test_coef_predict_same_as_predict_nn`, the model is defined with
the regression parameter `"k": 20`. The effect of this that within the 
`NN.fit` method, the `self.models.fit` method will be called instead of `self.cv_optim`.

This module is structurally similar to `test_cv_optim`, but tests a slightly
different code path.
"""

import numpy as np
import pytest
import xgboost as xgb
from conftest import BUMP_THRESHOLD_GRID, SHARPEN_GRID, generate_data

import flexcode
from flexcode.regression_models import NN, CustomModel, Lasso, RandomForest, XGBoost


@pytest.mark.skip(reason="The assertion is meaningless and the test is a duplicate")
def test_example():
    x_train, z_train = generate_data(1000)
    x_validation, z_validation = generate_data(1000)
    x_test, z_test = generate_data(1000)

    # Parameterize model
    model = flexcode.FlexCodeModel(NN, max_basis=31, basis_system="cosine", regression_params={"k": 20})

    # Fit and tune model
    model.fit(x_train, z_train)
    model.tune(x_validation, z_validation, bump_threshold_grid=BUMP_THRESHOLD_GRID, sharpen_grid=SHARPEN_GRID)

    # Estimate CDE loss
    model.estimate_error(x_test, z_test)

    cdes, z_grid = model.predict(x_test, n_grid=200)

    assert True


@pytest.mark.skip(reason="The assertion is meaningless and this test is a duplicate")
def test_unshaped_example():
    x_train, z_train = generate_data(1000)
    x_validation, z_validation = generate_data(1000)
    x_test, z_test = generate_data(1000)

    # Parameterize model
    model = flexcode.FlexCodeModel(NN, max_basis=31, basis_system="cosine", regression_params={"k": 20})

    # Fit and tune model
    model.fit(x_train, z_train)
    model.tune(x_validation, z_validation, bump_threshold_grid=BUMP_THRESHOLD_GRID, sharpen_grid=SHARPEN_GRID)

    # Estimate CDE loss
    model.estimate_error(x_test, z_test)

    cdes, z_grid = model.predict(x_test, n_grid=200)

    assert True


def test_coef_predict_same_as_predict_nn():
    x_train, z_train = generate_data(1000)
    x_validation, z_validation = generate_data(1000)
    x_test, _ = generate_data(1000)

    # Parameterize model
    model = flexcode.FlexCodeModel(NN, max_basis=31, basis_system="cosine", regression_params={"k": 20})

    # Fit and tune model
    model.fit(x_train, z_train)
    model.tune(x_validation, z_validation, bump_threshold_grid=BUMP_THRESHOLD_GRID, sharpen_grid=SHARPEN_GRID)

    cdes_predict, z_grid = model.predict(x_test, n_grid=200)

    coefs = model.predict_coefs(x_test)
    cdes_coefs = coefs.evaluate(z_grid)

    assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 1e-4


def test_coef_predict_same_as_predict_rf():
    x_train, z_train = generate_data(1000)
    x_validation, z_validation = generate_data(1000)
    x_test, _ = generate_data(1000)

    # Parameterize model
    model = flexcode.FlexCodeModel(
        RandomForest, max_basis=31, basis_system="cosine", regression_params={"n_estimators": 10}
    )

    # Fit and tune model
    model.fit(x_train, z_train)
    model.tune(x_validation, z_validation, bump_threshold_grid=BUMP_THRESHOLD_GRID, sharpen_grid=SHARPEN_GRID)

    cdes_predict, z_grid = model.predict(x_test, n_grid=200)

    coefs = model.predict_coefs(x_test)
    cdes_coefs = coefs.evaluate(z_grid)

    assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 1e-4


def test_coef_predict_same_as_predict_xgb():
    x_train, z_train = generate_data(1000)
    x_validation, z_validation = generate_data(1000)
    x_test, _ = generate_data(1000)

    # Parameterize model
    model = flexcode.FlexCodeModel(
        XGBoost, max_basis=31, basis_system="cosine", regression_params={"max_depth": 5}
    )

    # Fit and tune model
    model.fit(x_train, z_train)
    model.tune(x_validation, z_validation, bump_threshold_grid=BUMP_THRESHOLD_GRID, sharpen_grid=SHARPEN_GRID)

    cdes_predict, z_grid = model.predict(x_test, n_grid=200)

    coefs = model.predict_coefs(x_test)
    cdes_coefs = coefs.evaluate(z_grid)

    assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 1e-4


def test_coef_predict_same_as_predict_lasso():
    x_train, z_train = generate_data(1000)
    x_validation, z_validation = generate_data(1000)
    x_test, _ = generate_data(1000)

    # Parameterize model
    model = flexcode.FlexCodeModel(
        Lasso, max_basis=31, basis_system="cosine", regression_params={"alpha": 1.0}
    )

    # Fit and tune model
    model.fit(x_train, z_train)
    model.tune(x_validation, z_validation, bump_threshold_grid=BUMP_THRESHOLD_GRID, sharpen_grid=SHARPEN_GRID)

    cdes_predict, z_grid = model.predict(x_test, n_grid=200)

    coefs = model.predict_coefs(x_test)
    cdes_coefs = coefs.evaluate(z_grid)

    assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 0.5


def test_coef_predict_same_as_predict_custom_model():
    x_train, z_train = generate_data(1000)
    x_validation, z_validation = generate_data(1000)
    x_test, _ = generate_data(1000)

    # Parameterize model
    custom_model = xgb.XGBRegressor
    model = flexcode.FlexCodeModel(
        CustomModel,
        max_basis=31,
        basis_system="cosine",
        regression_params={"max_depth": 5},
        custom_model=custom_model,
    )

    # Fit and tune model
    model.fit(x_train, z_train)
    model.tune(x_validation, z_validation, bump_threshold_grid=BUMP_THRESHOLD_GRID, sharpen_grid=SHARPEN_GRID)

    cdes_predict, z_grid = model.predict(x_test, n_grid=200)

    coefs = model.predict_coefs(x_test)
    cdes_coefs = coefs.evaluate(z_grid)

    assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 1e-4
