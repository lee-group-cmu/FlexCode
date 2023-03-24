"""This module tests a particular code path in the FlexCodeModel.fit function.
Each of the test cases specifies a flexcode.FlexCodeModel that is parameterized
with a `regression_params` variable that defines a dictionary with a value that
is an array.

For example, in `test_coef_predict_same_as_predict_NN`, the model is defined with
the regression parameter `"k": [5, 30]`. The effect of this that within the 
`NN.fit` method, the `self.cv_optim` method will be called instead of `self.models.fit`.

This module is structurally similar to `test_models_fit`, but tests a slightly
different code path.
"""

import numpy as np
import xgboost as xgb
from conftest import BUMP_THRESHOLD_GRID, SHARPEN_GRID, generate_data

import flexcode
from flexcode.regression_models import NN, CustomModel, Lasso, RandomForest, XGBoost


def test_coef_predict_same_as_predict_nn():
    # Here we generate 3000 random variates because 1000 (like the other tests)
    # was causing test instability.
    x_train, z_train = generate_data(3000)
    x_validation, z_validation = generate_data(3000)
    x_test, _ = generate_data(3000)

    # Parameterize model
    model = flexcode.FlexCodeModel(NN, max_basis=31, basis_system="cosine", regression_params={"k": [5, 30]})

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
        RandomForest,
        max_basis=31,
        basis_system="cosine",
        regression_params={"n_estimators": [10, 30], "min_samples_split": [2]},
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
        XGBoost, max_basis=31, basis_system="cosine", regression_params={"max_depth": [3, 8], "eta": [0.1]}
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
        Lasso, max_basis=31, basis_system="cosine", regression_params={"alpha": [1.0, 1.1]}
    )

    # Fit and tune model
    model.fit(x_train, z_train)
    model.tune(
        x_validation,
        z_validation,
        bump_threshold_grid=np.linspace(0, 0.2, 3),
        sharpen_grid=np.linspace(0.5, 1.5, 3),
    )

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
        regression_params={"max_depth": [3, 8], "eta": [0.1]},
        custom_model=custom_model,
    )

    # Fit and tune model
    model.fit(x_train, z_train)
    model.tune(x_validation, z_validation, bump_threshold_grid=BUMP_THRESHOLD_GRID, sharpen_grid=SHARPEN_GRID)

    cdes_predict, z_grid = model.predict(x_test, n_grid=200)

    coefs = model.predict_coefs(x_test)
    cdes_coefs = coefs.evaluate(z_grid)

    assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 1e-4
