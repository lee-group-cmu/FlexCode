import flexcode
import numpy as np
import xgboost as xgb
from flexcode.regression_models import NN, RandomForest, XGBoost, CustomModel


BUMP_THRESHOLD_GRID = np.linspace(0, 0.2, 3)
SHARPEN_GRID = np.linspace(0.5, 1.5, 3)

def generate_data(n_draws):
    """Generate data p(z | x) = N(x, 1)

    Parameters
    ----------
    n_draws : int
        number of samples to generate

    Returns
    -------
    x : List[float]
        Samples drawn from a 0, 1 normal distribution
    z : List[float]
        Samples drawn from a `x`, 1 normal distribution. Where `x` is random variate
        created earlier.
    """
    x = np.random.normal(0, 1, n_draws)
    z = np.random.normal(x, 1, n_draws)
    return x, z

def test_coef_predict_same_as_predict_NN():
    x_train, z_train = generate_data(1000)
    x_validation, z_validation = generate_data(1000)
    x_test, z_test = generate_data(1000)

    # Parameterize model
    model = flexcode.FlexCodeModel(NN, max_basis=31, basis_system="cosine",
                                    regression_params={"k": [5, 10, 20, 25, 30]})

    # Fit and tune model
    model.fit(x_train, z_train)
    model.tune(x_validation, z_validation,
                bump_threshold_grid = BUMP_THRESHOLD_GRID,
                sharpen_grid = SHARPEN_GRID)

    cdes_predict, z_grid = model.predict(x_test, n_grid=200)

    coefs = model.predict_coefs(x_test)
    cdes_coefs = coefs.evaluate(z_grid)

    assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 1e-4


def test_coef_predict_same_as_predict_RF():
    x_train, z_train = generate_data(1000)
    x_validation, z_validation = generate_data(1000)
    x_test, _ = generate_data(1000)

    # Parameterize model
    model = flexcode.FlexCodeModel(RandomForest, max_basis=31, basis_system="cosine",
                                   regression_params={"n_estimators": [10, 20, 30], 'min_samples_split': [2, 5]})

    # Fit and tune model
    model.fit(x_train, z_train)
    model.tune(x_validation, z_validation,
                bump_threshold_grid = BUMP_THRESHOLD_GRID,
                sharpen_grid = SHARPEN_GRID)

    cdes_predict, z_grid = model.predict(x_test, n_grid=200)

    coefs = model.predict_coefs(x_test)
    cdes_coefs = coefs.evaluate(z_grid)

    assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 1e-4


def test_coef_predict_same_as_predict_XGB():
    x_train, z_train = generate_data(1000)
    x_validation, z_validation = generate_data(1000)
    x_test, z_test = generate_data(1000)

    # Parameterize model
    model = flexcode.FlexCodeModel(XGBoost, max_basis=31, basis_system="cosine",
                                    regression_params={"max_depth": [3, 5, 8],
                                                    'eta': [0.1, 0.2, 0.5]})

    # Fit and tune model
    model.fit(x_train, z_train)
    model.tune(x_validation, z_validation,
                bump_threshold_grid = BUMP_THRESHOLD_GRID,
                sharpen_grid = SHARPEN_GRID)

    cdes_predict, z_grid = model.predict(x_test, n_grid=200)

    coefs = model.predict_coefs(x_test)
    cdes_coefs = coefs.evaluate(z_grid)

    assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 1e-4


def test_coef_predict_same_as_predict_custom_model():
    x_train, z_train = generate_data(1000)
    x_validation, z_validation = generate_data(1000)
    x_test, z_test = generate_data(5000)

    # Parameterize model
    custom_model = xgb.XGBRegressor
    model = flexcode.FlexCodeModel(CustomModel, max_basis=31, basis_system="cosine",
                                    regression_params={"max_depth": [3, 5, 8],
                                                    'eta': [0.1, 0.2, 0.5]},
                                    custom_model=custom_model)

    # Fit and tune model
    model.fit(x_train, z_train)
    model.tune(x_validation, z_validation,
                bump_threshold_grid = BUMP_THRESHOLD_GRID,
                sharpen_grid = SHARPEN_GRID)

    cdes_predict, z_grid = model.predict(x_test, n_grid=200)

    coefs = model.predict_coefs(x_test)
    cdes_coefs = coefs.evaluate(z_grid)

    assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 1e-4