import flexcode
import numpy as np
from flexcode.regression_models import NN, RandomForest

def test_coef_predict_same_as_predict_NN():
    # Generate data p(z | x) = N(x, 1)
  def generate_data(n_draws):
    x = np.random.normal(0, 1, n_draws)
    z = np.random.normal(x, 1, n_draws)
    return x, z

  x_train, z_train = generate_data(5000)
  x_validation, z_validation = generate_data(5000)
  x_test, z_test = generate_data(5000)

  # Parameterize model
  model = flexcode.FlexCodeModel(NN, max_basis=31, basis_system="cosine",
                                 regression_params={"k": [5, 10, 20, 25, 30]})

  # Fit and tune model
  model.fit(x_train, z_train)
  model.tune(x_validation, z_validation,
             bump_threshold_grid = np.linspace(0, 0.2, 3),
             sharpen_grid = np.linspace(0.5, 1.5, 3))

  cdes_predict, z_grid = model.predict(x_test, n_grid=200)

  coefs = model.predict_coefs(x_test)
  cdes_coefs = coefs.evaluate(z_grid)

  assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 1e-4


def test_coef_predict_same_as_predict_RF():
    # Generate data p(z | x) = N(x, 1)
  def generate_data(n_draws):
    x = np.random.normal(0, 1, n_draws)
    z = np.random.normal(x, 1, n_draws)
    return x, z

  x_train, z_train = generate_data(5000)
  x_validation, z_validation = generate_data(5000)
  x_test, z_test = generate_data(5000)

  # Parameterize model
  model = flexcode.FlexCodeModel(RandomForest, max_basis=31, basis_system="cosine",
                                 regression_params={"n_estimators": [10, 20, 30], 'min_samples_split': [2, 5]})

  # Fit and tune model
  model.fit(x_train, z_train)
  model.tune(x_validation, z_validation,
             bump_threshold_grid = np.linspace(0, 0.2, 3),
             sharpen_grid = np.linspace(0.5, 1.5, 3))

  cdes_predict, z_grid = model.predict(x_test, n_grid=200)

  coefs = model.predict_coefs(x_test)
  cdes_coefs = coefs.evaluate(z_grid)

  assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 1e-4