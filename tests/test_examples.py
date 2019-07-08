import numpy as np
import flexcode
from flexcode.regression_models import NN, RandomForest, XGBoost, Lasso

def test_example():
  # Generate data p(z | x) = N(x, 1)
  def generate_data(n_draws):
    x = np.random.normal(0, 1, n_draws)
    z = np.random.normal(x, 1, n_draws)
    return x.reshape((len(x), 1)), z.reshape((len(z), 1))

  x_train, z_train = generate_data(10000)
  x_validation, z_validation = generate_data(10000)
  x_test, z_test = generate_data(10000)

  # Parameterize model
  model = flexcode.FlexCodeModel(NN, max_basis=31, basis_system="cosine",
                                 regression_params={"k": 20})

  # Fit and tune model
  model.fit(x_train, z_train)
  model.tune(x_validation, z_validation,
             bump_threshold_grid = np.linspace(0, 0.2, 3),
             sharpen_grid = np.linspace(0.5, 1.5, 3))

  # Estimate CDE loss
  model.estimate_error(x_test, z_test)

  cdes, z_grid = model.predict(x_test, n_grid=200)

  assert True

def test_unshaped_example():
  # Generate data p(z | x) = N(x, 1)
  def generate_data(n_draws):
    x = np.random.normal(0, 1, n_draws)
    z = np.random.normal(x, 1, n_draws)
    return x, z

  x_train, z_train = generate_data(10000)
  x_validation, z_validation = generate_data(10000)
  x_test, z_test = generate_data(10000)

  # Parameterize model
  model = flexcode.FlexCodeModel(NN, max_basis=31, basis_system="cosine",
                                 regression_params={"k":20})

  # Fit and tune model
  model.fit(x_train, z_train)
  model.tune(x_validation, z_validation,
             bump_threshold_grid = np.linspace(0, 0.2, 3),
             sharpen_grid = np.linspace(0.5, 1.5, 3))

  # Estimate CDE loss
  model.estimate_error(x_test, z_test)

  cdes, z_grid = model.predict(x_test, n_grid=200)

  assert True

def test_coef_predict_same_as_predict():
    # Generate data p(z | x) = N(x, 1)
  def generate_data(n_draws):
    x = np.random.normal(0, 1, n_draws)
    z = np.random.normal(x, 1, n_draws)
    return x, z

  x_train, z_train = generate_data(10000)
  x_validation, z_validation = generate_data(10000)
  x_test, z_test = generate_data(10000)

  # Parameterize model
  model = flexcode.FlexCodeModel(NN, max_basis=31, basis_system="cosine",
                                 regression_params={"k":20})

  # Fit and tune model
  model.fit(x_train, z_train)
  model.tune(x_validation, z_validation,
             bump_threshold_grid = np.linspace(0, 0.2, 3),
             sharpen_grid = np.linspace(0.5, 1.5, 3))

  cdes_predict, z_grid = model.predict(x_test, n_grid=200)

  coefs = model.predict_coefs(x_test)
  cdes_coefs = coefs.evaluate(z_grid)

  assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 1e-4

def test_coef_predict_same_as_predict_rf():

    # Generate data p(z | x) = N(x, 1)
    def generate_data(n_draws):
      x = np.random.normal(0, 1, n_draws)
      z = np.random.normal(x, 1, n_draws)
      return x, z

    x_train, z_train = generate_data(10000)
    x_validation, z_validation = generate_data(10000)
    x_test, z_test = generate_data(10000)

    # Parameterize model
    model = flexcode.FlexCodeModel(RandomForest, max_basis=31, basis_system="cosine",
                                   regression_params={"n_estimators": 10})

    # Fit and tune model
    model.fit(x_train, z_train)
    model.tune(x_validation, z_validation,
               bump_threshold_grid=np.linspace(0, 0.2, 3),
               sharpen_grid=np.linspace(0.5, 1.5, 3))

    cdes_predict, z_grid = model.predict(x_test, n_grid=200)

    coefs = model.predict_coefs(x_test)
    cdes_coefs = coefs.evaluate(z_grid)

    assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 1e-4


def test_coef_predict_same_as_predict_xgb():
  # Generate data p(z | x) = N(x, 1)
  def generate_data(n_draws):
    x = np.random.normal(0, 1, n_draws)
    z = np.random.normal(x, 1, n_draws)
    return x, z

  x_train, z_train = generate_data(10000)
  x_validation, z_validation = generate_data(10000)
  x_test, z_test = generate_data(10000)

  # Parameterize model
  model = flexcode.FlexCodeModel(XGBoost, max_basis=31, basis_system="cosine",
                                 regression_params={"max_depth": 5})

  # Fit and tune model
  model.fit(x_train, z_train)
  model.tune(x_validation, z_validation,
             bump_threshold_grid=np.linspace(0, 0.2, 3),
             sharpen_grid=np.linspace(0.5, 1.5, 3))

  cdes_predict, z_grid = model.predict(x_test, n_grid=200)

  coefs = model.predict_coefs(x_test)
  cdes_coefs = coefs.evaluate(z_grid)

  assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 1e-4


def test_coef_predict_same_as_predict_lasso():
  # Generate data p(z | x) = N(x, 1)
  def generate_data(n_draws):
    x = np.random.normal(0, 1, n_draws)
    z = np.random.normal(x, 1, n_draws)
    return x, z

  x_train, z_train = generate_data(10000)
  x_validation, z_validation = generate_data(10000)
  x_test, z_test = generate_data(10000)

  # Parameterize model
  model = flexcode.FlexCodeModel(Lasso, max_basis=31, basis_system="cosine",
                                 regression_params={"alpha": 1.0})

  # Fit and tune model
  model.fit(x_train, z_train)
  model.tune(x_validation, z_validation,
             bump_threshold_grid=np.linspace(0, 0.2, 3),
             sharpen_grid=np.linspace(0.5, 1.5, 3))

  cdes_predict, z_grid = model.predict(x_test, n_grid=200)

  coefs = model.predict_coefs(x_test)
  cdes_coefs = coefs.evaluate(z_grid)

  assert np.max(np.abs(cdes_predict - cdes_coefs)) <= 0.1
