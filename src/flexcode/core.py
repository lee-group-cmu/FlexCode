import numpy as np

from .basis_functions import BasisCoefs, evaluate_basis
from .helpers import box_transform, make_grid
from .loss_functions import cde_loss
from .post_processing import *


class FlexCodeModel(object):
    def __init__(
        self,
        model,
        max_basis,
        basis_system="cosine",
        z_min=None,
        z_max=None,
        regression_params={},
        custom_model=None,
    ):
        """Initialize FlexCodeModel object

        :param model: A FlexCodeRegression object
        :param max_basis: int, the maximal number of basis functions
        :param basis_system: string, the basis system: options are "cosine"
        :param z_min: float, the minimum z value; if None will default
        to the minimum of the training values
        :param z_max: float, the maximum z value; if None will default
        to the maximum of the training values
        :param regression_params: A dictionary of tuning parameters
        for the regression model
        :param custom_model: a scikit-learn-type model, i.e. with fit and
        predict method.
        """
        self.max_basis = max_basis
        self.best_basis = range(max_basis)
        self.basis_system = basis_system
        self.model = model(max_basis, regression_params, custom_model)

        self.z_min = z_min
        self.z_max = z_max

        self.bump_threshold = None
        self.sharpen_alpha = None

    def fit(self, x_train, z_train, weight=None):
        """Fits basis function regression models.

        :param x_train: a numpy matrix of training covariates.
        :param z_train: a numpy array of z values.
        :param weight: (optional) a numpy array of weights.
        :returns: None.
        :rtype:

        """
        if len(x_train.shape) == 1:
            x_train = x_train.reshape(-1, 1)
        if len(z_train.shape) == 1:
            z_train = z_train.reshape(-1, 1)

        if self.z_min is None:
            self.z_min = min(z_train)
        if self.z_max is None:
            self.z_max = max(z_train)

        z_basis = evaluate_basis(
            box_transform(z_train, self.z_min, self.z_max), self.max_basis, self.basis_system
        )

        self.model.fit(x_train, z_basis, weight)

    def tune(self, x_validation, z_validation, bump_threshold_grid=None, sharpen_grid=None, n_grid=1000):
        """Set tuning parameters to minimize CDE loss

        Sets best_basis, bump_delta, and sharpen_alpha values attributes

        :param x_validation: a numpy matrix of covariates
        :param z_validation: a numpy array of z values
        :param bump_threshold_grid: an array of candidate bump threshold values
        :param sharpen_grid: an array of candidate sharpen parameter values
        :param n_grid: integer, the number of grid points to evaluate
        :returns: None
        :rtype:

        """
        if len(x_validation.shape) == 1:
            x_validation = x_validation.reshape(-1, 1)
        if len(z_validation.shape) == 1:
            z_validation = z_validation.reshape(-1, 1)

        z_validation = box_transform(z_validation, self.z_min, self.z_max)
        z_basis = evaluate_basis(z_validation, self.max_basis, self.basis_system)

        coefs = self.model.predict(x_validation)

        term1 = np.mean(coefs**2, 0)
        term2 = np.mean(coefs * z_basis, 0)
        # losses = np.cumsum(term1 - 2 * term2)
        self.best_basis = np.where(term1 - 2 * term2 < 0.0)[0]

        if bump_threshold_grid is not None or sharpen_grid is not None:
            coefs = coefs[:, self.best_basis]
            z_grid = make_grid(n_grid, self.z_min, self.z_max)
            z_basis = evaluate_basis(
                box_transform(z_grid, self.z_min, self.z_max), max(self.best_basis) + 1, self.basis_system
            )
            z_basis = z_basis[:, self.best_basis]
            cdes = np.matmul(coefs, z_basis.T)
            normalize(cdes)

            if bump_threshold_grid is not None:
                self.bump_threshold = choose_bump_threshold(cdes, z_grid, z_validation, bump_threshold_grid)

                remove_bumps(cdes, self.bump_threshold)
                normalize(cdes)

            if sharpen_grid is not None:
                self.sharpen_alpha = choose_sharpen(cdes, z_grid, z_validation, sharpen_grid)

    def predict_coefs(self, x_new):
        if len(x_new.shape) == 1:
            x_new = x_new.reshape(-1, 1)

        coefs = self.model.predict(x_new)[:, self.best_basis]
        return BasisCoefs(
            coefs, self.basis_system, self.z_min, self.z_max, self.bump_threshold, self.sharpen_alpha
        )

    def predict(self, x_new, n_grid):
        """Predict conditional density estimates on new data

        n        :param x_new: A numpy matrix of covariates at which to predict
                :param n_grid: int, the number of grid points at which to
                predict the conditional density
                :returns: A numpy matrix where each row is a conditional
                density estimate at the grid points
                :rtype: numpy matrix

        """
        if len(x_new.shape) == 1:
            x_new = x_new.reshape(-1, 1)

        z_grid = make_grid(n_grid, 0.0, 1.0)
        z_basis = evaluate_basis(z_grid, max(self.best_basis) + 1, self.basis_system)
        z_basis = z_basis[:, self.best_basis]
        coefs = self.model.predict(x_new)[:, self.best_basis]
        cdes = np.matmul(coefs, z_basis.T)

        # Post-process
        normalize(cdes)
        if self.bump_threshold is not None:
            remove_bumps(cdes, self.bump_threshold)
        if self.sharpen_alpha is not None:
            sharpen(cdes, self.sharpen_alpha)
        cdes /= self.z_max - self.z_min
        return cdes, make_grid(n_grid, self.z_min, self.z_max)

    def estimate_error(self, x_test, z_test, n_grid=1000):
        """Estimates CDE loss on test data

        :param x_test: A numpy matrix of covariates
        :param z_test: A numpy matrix of z values
        :param n_grid: Number of grid points at which to predict the
        conditional density
        :returns: an estimate of the CDE loss
        :rtype: float

        """
        if len(x_test.shape) == 1:
            x_test = x_test.reshape(-1, 1)
        if len(z_test.shape) == 1:
            z_test = z_test.reshape(-1, 1)

        cde_estimate, z_grid = self.predict(x_test, n_grid)
        return cde_loss(cde_estimate, z_grid, z_test)
