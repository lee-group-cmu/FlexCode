import numpy as np

from .helpers import box_transform
from .basis_functions import evaluate_basis
from .post_processing import *
from .loss_functions import cde_loss


class FlexCodeModel(object):
    def __init__(self, model, max_basis, basis_system="cosine",
                 z_min=None, z_max=None, regression_params={}):
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

        """
        self.max_basis = max_basis
        self.best_basis = range(max_basis)
        self.basis_system = basis_system
        self.model = model(max_basis, regression_params)

        self.z_min = z_min
        self.z_max = z_max

        self.bump_threshold = None
        self.sharpen_alpha = None

    def fit(self, x_train, z_train):
        """Fits basis function regression models.

        :param x_train: a numpy matrix of training covariates.
        :param z_train: a numpy array of z values.
        :returns: None.
        :rtype:

        """
        if self.z_min is None:
            self.z_min = min(z_train)
        if self.z_max is None:
            self.z_max = max(z_train)

        z_basis = evaluate_basis(box_transform(z_train, self.z_min, self.z_max),
                                 self.max_basis, self.basis_system)

        self.model.fit(x_train, z_basis)

    def tune(self, x_validation, z_validation, bump_threshold_grid =
             None, sharpen_grid = None, n_grid=1000):
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
        z_basis = evaluate_basis(box_transform(z_validation, self.z_min, self.z_max),
                                 self.max_basis, self.basis_system)

        coefs = self.model.predict(x_validation)

        term1 = np.mean(coefs ** 2, 0)
        term2 = np.mean(coefs * z_basis, 0)
        # losses = np.cumsum(term1 - 2 * term2)
        self.best_basis = np.where(term1 - 2 * term2 < 0.0)[0]

        if bump_threshold_grid is not None or sharpen_grid is not None:
            coefs = coefs[:, self.best_basis]
            z_grid = np.linspace(self.z_min, self.z_max, n_grid)

            z_basis = evaluate_basis(np.linspace(0, 1, n_grid),
                                     max(self.best_basis) + 1, self.basis_system)
            z_basis = z_basis[:, self.best_basis]
            cdes = np.matmul(coefs, z_basis.T)
            cdes /= self.z_max - self.z_min

            normalize(cdes, z_grid)

            if bump_threshold_grid is not None:
                self.bump_threshold = choose_bump_threshold(cdes, z_grid,
                                                            z_validation,
                                                            bump_threshold_grid)

                remove_bumps(cdes, z_grid, self.bump_threshold)
                normalize(cdes, z_grid)

            if sharpen_grid is not None:
                self.sharpen_alpha = choose_sharpen(cdes, z_grid, z_validation,
                                                    sharpen_grid)

    def predict(self, x_new, n_grid):
        """Predict conditional density estimates on new data

        :param x_new: A numpy matrix of covariates at which to predict
        :param n_grid: int, the number of grid points at which to
        predict the conditional density
        :returns: A numpy matrix where each row is a conditional
        density estimate at the grid points
        :rtype: numpy matrix

        """
        z_grid = np.linspace(self.z_min, self.z_max, n_grid)
        z_basis = evaluate_basis(np.linspace(0, 1, n_grid),
                                 max(self.best_basis) + 1, self.basis_system)
        z_basis = z_basis[:, self.best_basis]
        coefs = self.model.predict(x_new)[:, self.best_basis]
        cdes = np.matmul(coefs, z_basis.T)
        cdes /= self.z_max - self.z_min

        # Post-process
        normalize(cdes, z_grid)
        if self.bump_threshold is not None:
            remove_bumps(cdes, z_grid, self.bump_threshold)
        if self.sharpen_alpha is not None:
            sharpen(cdes, z_grid, self.sharpen_alpha)
        return cdes, z_grid

    def estimate_error(self, x_test, z_test, n_grid = 1000):
        """Estimates CDE loss on test data

        :param x_test: A numpy matrix of covariates
        :param z_test: A numpy matrix of z values
        :param n_grid: Number of grid points at which to predict the
        conditional density
        :returns: an estimate of the CDE loss
        :rtype: float

        """
        cde_estimate, z_grid = self.predict(x_test, n_grid)
        return cde_loss(cde_estimate, z_grid, z_test)
