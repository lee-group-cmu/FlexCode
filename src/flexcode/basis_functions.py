"""Functions for evaluation of orthogonal basis functions."""

import numpy as np
import pywt

from .helpers import box_transform, make_grid
from .post_processing import *


def evaluate_basis(responses, n_basis, basis_system):
    """Evaluates a system of basis functions.

    Arguments
    ----------
    responses : array
        An array of responses in [0, 1].
    n_basis : integer
        The number of basis functions to calculate.
    basis_system : {'cosine', 'Fourier', 'db4'}
        String denoting the system of orthogonal basis functions.

    Returns
    -------
    numpy matrix
       A matrix of basis functions evaluations. Each row corresponds
       to a value of `responses`, each column corresponds to a basis function.

    Raises
    ------
    ValueError
        If the basis system isn't recognized.

    """
    systems = {"cosine": cosine_basis, "Fourier": fourier_basis, "db4": wavelet_basis}
    try:
        basis_fn = systems[basis_system]
    except KeyError:
        raise ValueError("Basis system {} not recognized".format(basis_system))

    n_dim = responses.shape[1]
    if n_dim == 1:
        return basis_fn(responses, n_basis)
    else:
        if len(n_basis) == 1:
            n_basis = [n_basis] * n_dim
        return tensor_basis(responses, n_basis, basis_fn)


def tensor_basis(responses, n_basis, basis_fn):
    """Evaluates tensor basis.

    Combines single-dimensional basis functions \phi_{d}(z) to form
    orthogonal tensor basis $\phi(z_{1}, \dots, z_{D}) = \prod_{d}
    \phi_{d}(z_{d})$.

    Arguments
    ---------
    responses : numpy matrix
        A matrix of responses in [0, 1]^(n_dim). Each column
        corresponds to a variable, each row corresponds to an
        observation.
    n_basis : list of integers
        The number of basis function for each dimension. Should have
        the same length as the number of columns of `responses`.
    basis_fn : function
        The function which evaluates the one-dimensional basis
        functions.

    Returns
    -------
    numpy matrix
        Returns a matrix where each column is a basis function and
        each row is an observation.

    """
    n_obs, n_dims = responses.shape

    basis = np.ones((n_obs, np.prod(n_basis)))
    period = 1
    for dim in range(n_dims):
        sub_basis = basis_fn(responses[:, dim], n_basis[dim])
        col = 0
        for _ in range(np.prod(n_basis) // (n_basis[dim] * period)):
            for sub_col in range(n_basis[dim]):
                for _ in range(period):
                    basis[:, col] *= sub_basis[:, sub_col]
                    col += 1
        period *= n_basis[dim]
    return basis


def cosine_basis(responses, n_basis):
    """Evaluates cosine basis.

    Arguments
    ----------
    responses : array
        An array of responses in [0, 1].
    n_basis : integer
        The number of basis functions to evaluate.

    Returns
    -------
    numpy matrix
        A matrix of cosine basis functions evaluated at `responses`. Each row
        corresponds to a value of `responses`, each column corresponds to a
        basis function.

    """
    n_obs = responses.shape[0]
    basis = np.empty((n_obs, n_basis))

    responses = responses.flatten()

    basis[:, 0] = 1.0
    for col in range(1, n_basis):
        basis[:, col] = np.sqrt(2) * np.cos(np.pi * col * responses)
    return basis


def fourier_basis(responses, n_basis):
    """Evaluates Fourier basis.

    Arguments
    ----------
    responses : array
        An array of responses in [0, 1].
    n_basis : integer
        The number of basis functions to evaluate.

    Returns
    -------
    numpy matrix
        A matrix of Fourier basis functions evaluated at `responses`. Each row
        corresponds to a value of `responses`, each column corresponds to a
        basis function.

    """
    n_obs = responses.shape[0]
    basis = np.zeros((n_obs, n_basis))

    responses = responses.flatten()

    basis[:, 0] = 1.0
    for col in range(1, (n_basis + 1) // 2):
        basis[:, 2 * col - 1] = np.sqrt(2) * np.sin(2 * np.pi * col * responses)
        basis[:, 2 * col] = np.sqrt(2) * np.cos(2 * np.pi * col * responses)
        if n_basis % 2 == 0:
            basis[:, -1] = np.sqrt(2) * np.sin(np.pi * n_basis * responses)
    return basis


def wavelet_basis(responses, n_basis, family="db4"):
    """Evaluates Daubechies basis.

    Arguments
    ----------
    responses : array
        An array of responses in [0, 1].
    n_basis : integer
        The number of basis functions to evaluate.
    family : string
        The wavelet family to evaluate.

    Returns
    -------
    numpy matrix
        A matrix of Fourier basis functions evaluated at `responses`. Each row
        corresponds to a value of `responses`, each column corresponds to a
        basis function.

    """
    responses = responses.flatten()

    n_aux = 15
    rez = pywt.DiscreteContinuousWavelet(family).wavefun(n_aux)
    if len(rez) == 2:
        wavelet, x_grid = rez
    else:
        _, wavelet, x_grid = rez
    wavelet *= np.sqrt(max(x_grid) - min(x_grid))
    x_grid = (x_grid - min(x_grid)) / (max(x_grid) - min(x_grid))

    def _wave_fun(val):
        if val < 0 or val > 1:
            return 0.0
        return wavelet[np.argmin(abs(val - x_grid))]

    n_obs = responses.shape[0]
    basis = np.empty((n_obs, n_basis))
    basis[:, 0] = 1.0

    loc = 0
    level = 0
    for col in range(1, n_basis):
        basis[:, col] = [2 ** (level / 2) * _wave_fun(a * 2**level - loc) for a in responses]
        loc += 1
        if loc == 2**level:
            loc = 0
            level += 1
    return basis


class BasisCoefs(object):
    def __init__(self, coefs, basis_system, z_min, z_max, bump_threshold=None, sharpen_alpha=None):
        self.coefs = coefs
        self.basis_system = basis_system
        self.z_min = z_min
        self.z_max = z_max
        self.bump_threshold = bump_threshold
        self.sharpen_alpha = sharpen_alpha

    def evaluate(self, z_grid):
        basis = evaluate_basis(
            box_transform(z_grid, self.z_min, self.z_max), self.coefs.shape[1], self.basis_system
        )
        cdes = np.matmul(self.coefs, basis.T)

        normalize(cdes)
        if self.bump_threshold is not None:
            remove_bumps(cdes, self.bump_threshold)
        if self.sharpen_alpha is not None:
            sharpen(cdes, self.sharpen_alpha)
        cdes /= self.z_max - self.z_min
        return cdes
