import numpy as np

def evaluate_basis(z, n_basis, basis_system):
    """Evaluates a system of basis functions

    :param z: array of z values in [0, 1]
    :param n_basis: int, the number of basis functions to calculate
    :param basis_system: string, the basis system: options include "cosine", "Fourier", "Haar", and "Daubechies"
    :returns: the matrix of basis functions evaluated at z
    :rtype: numpy matrix

    """
    if basis_system == "cosine":
        return cosine_basis(z, n_basis)
    elif basis_system == "Fourier":
        return fourier_basis(z, n_basis)
    elif basis_system == "Haar":
        return haar_basis(z, n_basis)
    elif basis_system == "Daubechies":
        return daubechies_basis(z, n_basis)

    raise ValueError("basis_system: {} not recognized".format(basis_system))

def cosine_basis(z, n_basis):
    """Evaluates cosine basis

    :param z: array of z values in [0, 1]
    :param n_basis: int, the number of basis functions to calculate
    :returns: the matrix of cosine basis functions evaluated at z
    :rtype: numpy matrix

    """
    n_obs = z.shape[0]
    basis = np.empty((n_obs, n_basis))

    basis[:, 0] = 1.0
    for ii in range(1, n_basis):
        basis[:, ii] = np.sqrt(2) * np.cos(np.pi * ii * z)
    return basis

def fourier_basis(z, n_basis):
    """Evaluates Fourier basis

    :param z: array of z values in [0, 1]
    :param n_basis: int, the number of basis functions to calculate
    :returns: the matrix of Fourier basis functions evaluated at z
    :rtype: numpy matrix

    """
    raise NotImplementedError

def daubechies_basis(z, n_basis):
    """Evaluates Daubechies basis

    :param z: array of z values in [0, 1]
    :param n_basis: int, the number of basis functions to calculate
    :returns: the matrix of Daubechies basis functions evaluated at z
    :rtype: numpy matrix

    """
    raise NotImplementedError


def haar_basis(z, n_basis):
    """Evaluates Haar basis

    :param z: array of z values in [0, 1]
    :param n_basis: int, the number of basis functions to calculate
    :returns: the matrix of Haar basis functions evaluated at z
    :rtype: numpy matrix

    """
    raise NotImplementedError
