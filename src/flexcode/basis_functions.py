import numpy as np
import pywt

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
    else:
        return wavelet_basis(z, n_basis, basis_system)


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

    n_obs = z.shape[0]
    basis = np.zeros((n_obs, n_basis))

    basis[:, 0] = 1.0
    for ii in range(1, (n_basis + 1) // 2):
        basis[:, 2 * ii - 1] = np.sqrt(2) * np.sin(2 * np.pi * ii * z)
        basis[:, 2 * ii] = np.sqrt(2) * np.cos(2 * np.pi * ii * z)
        if n_basis % 2 == 0:
            basis[:, -1] = np.sqrt(2) * np.sin(np.pi * n_basis * z)
    return basis

def wavelet_basis(z, n_basis, family='db4'):
    """Evaluates Daubechies basis

    :param z: array of z values in [0, 1]
    :param n_basis: int, the number of basis functions to calculate
    :returns: the matrix of Daubechies basis functions evaluated at z
    :rtype: numpy matrix

    """
    n_aux = 15
    rez = pywt.DiscreteContinuousWavelet(family).wavefun(n_aux)
    if len(rez) == 2:
        wavelet, x = rez
    else:
        _, wavelet, x = rez
    wavelet *= np.sqrt(max(x) - min(x))
    x = (x - min(x)) / (max(x) - min(x))
    def wave_fun(t):
        if t < 0 or t > 1:
            return 0.0
        return wavelet[np.argmin(abs(t - x))]

    n_obs = z.shape[0]
    basis = np.empty((n_obs, n_basis))
    basis[:, 0] = 1.0

    ii = 1
    loc = 0
    level = 0
    for ii in range(1, n_basis):
        basis[:, ii] = [2 ** (level / 2) * wave_fun(a * 2 ** level - loc) for a in z]
        loc += 1
        if loc == 2 ** level:
            loc = 0
            level += 1
    return basis

# def haar_basis(z, n_basis):
#     """Evaluates Haar basis

#     :param z: array of z values in [0, 1]
#     :param n_basis: int, the number of basis functions to calculate
#     :returns: the matrix of Haar basis functions evaluated at z
#     :rtype: numpy matrix

#     """
#     n_aux = 14
#     _, wavelet, x = pywt.Wavelet("haar").wavefun(n_aux)
#     x = (x - min(x)) / (max(x) - min(x))
#     def wave_fun(t):
#         if t < 0 or t > 1:
#             return 0.0
#         return wavelet[np.argmin(abs(t - x))]

#     n_obs = z.shape[0]
#     basis = np.empty((n_obs, n_basis))
#     basis[:, 0] = 1.0

#     ii = 1
#     loc = 0
#     level = 0
#     for ii in range(1, n_basis):
#         basis[:, ii] = [2 ** (level / 2) * wave_fun(2 ** level * a - loc) for a in z]

#         loc += 1
#         if loc == 2 ** level:
#             loc = 0
#             level += 1
#     return basis
