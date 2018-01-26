import numpy as np
from .loss_functions import cde_loss

def normalize(cde_estimates, z_grid, tol=1e-6, max_iter=200):
    """Normalizes conditional density estimates to be non-negative and
    integrate to one.

    :param cde_estimates: a numpy array or matrix of conditional density estimates.
    :param z_grid: an array, the grid points at the density is estimated.
    :param tol: float, the tolerance to accept for abs(area - 1).
    :param max_iter: int, the maximal number of search iterations.
    :returns: the normalized conditional density estimates.
    :rtype: numpy array or matrix.

    """
    if cde_estimates.ndim == 1:
        _normalize(cde_estimates, z_grid, tol, max_iter)
    else:
        np.apply_along_axis(_normalize, 1, cde_estimates,
                            z_grid=z_grid, tol=tol, max_iter=max_iter)

def _normalize(density, z_grid, tol=1e-6, max_iter=500):
    """Normalizes a density estimate to be non-negative and integrate to
    one.

    :param density: a numpy array of density estimates.
    :param z_grid: an array, the grid points at the density is estimated.
    :param tol: float, the tolerance to accept for abs(area - 1).
    :param max_iter: int, the maximal number of search iterations.
    :returns: the normalized density estimate.
    :rtype: numpy array.

    """
    hi = np.max(density)
    lo = 0.0

    area = np.trapz(np.maximum(density, 0.0), z_grid)
    if area == 0.0:
        # replace with uniform if all negative density
        density[:] = 1 / (max(z_grid) - min(z_grid))
    elif area < 1:
        density /= area
        density[density < 0.0] = 0.0
        return

    for _ in range(max_iter):
        mid = (hi + lo) / 2
        tmp_density = np.maximum(density - mid, 0.0)

        area = np.trapz(tmp_density, z_grid)
        if abs(1.0 - area) <= tol:
            break
        if area < 1.0:
            hi = mid
        else:
            lo = mid

    # update in place
    density -= mid
    density[density < 0.0] = 0.0

def sharpen(cde_estimates, z_grid, alpha):
    """Sharpens conditional density estimates.

    :param cde_estimates: a numpy array or matrix of conditional density estimates.
    :param z_grid: an array, the grid points at the density is estimated.
    :param alpha: float, the exponent to which the estimate is raised.
    :returns: the sharpened conditional density estimate.
    :rtype: numpy array or matrix.

    """
    cde_estimates **= alpha
    normalize(cde_estimates, z_grid)

def choose_sharpen(cde_estimates, z_grid, true_z, alpha_grid):
    """Chooses the sharpen parameter by minimizing cde loss.

    :param cde_estimates: a numpy matrix of conditional density estimates
    :param z_grid: an array, the grid points at the density is estimated.
    :param true_z: an array of the true z values corresponding to the cde_estimates.
    :param alpha_grid: an array of candidate sharpen parameter values.
    :returns: the sharpen parameter value from alpha_grid which minimizes cde loss.
    :rtype: float

    """
    best_alpha = None
    best_loss = np.inf
    for alpha in alpha_grid:
        tmp_estimates = cde_estimates.copy()
        sharpen(tmp_estimates, z_grid, alpha)
        loss = cde_loss(tmp_estimates, z_grid, true_z)
        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
    return best_alpha

def remove_bumps(cde_estimates, z_grid, delta):
    """Removes bumps in conditional density estimates

    :param cde_estimates: a numpy array or matrix of conditional density estimates.
    :param z_grid: an array, the grid points at the density is estimated.
    :param delta: float, the threshold for bump removal
    :returns: the conditional density estimates with bumps removed
    :rtype: numpy array or matrix

    """
    if cde_estimates.ndim == 1:
        _remove_bumps(cde_estimates, z_grid, delta)
    else:
        np.apply_along_axis(_remove_bumps, 1, cde_estimates,
                            z_grid=z_grid, delta = delta)

def _remove_bumps(density, z_grid, delta):
    """Removes bumps in conditional density estimates.

    :param density: a numpy array of conditional density estimate.
    :param z_grid: an array, the grid points at the density is estimated.
    :param delta: float, the threshold for bump removal.
    :returns: the conditional density estimate with bumps removed.
    :rtype: numpy array.

    """
    bin_size = (max(z_grid) - min(z_grid)) / len(z_grid)
    area = 0.0
    left_idx = 0
    for right_idx, val in enumerate(density):
        if val <= 0.0:
            if area < delta:
                density[left_idx:(right_idx + 1)] = 0.0
            left_idx = right_idx + 1
            area = 0.0
        else:
            area += val * bin_size
    if area < delta: # final check at end
        density[left_idx:] = 0.0
        _normalize(density, z_grid)

def choose_bump_threshold(cde_estimates, z_grid, true_z, delta_grid):
    """Chooses the bump threshold which minimizes cde loss.

    :param cde_estimates: a numpy array or matrix of conditional density estimates.
    :param z_grid: an array, the grid points at which the density is estimated.
    :param true_z: the true z values corresponding to the conditional
    denstity estimates.
    :param delta_grid: an array of candidate bump threshold values
    :returns: the bump threshold value from delta_grid which minimizes CDE loss
    :rtype: float

    """
    best_delta = None
    best_loss = np.inf
    for delta in delta_grid:
        tmp_estimates = cde_estimates.copy()
        remove_bumps(tmp_estimates, z_grid, delta)
        normalize(tmp_estimates, z_grid)
        loss = cde_loss(tmp_estimates, z_grid, true_z)
        if loss < best_loss:
            best_loss = loss
            best_delta = delta
    return best_delta
