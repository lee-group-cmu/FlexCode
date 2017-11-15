import numpy as np

def cde_loss(cde_estimates, z_grid, true_z):
    """Calculates conditional density estimation loss on holdout data

    @param cde_estimates: a numpy array where each row is a density
    estimate on z_grid
    @param z_grid: a numpy array of the grid points at which cde_estimates is evaluated
    @param true_z: a numpy array of the true z values corresponding to the rows of cde_estimates

    @returns The CDE loss (up to a constant) for the CDE estimator on
    the holdout data
    """

    n_obs, n_grid = cde_estimates.shape

    term1 = np.mean(np.trapz(cde_estimates ** 2, z_grid))

    nns = [np.argmin(np.abs(z_grid - true_z[ii])) for ii in range(n_obs)]
    term2 = np.mean(cde_estimates[range(n_obs), nns])
    return term1 - 2 * term2
