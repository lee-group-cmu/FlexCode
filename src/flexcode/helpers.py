import numpy as np

def box_transform(z, z_min, z_max):
    """Projects z from box [z_min, z_max] to [0, 1]

    :param z: an array of z values
    :param z_min: float, the minimum value of the z box
    :param z_max: float, the maximum value of the z box
    :returns: z projected onto [0, 1]
    :rtype: array

    """

    return (z - z_min) / (z_max - z_min)

def make_grid(n_grid, z_min, z_max):
    """Create grid of equally spaced points

    :param n_grid: integer number of grid points
    :param z_min: float, the minimum value of the z box
    :param z_max: float, the maximum value of the z box
    :returns: a grid of n_grid equally spaced points between z_min and z_max
    :rtype: array

    """
    return np.linspace(z_min, z_max, n_grid).reshape((n_grid, 1))
