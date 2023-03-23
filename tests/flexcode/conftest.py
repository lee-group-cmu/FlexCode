import numpy as np
import pytest

# defining some constants used throughout the test suite
BUMP_THRESHOLD_GRID = np.linspace(0, 0.2, 3)
SHARPEN_GRID = np.linspace(0.5, 1.5, 3)


def generate_data(n_draws):
    """Generate data p(z | x) = N(x, 1)

    Parameters
    ----------
    n_draws : int
        number of samples to generate

    Returns
    -------
    x : List[float]
        Samples drawn from a 0, 1 normal distribution
    z : List[float]
        Samples drawn from a `x`, 1 normal distribution. Where `x` is random variate
        created earlier.
    """
    x = np.random.normal(0, 1, n_draws)
    z = np.random.normal(x, 1, n_draws)
    return x, z
