import numpy as np
import pytest
from context import flexcode
from flexcode.post_processing import *

def test_remove_bumps():
    density = np.ones(100)
    density[4] = 0.0
    density[96] = 0.0

    z_grid = np.linspace(0, 1, 100)
    delta = 0.1

    target_density = density.copy()
    target_density[:4] = 0.0
    target_density[96:] = 0.0
    target_density = normalize(target_density, z_grid)

    np.testing.assert_array_equal(remove_bumps(density, z_grid, delta), target_density)

def test_normalize():
    n_grid = 1000

    min_z = -1
    max_z = 1
    z_grid = np.linspace(min_z, max_z, n_grid)

    for _ in range(10):
        density = np.random.gamma(1, 1, size=n_grid)
        normalized = normalize(density, z_grid)
        area = np.trapz(normalized, z_grid)
        assert all(normalized >= 0.0)
        assert area == pytest.approx(1.0)
