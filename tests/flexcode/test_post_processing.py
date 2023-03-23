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
    normalize(target_density)

    remove_bumps(density, delta)

    np.testing.assert_array_equal(density, target_density)


def test_normalize():
    n_grid = 1000

    for _ in range(10):
        density = np.random.gamma(1, 1, size=n_grid)
        normalize(density)
        area = np.mean(density)
        assert all(density >= 0.0)
        assert area == pytest.approx(1.0)
