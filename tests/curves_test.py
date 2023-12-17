from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from femto.curves import abv
from femto.curves import arc
from femto.curves import arctan
from femto.curves import circ
from femto.curves import erf
from femto.curves import euler
from femto.curves import euler_S2
from femto.curves import euler_S4
from femto.curves import rad
from femto.curves import sin
from femto.curves import spline
from femto.curves import spline_bridge
from femto.curves import tanh


@pytest.fixture
def values() -> list[list[float]]:
    dx = [1, 2, 3, 4, 5, 10]
    dy = [-0.1, 0.4, 0.6, 1, -0.08, 0.001]
    dz = [0, 0, 1, 3, 0.54, 0.004]
    return dx, dy, dz


@pytest.mark.parametrize('fx', [sin, spline, arctan, rad, abv, euler_S2, euler_S4])
def test_curves(fx, values):
    disp_x, disp_y, disp_z = values

    for dx, dy, dz in zip(disp_x, disp_y, disp_z):
        x, y, z = fx(dx=dx, dy=dy, dz=dz, radius=10, num_points=100)

        assert pytest.approx(x[-1]) == dx
        assert pytest.approx(y[-1]) == dy
        assert pytest.approx(z[-1]) == dz
