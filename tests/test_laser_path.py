import pytest

from femto.LaserPath import LaserPath
import numpy as np


@pytest.fixture
def laser_path() -> LaserPath:
    p = {
        'scan': 6,
        'speed': 20.0,
        'y_init': 1.5,
        'z_init': 0.035,
        'lsafe': 4.3,
        'speed_closed': 75,
        'speed_pos': 0.1,
        'samplesize': (100, 15)
    }
    # create LaserPath instance
    lp = LaserPath(**p)

    # add path
    x = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2])
    y = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3])
    z = np.array([0, 0, 1, 1, 1, 0, 3, 3, 3])
    f = np.array([1, 2, 1, 1, 1, 3, 4, 4, 4])
    s = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    lp.add_path(x, y, z, f, s)
    return lp


def test_default_values() -> None:
    lp = LaserPath()
    assert lp.scan == int(1)
    assert lp.speed == float(1.0)
    assert lp.x_init == float(-2.0)
    assert lp.y_init == float(0.0)
    assert lp.z_init is None
    assert lp.lsafe == float(2.0)
    assert lp.speed_closed == float(5.0)
    assert lp.speed_pos == float(0.5)
    assert lp.cmd_rate_max == int(1200)
    assert lp.acc_max == int(500)
    assert lp.samplesize == (None, None)


def test_laserpath_values(laser_path) -> None:
    assert laser_path.scan == int(6)
    assert laser_path.speed == float(20.0)
    assert laser_path.x_init == float(-2.0)
    assert laser_path.y_init == float(1.5)
    assert laser_path.z_init == float(0.035)
    assert laser_path.lsafe == float(4.3)
    assert laser_path.speed_closed == float(75)
    assert laser_path.speed_pos == float(0.1)
    assert laser_path.cmd_rate_max == int(1200)
    assert laser_path.acc_max == int(500)
    assert laser_path.samplesize == (100, 15)


def test_init_point(laser_path) -> None:
    assert laser_path.init_point == [-2.0, 1.5, 0.035]


def test_lvelo(laser_path) -> None:
    assert pytest.approx(laser_path.lvelo, 1.2)


def test_dl(laser_path) -> None:
    assert pytest.approx(laser_path.dl, (1 / 60))


def test_x_end(laser_path) -> None:
    assert pytest.approx(laser_path.x_end, 104.3)


def test_add_path(laser_path) -> None:
    np.testing.assert_almost_equal(laser_path._x, np.array([0, 1, 1, 1, 1, 1, 2, 2, 2]))
    np.testing.assert_almost_equal(laser_path._y, np.array([0, 0, 1, 1, 1, 2, 3, 3, 3]))
    np.testing.assert_almost_equal(laser_path._z, np.array([0, 0, 1, 1, 1, 0, 3, 3, 3]))
    np.testing.assert_almost_equal(laser_path._f, np.array([1, 2, 1, 1, 1, 3, 4, 4, 4]))
    np.testing.assert_almost_equal(laser_path._s, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))


def test_fabrication_time(laser_path) -> None:
    assert pytest.approx(laser_path.fabrication_time == 19.288646)
