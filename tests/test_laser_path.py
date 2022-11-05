import numpy as np
import pytest

from femto.LaserPath import LaserPath


@pytest.fixture
def param() -> dict:
    p = {
            'scan'        : 6,
            'speed'       : 20.0,
            'y_init'      : 1.5,
            'z_init'      : 0.035,
            'lsafe'       : 4.3,
            'speed_closed': 75,
            'speed_pos'   : 0.1,
            'samplesize'  : (100, 15)
    }
    return p


@pytest.fixture
def empty_path(param) -> LaserPath:
    return LaserPath(**param)


@pytest.fixture
def laser_path(param) -> LaserPath:
    # create LaserPath instance
    lp = LaserPath(**param)

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


def test_fabrication_time_empty(empty_path) -> None:
    assert pytest.approx(empty_path.fabrication_time == 0.0)


def test_x(laser_path) -> None:
    np.testing.assert_almost_equal(laser_path.x, np.array([0.0, 1.0, 1.0, 1.0, 2.0]))


def test_x_empty(empty_path) -> None:
    np.testing.assert_array_equal(empty_path.x, np.array([]))


def test_lastx(laser_path) -> None:
    assert laser_path.lastx == 2.0


def test_lastx_empty(empty_path) -> None:
    assert empty_path.lastx is None


def test_y(laser_path) -> None:
    np.testing.assert_almost_equal(laser_path.y, np.array([0.0, 0.0, 1.0, 2.0, 3.0]))


def test_y_empty(empty_path) -> None:
    np.testing.assert_array_equal(empty_path.y, np.array([]))


def test_lasty(laser_path) -> None:
    assert laser_path.lasty == 3.0


def test_lasty_empty(empty_path) -> None:
    assert empty_path.lasty is None


def test_z(laser_path) -> None:
    np.testing.assert_almost_equal(laser_path.z, np.array([0.0, 0.0, 1.0, 0.0, 3.0]))


def test_z_empty(empty_path) -> None:
    np.testing.assert_array_equal(empty_path.x, np.array([]))


def test_lastz(laser_path) -> None:
    assert laser_path.lastz == 3.0


def test_lastz_empty(empty_path) -> None:
    assert empty_path.lastz is None


def test_last_point(laser_path) -> None:
    np.testing.assert_almost_equal(laser_path.lastpt, np.array([2.0, 3.0, 3.0]))


def test_last_point_empty(empty_path) -> None:
    np.testing.assert_array_equal(empty_path.lastpt, np.array([]))
