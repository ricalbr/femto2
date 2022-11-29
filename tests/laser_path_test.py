from __future__ import annotations

from pathlib import Path

import dill
import numpy as np
import pytest
from femto.LaserPath import LaserPath


@pytest.fixture
def param() -> dict:
    p = {
        'scan': 6,
        'speed': 20.0,
        'y_init': 1.5,
        'z_init': 0.035,
        'lsafe': 4.3,
        'speed_closed': 75,
        'speed_pos': 0.1,
        'samplesize': (100, 15),
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


def test_from_dict(param) -> None:
    lp = LaserPath.from_dict(param)

    assert lp.scan == int(6)
    assert lp.speed == float(20.0)
    assert lp.x_init == float(-2.0)
    assert lp.y_init == float(1.5)
    assert lp.z_init == float(0.035)
    assert lp.lsafe == float(4.3)
    assert lp.speed_closed == float(75)
    assert lp.speed_pos == float(0.1)
    assert lp.cmd_rate_max == int(1200)
    assert lp.acc_max == int(500)
    assert lp.samplesize == (100, 15)


def test_repr(param) -> None:
    r = LaserPath(**param).__repr__()
    print()
    print(r)
    cname, _ = r.split('@')
    assert cname == 'LaserPath'


def test_init_point(laser_path) -> None:
    assert laser_path.init_point == [-2.0, 1.5, 0.035]


def test_lvelo(laser_path) -> None:
    assert pytest.approx(laser_path.lvelo) == 1.2


def test_dl(laser_path) -> None:
    assert pytest.approx(laser_path.dl) == (1 / 60)


def test_x_end(laser_path) -> None:
    assert pytest.approx(laser_path.x_end) == 104.3


def test_x_end_none(laser_path) -> None:
    laser_path.samplesize = (None, None)
    assert laser_path.x_end is None


def test_start(param) -> None:
    lp = LaserPath(**param)
    lp.start()
    np.testing.assert_almost_equal(lp._x, np.array([-2.0, -2.0]))
    np.testing.assert_almost_equal(lp._y, np.array([1.5, 1.5]))
    np.testing.assert_almost_equal(lp._z, np.array([0.035, 0.035]))
    np.testing.assert_almost_equal(lp._f, np.array([0.1, 0.1]))
    np.testing.assert_almost_equal(lp._s, np.array([0.0, 1.0]))


def test_start_values(param) -> None:
    init_p = [0.0, 1.0, -0.1]
    speed_p = 1.25
    lp = LaserPath(**param)

    lp.start(init_pos=init_p, speed_pos=speed_p)
    np.testing.assert_almost_equal(lp._x, np.array([0.0, 0.0]))
    np.testing.assert_almost_equal(lp._y, np.array([1.0, 1.0]))
    np.testing.assert_almost_equal(lp._z, np.array([-0.1, -0.1]))
    np.testing.assert_almost_equal(lp._f, np.array([1.25, 1.25]))
    np.testing.assert_almost_equal(lp._s, np.array([0.0, 1.0]))


def test_start_value_error(param) -> None:
    init_p = [0.0, 1.0, -0.1, 0.3]
    lp = LaserPath(**param)
    with pytest.raises(ValueError):
        lp.start(init_p)

    init_p = [0.0]
    lp = LaserPath(**param)
    with pytest.raises(ValueError):
        lp.start(init_p)

    init_p = []
    lp = LaserPath(**param)
    with pytest.raises(ValueError):
        lp.start(init_p)


def test_end(param) -> None:
    lp = LaserPath(**param)
    lp.start([0.0, 0.0, 0.0])
    lp.end()

    np.testing.assert_almost_equal(lp._x, np.array([0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(lp._y, np.array([0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(lp._z, np.array([0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(lp._f, np.array([0.1, 0.1, 0.1, 75.0]))
    np.testing.assert_almost_equal(lp._s, np.array([0.0, 1.0, 0.0, 0.0]))


def test_empty_end(param) -> None:
    lp = LaserPath(**param)
    with pytest.raises(IndexError):
        lp.end()


def test_linear_value_error(param) -> None:
    lp = LaserPath(**param)
    with pytest.raises(ValueError):
        lp.start().linear([1, 1, 1], mode='rand').end()

    lp = LaserPath(**param)
    lp.speed = None
    with pytest.raises(ValueError):
        lp.start().linear([1, 1, 1], mode='ABS', speed=None).end()


def test_linear_invalid_increment(param) -> None:
    lp = LaserPath(**param)
    with pytest.raises(ValueError):
        lp.start().linear([1, 2]).end()

    with pytest.raises(ValueError):
        lp.start().linear([1, 2, 3, 4]).end()


def test_linear_default_speed(param) -> None:
    lp = LaserPath(**param)
    lp.start().linear([1, 2, 3])
    assert lp._f[-1] == param['speed']


def test_linear_abs(param) -> None:
    lp = LaserPath(**param)
    init_p = [1, 1, 1]
    increm = [3, 4, 5]
    lp.start(init_p).linear(increm, mode='abs').end()

    np.testing.assert_almost_equal(lp._x, np.array([1.0, 1.0, 3.0, 3.0, 1.0]))
    np.testing.assert_almost_equal(lp._y, np.array([1.0, 1.0, 4.0, 4.0, 1.0]))
    np.testing.assert_almost_equal(lp._z, np.array([1.0, 1.0, 5.0, 5.0, 1.0]))
    np.testing.assert_almost_equal(lp._f, np.array([0.1, 0.1, 20.0, 20.0, 75.0]))
    np.testing.assert_almost_equal(lp._s, np.array([0.0, 1.0, 1.0, 0.0, 0.0]))


def test_linear_inc(param) -> None:
    lp = LaserPath(**param)
    init_p = [1, 1, 1]
    increm = [3, 4, 5]
    lp.start(init_p).linear(increm, mode='inc').end()

    np.testing.assert_almost_equal(lp._x, np.array([1.0, 1.0, 4.0, 4.0, 1.0]))
    np.testing.assert_almost_equal(lp._y, np.array([1.0, 1.0, 5.0, 5.0, 1.0]))
    np.testing.assert_almost_equal(lp._z, np.array([1.0, 1.0, 6.0, 6.0, 1.0]))
    np.testing.assert_almost_equal(lp._f, np.array([0.1, 0.1, 20.0, 20.0, 75.0]))
    np.testing.assert_almost_equal(lp._s, np.array([0.0, 1.0, 1.0, 0.0, 0.0]))


def test_linear_none(param) -> None:
    lp = LaserPath(**param)
    init_p = [1, 1, 1]
    increm = [4, None, None]
    lp.start(init_p).linear(increm, mode='abs').end()

    np.testing.assert_almost_equal(lp._x, np.array([1.0, 1.0, 4.0, 4.0, 1.0]))
    np.testing.assert_almost_equal(lp._y, np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(lp._z, np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(lp._f, np.array([0.1, 0.1, 20.0, 20.0, 75.0]))
    np.testing.assert_almost_equal(lp._s, np.array([0.0, 1.0, 1.0, 0.0, 0.0]))

    lp = LaserPath(**param)
    init_p = [1, 1, 1]
    increm = [5, None, None]
    lp.start(init_p).linear(increm, mode='inc').end()

    np.testing.assert_almost_equal(lp._x, np.array([1.0, 1.0, 6.0, 6.0, 1.0]))
    np.testing.assert_almost_equal(lp._y, np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(lp._z, np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(lp._f, np.array([0.1, 0.1, 20.0, 20.0, 75.0]))
    np.testing.assert_almost_equal(lp._s, np.array([0.0, 1.0, 1.0, 0.0, 0.0]))


def test_add_path(laser_path) -> None:
    np.testing.assert_almost_equal(laser_path._x, np.array([0, 1, 1, 1, 1, 1, 2, 2, 2]))
    np.testing.assert_almost_equal(laser_path._y, np.array([0, 0, 1, 1, 1, 2, 3, 3, 3]))
    np.testing.assert_almost_equal(laser_path._z, np.array([0, 0, 1, 1, 1, 0, 3, 3, 3]))
    np.testing.assert_almost_equal(laser_path._f, np.array([1, 2, 1, 1, 1, 3, 4, 4, 4]))
    np.testing.assert_almost_equal(laser_path._s, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))


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


def test_path3d(laser_path) -> None:
    xpath, ypath, zpath = laser_path.path3d

    np.testing.assert_almost_equal(xpath, np.array([0.0, 1.0, 1.0, 1.0, 2.0]))
    np.testing.assert_almost_equal(ypath, np.array([0.0, 0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_almost_equal(zpath, np.array([0.0, 0.0, 1.0, 0.0, 3.0]))


def test_path3d_empty(empty_path) -> None:
    xpath, ypath, zpath = empty_path.path3d

    np.testing.assert_array_equal(xpath, np.array([]))
    np.testing.assert_array_equal(ypath, np.array([]))
    np.testing.assert_array_equal(zpath, np.array([]))


def test_path(laser_path) -> None:
    xpath, ypath = laser_path.path

    np.testing.assert_almost_equal(xpath, np.array([0.0, 1.0, 1.0, 1.0, 2.0]))
    np.testing.assert_almost_equal(ypath, np.array([0.0, 0.0, 1.0, 2.0, 3.0]))


def test_path_empty(empty_path) -> None:
    xpath, ypath = empty_path.path

    np.testing.assert_array_equal(xpath, np.array([]))
    np.testing.assert_array_equal(ypath, np.array([]))


def test_length(laser_path) -> None:
    assert pytest.approx(laser_path.length) == 7.145052


def test_length_empty(empty_path) -> None:
    assert empty_path.length == 0.0


def test_fabrication_time(laser_path) -> None:
    assert pytest.approx(laser_path.fabrication_time) == 42.740724


def test_fabrication_time_empty(empty_path) -> None:
    assert pytest.approx(empty_path.fabrication_time) == 0.0


def test_subs_num_exception(laser_path) -> None:
    with pytest.raises(ValueError):
        laser_path.subs_num(10, 0)


def test_subs_num_custom_inputs(laser_path) -> None:
    assert laser_path.subs_num(103.45, 34), 3652


def test_subs_num_empty_custom_inputs(empty_path) -> None:
    assert empty_path.subs_num(103.45, 34), 3652


def test_subs_num(laser_path) -> None:
    assert laser_path.subs_num(10), 600


def test_subs_num_empty(empty_path) -> None:
    assert empty_path.subs_num(10), 600


def test_subs_num_base_case(laser_path) -> None:
    assert laser_path.subs_num(0.010), 3


def test_subs_num_empty_base_case(empty_path) -> None:
    assert empty_path.subs_num(0.010), 3


def test_pickle(laser_path) -> None:
    filename = Path('test.pkl')
    laser_path.export(filename.name)
    assert filename.is_file()

    with open(filename, 'rb') as f:
        lp = dill.load(f)
    assert type(lp) == type(laser_path)
