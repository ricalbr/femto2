import numpy as np
import pytest

from femto.Waveguide import Waveguide


@pytest.fixture
def param() -> dict:
    p = {
            'scan': 6,
            'speed': 20.0,
            'y_init': 1.5,
            'z_init': 0.050,
            'speed_closed': 75,
            'samplesize': (100, 15),
            'depth': 0.035,
            'radius': 25,
            'pitch': 0.127,
            'int_dist': 0.005,
            'int_length': 0.0,
            'arm_length': 1.0,
            'ltrench': 1.5,
            'dz_bridge': 0.006,
    }
    return p


@pytest.fixture
def empty_wg(param) -> Waveguide:
    return Waveguide(**param)


@pytest.fixture
def waveguide(param) -> Waveguide:
    # create LaserPath instance
    wg = Waveguide(**param)

    # init_p = [0.0, 0.0, 0.0]
    # wg.start(init_p)
    # wg.linear([1, 2, 3], mode='ABS')
    # wg.arc_bend(wg.dy_bend)
    # wg.linear([5, 0, 0], mode='INC')
    # wg.spline_bridge(wg.pitch, wg.dz_bridge, )
    # wg.linear([30, wg.lasty, wg.lastz], mode='ABS')
    # wg.end()
    return wg


def test_default_values() -> None:
    wg = Waveguide()
    assert wg.scan == int(1)
    assert wg.speed == float(1.0)
    assert wg.x_init == float(-2.0)
    assert wg.y_init == float(0.0)
    assert wg.z_init == float(0.035)
    assert wg.lsafe == float(2.0)
    assert wg.speed_closed == float(5.0)
    assert wg.speed_pos == float(0.5)
    assert wg.cmd_rate_max == int(1200)
    assert wg.acc_max == int(500)
    assert wg.samplesize == (None, None)
    assert wg.depth == float(0.035)
    assert wg.radius == float(15)
    assert wg.pitch == float(0.080)
    assert wg.pitch_fa == float(0.127)
    assert wg.int_dist is None
    assert wg.int_length == float(0.0)
    assert wg.arm_length == float(0.0)
    assert wg.ltrench == float(1.0)
    assert wg.dz_bridge == float(0.007)
    assert wg.margin == float(1.0)


def test_wg_values(param) -> None:
    wg = Waveguide(**param)
    assert wg.scan == int(6)
    assert wg.speed == float(20.0)
    assert wg.x_init == float(-2.0)
    assert wg.y_init == float(1.5)
    assert wg.z_init == float(0.050)
    assert wg.lsafe == float(2.0)
    assert wg.speed_closed == float(75)
    assert wg.speed_pos == float(0.5)
    assert wg.cmd_rate_max == int(1200)
    assert wg.acc_max == int(500)
    assert wg.samplesize == (100, 15)
    assert wg.depth == float(0.035)
    assert wg.radius == float(25)
    assert wg.pitch == float(0.127)
    assert wg.pitch_fa == float(0.127)
    assert wg.int_dist == float(0.005)
    assert wg.int_length == float(0.0)
    assert wg.arm_length == float(1.0)
    assert wg.ltrench == float(1.5)
    assert wg.dz_bridge == float(0.006)
    assert wg.margin == float(1.0)


def test_from_dict(param) -> None:
    wg = Waveguide.from_dict(param)
    assert wg.scan == int(6)
    assert wg.speed == float(20.0)
    assert wg.x_init == float(-2.0)
    assert wg.y_init == float(1.5)
    assert wg.z_init == float(0.050)
    assert wg.lsafe == float(2.0)
    assert wg.speed_closed == float(75)
    assert wg.speed_pos == float(0.5)
    assert wg.cmd_rate_max == int(1200)
    assert wg.acc_max == int(500)
    assert wg.samplesize == (100, 15)
    assert wg.depth == float(0.035)
    assert wg.radius == float(25)
    assert wg.pitch == float(0.127)
    assert wg.pitch_fa == float(0.127)
    assert wg.int_dist == float(0.005)
    assert wg.int_length == float(0.0)
    assert wg.arm_length == float(1.0)
    assert wg.ltrench == float(1.5)
    assert wg.dz_bridge == float(0.006)
    assert wg.margin == float(1.0)


def test_z_init(param) -> None:
    param['z_init'] = None
    param['depth'] = 0.05
    wg = Waveguide(**param)
    assert wg.z_init == float(0.05)


def test_scan(param) -> None:
    param['scan'] = 1.2
    with pytest.raises(ValueError):
        Waveguide(**param)


def test_dy_bend_pitch_error(param) -> None:
    wg = Waveguide(**param)
    wg.pitch = None
    with pytest.raises(ValueError):
        wg.dy_bend()

    param['pitch'] = None
    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        wg.dy_bend()


def test_dy_bend_int_dist_error(param) -> None:
    wg = Waveguide(**param)
    wg.int_dist = None
    with pytest.raises(ValueError):
        wg.dy_bend()

    param['int_dist'] = None
    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        wg.dy_bend()


def test_dy_bend(param) -> None:
    wg = Waveguide(**param)
    assert wg.dy_bend == 0.061


def test_dx_bend_radius_error(param) -> None:
    wg = Waveguide(**param)
    wg.radius = None
    with pytest.raises(ValueError):
        wg.dx_bend()

    param['radius'] = None
    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        wg.dx_bend()


def test_dx_bend(param) -> None:
    wg = Waveguide(**param)
    assert pytest.approx(wg.dx_bend) == 2.469064


def test_dx_acc_none(param) -> None:
    param['int_length'] = None
    wg = Waveguide(**param)
    assert wg.dx_acc is None


def test_dx_acc(param) -> None:
    wg = Waveguide(**param)
    assert pytest.approx(wg.dx_acc) == 4.938129


def test_dx_acc_int_l(param) -> None:
    param['int_length'] = 2
    wg = Waveguide(**param)
    assert pytest.approx(wg.dx_acc) == 6.938129


def test_dx_mzi_none_intl(param) -> None:
    param['int_length'] = None
    wg = Waveguide(**param)
    assert wg.dx_mzi is None


def test_dx_mzi_none_arml(param) -> None:
    param['arm_length'] = None
    wg = Waveguide(**param)
    assert wg.dx_mzi is None


def test_dx_mzi(param) -> None:
    wg = Waveguide(**param)
    assert pytest.approx(wg.dx_mzi) == 10.876258


def test_dx_mzi_int_l(param) -> None:
    param['int_length'] = 2
    wg = Waveguide(**param)
    assert pytest.approx(wg.dx_mzi) == 14.876258


def test_dx_mzi_arml(param) -> None:
    param['arm_length'] = 3
    wg = Waveguide(**param)
    assert pytest.approx(wg.dx_mzi) == 12.876258


def test_get_sbend_param_error(param) -> None:
    dy = 0.08
    r = 0
    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        wg.get_sbend_parameter(dy, r)


def test_get_sbend_param(param) -> None:
    dy = 0.08
    r = 30
    wg = Waveguide(**param)
    assert type(wg.get_sbend_parameter(dy, r)) == tuple
    assert pytest.approx(wg.get_sbend_parameter(dy, r)[0]) == 0.0516455
    assert pytest.approx(wg.get_sbend_parameter(dy, r)[1]) == 3.097354


def test_get_sbend_length(param) -> None:
    dy = 0.127
    r = 15
    wg = Waveguide(**param)
    assert pytest.approx(wg.sbend_length(dy, r)) == 2.757512


def test_get_sbend_length_nil_dy(param) -> None:
    dy = 0.0
    r = 15
    wg = Waveguide(**param)
    assert pytest.approx(wg.sbend_length(dy, r)) == 0.0


def test_get_spline_raise_dy(param) -> None:
    disp_x = 0.5
    disp_z = 1
    radius = 30
    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        wg.get_spline_parameter(disp_x=disp_x, disp_z=disp_z, radius=radius)


def test_get_spline_raise_dz(param) -> None:
    disp_x = 0.5
    disp_y = 0.04
    radius = 20
    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        wg.get_spline_parameter(disp_x=disp_x, disp_y=disp_y, radius=radius)


def test_get_spline_nil_dispx(param) -> None:
    disp_y = 0.5
    disp_z = 0.6
    radius = 20
    wg = Waveguide(**param)
    dx, dy, dz, lc = wg.get_spline_parameter(disp_y=disp_y, disp_z=disp_z, radius=radius)

    assert pytest.approx(dx) == 7.865876
    assert dy == disp_y
    assert dz == disp_z
    assert pytest.approx(lc) == 7.917474


def test_get_spline_dispx(param) -> None:
    disp_x = 0.9
    disp_y = 0.5
    disp_z = 0.6
    radius = 40
    wg = Waveguide(**param)
    dx, dy, dz, lc = wg.get_spline_parameter(disp_x=disp_x, disp_y=disp_y, disp_z=disp_z, radius=radius)

    assert dx == disp_x
    assert dy == disp_y
    assert dz == disp_z
    assert pytest.approx(lc) == 1.191638


def test_repr(param) -> None:
    r = Waveguide(**param).__repr__()
    cname, _ = r.split('@')
    assert cname == 'Waveguide'


def test_start(param) -> None:
    wg = Waveguide(**param)
    wg.start()
    np.testing.assert_almost_equal(wg._x, np.array([-2.0, -2.0]))
    np.testing.assert_almost_equal(wg._y, np.array([1.5, 1.5]))
    np.testing.assert_almost_equal(wg._z, np.array([0.05, 0.05]))
    np.testing.assert_almost_equal(wg._f, np.array([0.5, 0.5]))
    np.testing.assert_almost_equal(wg._s, np.array([0.0, 1.0]))


def test_start_values(param) -> None:
    init_p = [0.0, 1.0, -0.1]
    speed_p = 1.25
    wg = Waveguide(**param)

    wg.start(init_pos=init_p, speed_pos=speed_p)
    np.testing.assert_almost_equal(wg._x, np.array([0.0, 0.0]))
    np.testing.assert_almost_equal(wg._y, np.array([1.0, 1.0]))
    np.testing.assert_almost_equal(wg._z, np.array([-0.1, -0.1]))
    np.testing.assert_almost_equal(wg._f, np.array([1.25, 1.25]))
    np.testing.assert_almost_equal(wg._s, np.array([0.0, 1.0]))


def test_start_value_error(param) -> None:
    init_p = [0.0, 1.0, -0.1, 0.3]
    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        wg.start(init_p)

    init_p = [0.0, ]
    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        wg.start(init_p)

    init_p = []
    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        wg.start(init_p)


def test_end(param) -> None:
    wg = Waveguide(**param)
    wg.start([0.0, 0.0, 0.0])
    wg.end()

    np.testing.assert_almost_equal(wg._x, np.array([0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(wg._y, np.array([0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(wg._z, np.array([0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(wg._f, np.array([0.5, 0.5, 0.5, 75.0]))
    np.testing.assert_almost_equal(wg._s, np.array([0.0, 1.0, 0.0, 0.0]))


def test_empty_end(param) -> None:
    wg = Waveguide(**param)
    with pytest.raises(IndexError):
        wg.end()


def test_linear_value_error(param) -> None:
    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        wg.start().linear([1, 1, 1], mode='rand').end()


def test_linear_invalid_increment(param) -> None:
    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        wg.start().linear([1, 2]).end()

    with pytest.raises(ValueError):
        wg.start().linear([1, 2, 3, 4]).end()


def test_linear_default_speed(param) -> None:
    wg = Waveguide(**param)
    wg.start().linear([1, 2, 3])
    assert wg._f[-1] == param['speed']


def test_linear_abs(param) -> None:
    wg = Waveguide(**param)
    init_p = [1, 1, 1]
    increm = [3, 4, 5]
    wg.start(init_p).linear(increm, mode='abs').end()

    np.testing.assert_almost_equal(wg._x, np.array([1.0, 1.0, 3.0, 3.0, 1.0]))
    np.testing.assert_almost_equal(wg._y, np.array([1.0, 1.0, 4.0, 4.0, 1.0]))
    np.testing.assert_almost_equal(wg._z, np.array([1.0, 1.0, 5.0, 5.0, 1.0]))
    np.testing.assert_almost_equal(wg._f, np.array([0.5, 0.5, 20., 20., 75.0]))
    np.testing.assert_almost_equal(wg._s, np.array([0.0, 1.0, 1.0, 0.0, 0.0]))


def test_linear_inc(param) -> None:
    wg = Waveguide(**param)
    init_p = [1, 1, 1]
    increm = [3, 4, 5]
    wg.start(init_p).linear(increm, mode='inc').end()

    np.testing.assert_almost_equal(wg._x, np.array([1.0, 1.0, 4.0, 4.0, 1.0]))
    np.testing.assert_almost_equal(wg._y, np.array([1.0, 1.0, 5.0, 5.0, 1.0]))
    np.testing.assert_almost_equal(wg._z, np.array([1.0, 1.0, 6.0, 6.0, 1.0]))
    np.testing.assert_almost_equal(wg._f, np.array([0.5, 0.5, 20., 20., 75.0]))
    np.testing.assert_almost_equal(wg._s, np.array([0.0, 1.0, 1.0, 0.0, 0.0]))


def test_linear_none(param) -> None:
    wg = Waveguide(**param)
    init_p = [1, 1, 1]
    increm = [4, None, None]
    wg.start(init_p).linear(increm, mode='abs').end()

    np.testing.assert_almost_equal(wg._x, np.array([1.0, 1.0, 4.0, 4.0, 1.0]))
    np.testing.assert_almost_equal(wg._y, np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(wg._z, np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(wg._f, np.array([0.5, 0.5, 20., 20., 75.0]))
    np.testing.assert_almost_equal(wg._s, np.array([0.0, 1.0, 1.0, 0.0, 0.0]))

    wg = Waveguide(**param)
    init_p = [1, 1, 1]
    increm = [5, None, None]
    wg.start(init_p).linear(increm, mode='inc').end()

    np.testing.assert_almost_equal(wg._x, np.array([1.0, 1.0, 6.0, 6.0, 1.0]))
    np.testing.assert_almost_equal(wg._y, np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(wg._z, np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(wg._f, np.array([0.5, 0.5, 20., 20., 75.0]))
    np.testing.assert_almost_equal(wg._s, np.array([0.0, 1.0, 1.0, 0.0, 0.0]))


def test_circ_input_validation(param) -> None:
    a_i, a_f = 0, np.pi / 2

    param['radius'] = None
    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        wg.start([0, 0, 0]).circ(a_i, a_f)

    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        wg.start([0, 0, 0]).circ(a_i, a_f, radius=None)

    param['radius'] = 20
    wg = Waveguide(**param)
    wg.radius = None
    with pytest.raises(ValueError):
        wg.start([0, 0, 0]).circ(a_i, a_f)

    param['speed'] = None
    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        wg.start([0, 0, 0]).circ(a_i, a_f)

    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        wg.start([0, 0, 0]).circ(a_i, a_f, speed=None)

    param['speed'] = 15
    wg = Waveguide(**param)
    wg.speed = None
    with pytest.raises(ValueError):
        wg.start([0, 0, 0]).circ(a_i, a_f)


def test_circ_length(param) -> None:
    a_i, a_f = 0, np.pi / 2

    # DEFAULT VALUES FROM PARAM
    wg = Waveguide(**param)
    wg.start([0, 0, 0]).circ(a_i, a_f)
    lc = np.abs(a_f - a_i) * wg.radius
    # add two points from the start method
    assert wg._x.size == int(np.ceil(lc * wg.cmd_rate_max / wg.speed)) + 2

    # CUSTOM RADIUS
    wg = Waveguide(**param)
    custom_r = 5
    wg.start([0, 0, 0]).circ(a_i, a_f, radius=custom_r)
    lc = np.abs(a_f - a_i) * custom_r
    assert wg._x.size == int(np.ceil(lc * wg.cmd_rate_max / wg.speed)) + 2

    # CUSTOM SPEED
    wg = Waveguide(**param)
    custom_f = 5
    wg.start([0, 0, 0]).circ(a_i, a_f, speed=custom_f)
    lc = np.abs(a_f - a_i) * wg.radius
    assert wg._x.size == int(np.ceil(lc * wg.cmd_rate_max / custom_f)) + 2


def test_circ_coordinates(param) -> None:
    a_i, a_f = 1.5 * np.pi, 0

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).circ(a_i, a_f)
    assert pytest.approx(wg._x[-1]) == wg.radius
    assert pytest.approx(wg._y[-1]) == wg.radius
    assert wg._z[-1] == wg._z[0]
    wg.end()

    a_i, a_f = 1.5 * np.pi, 1.75 * np.pi

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).circ(a_i, a_f)
    assert pytest.approx(wg._x[-1]) == wg.radius / np.sqrt(2)
    assert pytest.approx(wg._y[-1]) == wg.radius * (1 - 1 / np.sqrt(2))
    assert wg._z[-1] == wg._z[0]
    wg.end()


def test_circ_negative_radius(param) -> None:
    param['radius'] = -60
    a_i, a_f = 1.5 * np.pi, 0

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).circ(a_i, a_f)
    assert pytest.approx(wg._x[-1]) == np.abs(wg.radius)
    assert pytest.approx(wg._y[-1]) == np.abs(wg.radius)
    assert wg._z[-1] == wg._z[0]
    wg.end()

    a_i, a_f = 1.5 * np.pi, 1.75 * np.pi

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).circ(a_i, a_f)
    assert pytest.approx(wg._x[-1]) == np.abs(wg.radius) / np.sqrt(2)
    assert pytest.approx(wg._y[-1]) == np.abs(wg.radius) * (1 - 1 / np.sqrt(2))
    assert wg._z[-1] == wg._z[0]
    wg.end()

# def test_arc_bend(param):
# assert dy sia giusto
# assert segno di dy
# assert lunghezza sia 2 dx
# assert raggio None (?)

# def test_arc_acc(param):
