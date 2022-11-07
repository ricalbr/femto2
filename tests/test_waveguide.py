import numpy as np
import pytest

from src.femto.Waveguide import Waveguide


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

    # add path
    pass

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


def test_wg_values(waveguide) -> None:
    assert waveguide.scan == int(6)
    assert waveguide.speed == float(20.0)
    assert waveguide.x_init == float(-2.0)
    assert waveguide.y_init == float(1.5)
    assert waveguide.z_init == float(0.050)
    assert waveguide.lsafe == float(2.0)
    assert waveguide.speed_closed == float(75)
    assert waveguide.speed_pos == float(0.5)
    assert waveguide.cmd_rate_max == int(1200)
    assert waveguide.acc_max == int(500)
    assert waveguide.samplesize == (100, 15)
    assert waveguide.depth == float(0.035)
    assert waveguide.radius == float(25)
    assert waveguide.pitch == float(0.127)
    assert waveguide.pitch_fa == float(0.127)
    assert waveguide.int_dist == float(0.005)
    assert waveguide.int_length == float(0.0)
    assert waveguide.arm_length == float(1.0)
    assert waveguide.ltrench == float(1.5)
    assert waveguide.dz_bridge == float(0.006)
    assert waveguide.margin == float(1.0)


def test_z_init(param) -> None:
    param['z_init'] = None
    param['depth'] = 0.05
    wg = Waveguide(**param)
    assert wg.z_init == float(0.05)


def test_scan(param) -> None:
    param['scan'] = 1.2
    with pytest.raises(ValueError):
        Waveguide(**param)


def test_dy_bend_pitch_error(waveguide) -> None:
    waveguide.pitch = None
    with pytest.raises(ValueError):
        waveguide.dy_bend()


def test_dy_bend_int_dist_error(waveguide) -> None:
    waveguide.int_dist = None
    with pytest.raises(ValueError):
        waveguide.dy_bend()


def test_dy_bend(waveguide) -> None:
    assert waveguide.dy_bend == 0.061


def test_dx_bend_radius_error(waveguide) -> None:
    waveguide.radius = None
    with pytest.raises(ValueError):
        waveguide.dx_bend()


def test_dx_bend(waveguide) -> None:
    assert pytest.approx(waveguide.dx_bend) == 2.469064


def test_dx_acc_none(param) -> None:
    param['int_length'] = None
    wg = Waveguide(**param)
    assert wg.dx_acc is None


def test_dx_acc(waveguide):
    assert pytest.approx(waveguide.dx_acc) == 4.938129


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


def test_dx_mzi(waveguide) -> None:
    assert pytest.approx(waveguide.dx_mzi) == 10.876258


def test_dx_mzi_int_l(param) -> None:
    param['int_length'] = 2
    wg = Waveguide(**param)
    assert pytest.approx(wg.dx_mzi) == 14.876258


def test_dx_mzi_arml(param):
    param['arm_length'] = 3
    wg = Waveguide(**param)
    assert pytest.approx(wg.dx_mzi) == 12.876258


def test_get_sbend_param_error(waveguide) -> None:
    dy = 0.08
    r = 0
    with pytest.raises(ValueError):
        waveguide.get_sbend_parameter(dy, r)


def test_get_sbend_param(waveguide) -> None:
    dy = 0.08
    r = 30
    assert type(waveguide.get_sbend_parameter(dy, r)) == tuple
    assert pytest.approx(waveguide.get_sbend_parameter(dy, r)[0]) == 0.0516455
    assert pytest.approx(waveguide.get_sbend_parameter(dy, r)[1]) == 3.097354


def test_get_sbend_length(waveguide) -> None:
    dy = 0.127
    r = 15
    assert pytest.approx(waveguide.sbend_length(dy, r)) == 2.757512


def test_get_sbend_length_nil_dy(waveguide) -> None:
    dy = 0.0
    r = 15
    assert pytest.approx(waveguide.sbend_length(dy, r)) == 0.0


def test_get_spline_raise_dy(waveguide) -> None:
    disp_x = 0.5
    disp_z = 1
    radius = 30
    with pytest.raises(ValueError):
        waveguide.get_spline_parameter(disp_x=disp_x, disp_z=disp_z, radius=radius)


def test_get_spline_raise_dz(waveguide) -> None:
    disp_x = 0.5
    disp_y = 0.04
    radius = 20
    with pytest.raises(ValueError):
        waveguide.get_spline_parameter(disp_x=disp_x, disp_y=disp_y, radius=radius)


def test_get_spline_nil_dispx(waveguide) -> None:
    disp_y = 0.5
    disp_z = 0.6
    radius = 20
    dx, dy, dz, lc = waveguide.get_spline_parameter(disp_y=disp_y, disp_z=disp_z, radius=radius)

    assert pytest.approx(dx) == 7.865876
    assert dy == disp_y
    assert dz == disp_z
    assert pytest.approx(lc) == 7.917474


def test_get_spline_dispx(waveguide) -> None:
    disp_x = 0.9
    disp_y = 0.5
    disp_z = 0.6
    radius = 40
    dx, dy, dz, lc = waveguide.get_spline_parameter(disp_x=disp_x, disp_y=disp_y, disp_z=disp_z, radius=radius)

    assert dx == disp_x
    assert dy == disp_y
    assert dz == disp_z
    assert pytest.approx(lc) == 1.191638


def test_repr(waveguide) -> None:
    r = waveguide.__repr__()
    cname, _ = r.split('@')
    assert cname == 'Waveguide'


def test_start(waveguide) -> None:
    waveguide.start()
    np.testing.assert_almost_equal(waveguide._x, np.array([-2.0, -2.0]))
    np.testing.assert_almost_equal(waveguide._y, np.array([1.5, 1.5]))
    np.testing.assert_almost_equal(waveguide._z, np.array([0.05, 0.05]))
    np.testing.assert_almost_equal(waveguide._f, np.array([0.5, 0.5]))
    np.testing.assert_almost_equal(waveguide._s, np.array([0.0, 1.0]))


def test_start_values(waveguide) -> None:
    init_p = [0.0, 1.0, -0.1]
    speed_p = 1.25

    waveguide.start(init_pos=init_p, speed_pos=speed_p)
    np.testing.assert_almost_equal(waveguide._x, np.array([0.0, 0.0]))
    np.testing.assert_almost_equal(waveguide._y, np.array([1.0, 1.0]))
    np.testing.assert_almost_equal(waveguide._z, np.array([-0.1, -0.1]))
    np.testing.assert_almost_equal(waveguide._f, np.array([1.25, 1.25]))
    np.testing.assert_almost_equal(waveguide._s, np.array([0.0, 1.0]))


def test_start_value_error(waveguide) -> None:
    init_p = [0.0, 1.0, -0.1, 0.3]
    with pytest.raises(ValueError):
        waveguide.start(init_p)
    init_p = [0.0, ]
    with pytest.raises(ValueError):
        waveguide.start(init_p)
    init_p = []
    with pytest.raises(ValueError):
        waveguide.start(init_p)