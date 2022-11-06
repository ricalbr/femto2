import pytest

from src.femto.Waveguide import _Waveguide


@pytest.fixture
def param() -> dict:
    p = {
            'scan'        : 6,
            'speed'       : 20.0,
            'y_init'      : 1.5,
            'z_init'      : 0.050,
            'speed_closed': 75,
            'samplesize'  : (100, 15),
            'depth'       : 0.035,
            'radius'      : 25,
            'pitch'       : 0.127,
            'int_dist'    : 0.005,
            'int_length'  : 0.0,
            'arm_length'  : 1.0,
            'ltrench'     : 1.5,
            'dz_bridge'   : 0.006,
    }
    return p


@pytest.fixture
def empty_path(param) -> _Waveguide:
    return _Waveguide(**param)


@pytest.fixture
def waveguide(param) -> _Waveguide:
    # create LaserPath instance
    wg = _Waveguide(**param)

    # add path
    pass

    return wg


def test_default_values() -> None:
    wg = _Waveguide()
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


def test_z_init(param):
    param['z_init'] = None
    param['depth'] = 0.05
    wg = _Waveguide(**param)
    assert wg.z_init == float(0.05)


def test_scan(param):
    param['scan'] = 1.2
    with pytest.raises(ValueError):
        _Waveguide(**param)


def test_dy_bend_pitch_error(waveguide):
    waveguide.pitch = None
    with pytest.raises(ValueError):
        waveguide.dy_bend()


def test_dy_bend_int_dist_error(waveguide):
    waveguide.int_dist = None
    with pytest.raises(ValueError):
        waveguide.dy_bend()


def test_dy_bend(waveguide):
    assert waveguide.dy_bend == 0.061


def test_dx_bend_radius_error(waveguide):
    waveguide.radius = None
    with pytest.raises(ValueError):
        waveguide.dx_bend()


def test_dx_bend(waveguide):
    assert pytest.approx(waveguide.dx_bend, 2.469064)


def test_dx_acc_none(param):
    param['int_length'] = None
    wg = _Waveguide(**param)
    assert wg.dx_acc is None


def test_dx_acc(waveguide):
    assert pytest.approx(waveguide.dx_acc, 4.938129)


def test_dx_acc_int_l(param):
    param['int_length'] = 2
    wg = _Waveguide(**param)
    assert pytest.approx(wg.dx_acc, 6.938129)


def test_dx_mzi_none_intl(param):
    param['int_length'] = None
    wg = _Waveguide(**param)
    assert wg.dx_mzi is None


def test_dx_mzi_none_arml(param):
    param['arm_length'] = None
    wg = _Waveguide(**param)
    assert wg.dx_mzi is None


def test_dx_mzi(waveguide):
    assert pytest.approx(waveguide.dx_mzi, 10.876258)


def test_dx_mzi_int_l(param):
    param['int_length'] = 2
    wg = _Waveguide(**param)
    assert pytest.approx(wg.dx_mzi, 14.876258)


def test_dx_mzi_arml(param):
    param['arm_length'] = 3
    wg = _Waveguide(**param)
    assert pytest.approx(wg.dx_mzi, 13.876258)


def test_get_sbend_param_error(waveguide):
    dy = 0.08
    r = 0
    with pytest.raises(ValueError):
        waveguide.get_sbend_parameter(dy, r)


def test_get_sbend_param(waveguide):
    dy = 0.08
    r = 30
    assert type(waveguide.get_sbend_parameter(dy, r)) == tuple
    assert pytest.approx(waveguide.get_sbend_parameter(dy, r)[0], 0.999848)
    assert pytest.approx(waveguide.get_sbend_parameter(dy, r)[1], 1.046985)


def test_get_sbend_length(waveguide):
    dy = 0.127
    r = 15
    assert pytest.approx(waveguide.sbend_length(dy, r), 2.757512)


def test_get_sbend_length_nil_dy(waveguide):
    dy = 0.0
    r = 15
    assert pytest.approx(waveguide.sbend_length(dy, r), 0.0)


def test_get_spline_raise_dy(waveguide):
    disp_x = 0.5
    disp_z = 1
    radius = 30
    with pytest.raises(ValueError):
        waveguide.get_spline_parameter(disp_x=disp_x, disp_z=disp_z, radius=radius)


def test_get_spline_raise_dz(waveguide):
    disp_x = 0.5
    disp_y = 0.04
    radius = 20
    with pytest.raises(ValueError):
        waveguide.get_spline_parameter(disp_x=disp_x, disp_y=disp_y, radius=radius)


def test_get_spline_nil_dispx(waveguide):
    disp_x = 0
    disp_y = 0.5
    disp_z = 0.6
    radius = 20
    dx, dy, dz, lc = waveguide.get_spline_parameter(disp_x, disp_y, disp_z, radius)

    assert pytest.approx(dx, 7.865876)
    assert dy == disp_y
    assert dz == disp_z
    assert pytest.approx(lc, 7.917474)


def test_get_spline_dispx(waveguide):
    disp_x = 0.9
    disp_y = 0.5
    disp_z = 0.6
    radius = 40
    dx, dy, dz, lc = waveguide.get_spline_parameter(disp_x, disp_y, disp_z, radius)

    assert dx == disp_x
    assert dy == disp_y
    assert dz == disp_z
    assert pytest.approx(lc, 1.191638)
