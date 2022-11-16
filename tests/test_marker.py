import pytest

from femto.Marker import Marker


@pytest.fixture
def param() -> dict:
    p = {
        "scan": 1,
        "speed": 2.0,
        "y_init": 1.5,
        "z_init": 0.050,
        "speed_closed": 24,
        "depth": 0.001,
        "lx": 2.0,
        "ly": 0.050,
    }
    return p


@pytest.fixture
def empty_mk(param) -> Marker:
    return Marker(**param)


def test_default_values() -> None:
    mk = Marker()
    assert mk.scan == int(1)
    assert mk.speed == float(1.0)
    assert mk.x_init == float(-2.0)
    assert mk.y_init == float(0.0)
    assert mk.z_init == float(0.0)
    assert mk.lsafe == float(2.0)
    assert mk.speed_closed == float(5.0)
    assert mk.speed_pos == float(0.5)
    assert mk.cmd_rate_max == int(1200)
    assert mk.acc_max == int(500)
    assert mk.samplesize == (None, None)
    assert mk.depth == float(0.0)
    assert mk.lx == float(1.0)
    assert mk.ly == float(0.060)


def test_mk_values(param) -> None:
    mk = Marker(**param)
    assert mk.scan == int(1)
    assert mk.speed == float(2.0)
    assert mk.x_init == float(-2.0)
    assert mk.y_init == float(1.5)
    assert mk.z_init == float(0.050)
    assert mk.lsafe == float(2.0)
    assert mk.speed_closed == float(24)
    assert mk.speed_pos == float(0.5)
    assert mk.cmd_rate_max == int(1200)
    assert mk.acc_max == int(500)
    assert mk.samplesize == (None, None)
    assert mk.depth == float(0.001)
    assert mk.lx == float(2.0)
    assert mk.ly == float(0.050)


def test_mk_from_dict(param) -> None:
    mk = Marker.from_dict(param)
    assert mk.scan == int(1)
    assert mk.speed == float(2.0)
    assert mk.x_init == float(-2.0)
    assert mk.y_init == float(1.5)
    assert mk.z_init == float(0.050)
    assert mk.lsafe == float(2.0)
    assert mk.speed_closed == float(24)
    assert mk.speed_pos == float(0.5)
    assert mk.cmd_rate_max == int(1200)
    assert mk.acc_max == int(500)
    assert mk.samplesize == (None, None)
    assert mk.depth == float(0.001)
    assert mk.lx == float(2.0)
    assert mk.ly == float(0.050)


def test_z_init(param) -> None:
    param["z_init"] = None
    param["depth"] = -0.001
    mk = Marker(**param)
    assert mk.z_init == float(-0.001)
