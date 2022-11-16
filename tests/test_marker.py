import numpy as np
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


def test_scan(param) -> None:
    param["scan"] = 1.2
    with pytest.raises(ValueError):
        Marker(**param)


def test_repr(param) -> None:
    r = Marker(**param).__repr__()
    cname, _ = r.split("@")
    assert cname == "Marker"


def test_cross_init_pos(param) -> None:
    i_pos = [1, 2]
    mk = Marker(**param)
    mk.cross(i_pos)
    x, y, z, *_ = mk.points

    assert pytest.approx(x[-1]) == i_pos[0]
    assert pytest.approx(y[-1]) == i_pos[1]
    assert pytest.approx(z[-1]) == mk.depth

    i_pos = [0.0, 0.2, 5.9]
    mk = Marker(**param)
    mk.cross(i_pos)
    x, y, z, *_ = mk.points

    assert pytest.approx(x[-1]) == i_pos[0]
    assert pytest.approx(y[-1]) == i_pos[1]
    assert pytest.approx(z[-1]) == i_pos[2]

    i_pos = []
    mk = Marker(**param)
    with pytest.raises(ValueError):
        mk.cross(i_pos)

    i_pos = [1]
    mk = Marker(**param)
    with pytest.raises(ValueError):
        mk.cross(i_pos)

    i_pos = [8, 4, 3.45, 22.12, 89]
    mk = Marker(**param)
    with pytest.raises(ValueError):
        mk.cross(i_pos)


def test_cross_l_error(param) -> None:
    param["lx"] = None
    param["ly"] = 3
    mk = Marker(**param)
    with pytest.raises(ValueError):
        mk.cross([0, 0, 0])

    param["lx"] = 2
    param["ly"] = None
    mk = Marker(**param)
    with pytest.raises(ValueError):
        mk.cross([0, 0, 0])


def test_cross_l_default(param) -> None:
    i_pos = [0, 0, 0]
    mk = Marker(**param)
    mk.cross(i_pos)

    assert pytest.approx(np.max(mk.x)) == float(1.0)
    assert pytest.approx(np.min(mk.x)) == float(-1.0)
    assert pytest.approx(np.max(mk.y)) == float(0.025)
    assert pytest.approx(np.min(mk.y)) == float(-0.025)


def test_cross_l_custom(param) -> None:
    i_pos = [0, 0, 0]
    lx = 3
    ly = 1
    mk = Marker(**param)
    mk.cross(i_pos, lx, ly)

    assert pytest.approx(np.max(mk.x)) == float(1.5)
    assert pytest.approx(np.min(mk.x)) == float(-1.5)
    assert pytest.approx(np.max(mk.y)) == float(0.5)
    assert pytest.approx(np.min(mk.y)) == float(-0.5)


def test_cross_points(param) -> None:
    i_pos = [0, 0, 0.1]
    mk = Marker(**param)
    mk.cross(i_pos)

    x, y, z, f, s = mk.points
    np.testing.assert_almost_equal(x, np.array([-1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(y, np.array([0.0, 0.0, 0.0, 0.0, -0.025, -0.025, 0.025, 0.025, 0.0]))
    np.testing.assert_almost_equal(z, np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    np.testing.assert_almost_equal(f, np.array([0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]))
    np.testing.assert_almost_equal(s, np.array([0, 1, 1, 0, 0, 1, 1, 0, 0]))


def test_ruler_yticks(param) -> None:
    yt = []
    mk = Marker(**param)

    # empty list return None
    assert mk.ruler(yt) is None
    assert len(mk.y) == 0

    # check the given values are present in the matrix of points
    yt = np.array([0, 0.3, 0.6, 0.78, 0.99, 1, 45])
    mk = Marker(**param)
    mk.ruler(yt)
    assert pytest.approx(mk.y[0:-2:4]) == yt

    # check the list is ordered and unique
    yt = np.array([3, 2, 6, 4, 8, 9, 1, 3])
    mk = Marker(**param)
    mk.ruler(yt)
    assert pytest.approx(mk.y[0:-2:4]) == np.unique(yt)


def test_ruler_lx(param) -> None:
    # test default lx
    mk = Marker(**param)
    mk.ruler([1, 2, 3])
    assert pytest.approx(np.max(mk.x)) == mk.lx

    # test custom lx
    l = 5
    mk = Marker(**param)
    mk.ruler([1, 2, 3], lx=l)
    assert pytest.approx(np.max(mk.x)) == l

    # test none lx
    l = None
    param["lx"] = None
    mk = Marker(**param)
    with pytest.raises(ValueError):
        mk.ruler([1, 2, 3], lx=l)


def test_ruler_lx_short(param) -> None:
    # test default lx2
    mk = Marker(**param)
    mk.ruler([1, 2, 3])
    assert pytest.approx(mk.x[6]) == 0.75 * mk.lx

    # test custom lx2
    l = 5
    mk = Marker(**param)
    mk.ruler([1, 2, 3], lx2=l)
    assert pytest.approx(mk.x[6]) == l


def test_ruler_x_init(param) -> None:
    # test default x_init
    mk = Marker(**param)
    mk.ruler([1, 2, 3])
    assert pytest.approx(mk.x[0]) == mk.x_init

    # test custom x_init
    x0 = 0.0
    mk = Marker(**param)
    mk.ruler([1, 2, 3], x_init=x0)
    assert pytest.approx(mk.x[0]) == x0

    x1 = 2.0
    mk = Marker(**param)
    mk.ruler([1, 2, 3], x_init=x1)
    assert pytest.approx(mk.x[0]) == x1

    # test None x_init
    xi = None
    param["x_init"] = None
    mk = Marker(**param)
    with pytest.raises(ValueError):
        mk.ruler([1, 2, 3], x_init=xi)


def test_ruler_points(param) -> None:
    mk = Marker(**param)
    mk.ruler([1, 2, 3, 4], 5, 2, x_init=-2)

    x, y, z, f, s = mk.points
    np.testing.assert_almost_equal(
        x, np.array([-2.0, -2.0, 5.0, 5.0, -2.0, -2.0, 2.0, 2.0, -2.0, -2.0, 2.0, 2.0, -2.0, -2.0, 2.0, 2.0, -2.0])
    )
    np.testing.assert_almost_equal(
        y, np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 1.0])
    )
    np.testing.assert_almost_equal(
        z,
        np.array(
            [
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        f, np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 24.0])
    )
    np.testing.assert_almost_equal(
        s, np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    )
