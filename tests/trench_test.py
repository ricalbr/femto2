from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from femto.helpers import almost_equal
from femto.trench import Trench
from femto.trench import TrenchColumn
from shapely.geometry import box
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon


@pytest.fixture
def param() -> dict:
    p = {
        'x_center': 6,
        'y_min': 1,
        'y_max': 2,
        'bridge': 0.050,
        'length': 3,
        'nboxz': 4,
        'z_off': -0.02,
        'h_box': 0.080,
        'base_folder': '',
        'deltaz': 0.002,
        'delta_floor': 0.0015,
        'beam_waist': 0.002,
        'round_corner': 0.010,
        'u': [28, 29.47],
        'speed': 5,
        'speed_closed': 5,
        'speed_pos': 0.5,
    }
    return p


@pytest.fixture
def tc(param) -> TrenchColumn:
    return TrenchColumn(**param)


@pytest.fixture
def poly() -> Polygon:
    x = [1.0, 0.0, 0.0, 0.0, 1.0]
    y = [0.0, 0.0, 0.0, 1.0, 1.0]

    return Polygon(zip(x, y))


def test_init_default(poly) -> None:
    tc = Trench(poly)

    assert tc.block == poly
    assert tc.delta_floor == 0.001
    assert tc.floor_length == 0.0
    assert tc.wall_length == 0.0


def test_init_values(poly) -> None:
    df = 0.005
    tc = Trench(poly, delta_floor=df)

    assert tc.block == poly
    assert tc.delta_floor == df
    assert tc.floor_length == 0.0
    assert tc.wall_length == 0.0


def test_repr(poly) -> None:
    r = Trench(poly).__repr__()
    cname, _ = r.split('@')
    assert cname == 'Trench'


def test_repr_trenchcol(param) -> None:
    r = TrenchColumn(**param).__repr__()
    cname, _ = r.split('@')
    assert cname == 'TrenchColumn'


def test_lt() -> None:
    x = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    y1 = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    y2 = np.array([0.0, 0.0, 0.0, 1.0, 1.0]) + 5

    p1 = Polygon(zip(x, y1))
    p2 = Polygon(zip(x, y2))

    tc1 = Trench(p1)
    tc2 = Trench(p2)

    assert tc1 < tc2
    assert tc1 <= tc2


def test_gt() -> None:
    x = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    y1 = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    y2 = np.array([0.0, 0.0, 0.0, 1.0, 1.0]) - 5

    p1 = Polygon(zip(x, y1))
    p2 = Polygon(zip(x, y2))

    tc1 = Trench(p1)
    tc2 = Trench(p2)

    assert tc1 > tc2
    assert tc1 >= tc2


def test_eq() -> None:
    x = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    y1 = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    y2 = np.array([0.0, 0.0, 0.0, 1.0, 1.0]) + 1e-8

    p1 = Polygon(zip(x, y1))
    p2 = Polygon(zip(x, y2))

    tc1 = Trench(p1)
    tc2 = Trench(p2)

    assert tc1 == tc2


def test_eq_false() -> None:
    x = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    y1 = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    y2 = np.array([0.0, 0.0, 0.0, 1.0, 1.0]) + 1e-5

    p1 = Polygon(zip(x, y1))
    p2 = Polygon(zip(x, y2))

    tc1 = Trench(p1)
    tc2 = Trench(p2)

    assert not tc1 == tc2


def test_lt_gt_eq_raise(poly) -> None:
    from femto.waveguide import Waveguide

    tc = Trench(poly)
    wg = Waveguide()

    with pytest.raises(TypeError):
        tc == wg
    with pytest.raises(TypeError):
        tc > wg
    with pytest.raises(TypeError):
        tc >= wg
    with pytest.raises(TypeError):
        tc < wg
    with pytest.raises(TypeError):
        tc <= wg


def test_border(poly) -> None:
    tc = Trench(poly)
    xb = tc.xborder
    yb = tc.yborder

    np.testing.assert_array_equal(xb, np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0]))
    np.testing.assert_array_equal(yb, np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0]))
    assert type(xb) == np.ndarray
    assert type(yb) == np.ndarray


@pytest.mark.parametrize(
    'x, y, xmin, xmax, ymin, ymax',
    [
        ([-6, 29, 0, 26, 1], [24, 15, -10, 3, 20], -6, 29, -10, 24),
        ([14, 27, 28, 1, 16], [-7, 10, 1, -8, -10], 1, 28, -10, 10),
        ([-2, -1, 15, 24, 12], [29, 12, 15, -2, 1], -2, 24, -2, 29),
        ([21, -3, -2, 9, 6], [-5, -10, 19, 12, 7], -3, 21, -10, 19),
        ([-4, -1, 19, -8, 11], [10, 7, 6, 24, -2], -8, 19, -2, 24),
    ],
)
def test_bounds(x, y, xmin, xmax, ymin, ymax) -> None:
    poly = Polygon(zip(x, y))
    tc = Trench(poly)

    assert tc.xmin == xmin
    assert tc.xmax == xmax
    assert tc.ymin == ymin
    assert tc.ymax == ymax


def test_centroid(poly) -> None:
    assert Trench(poly).center == (0.5, 0.5)


@pytest.mark.parametrize(
    'p, offset, p_expected',
    [
        (Point(0, 0), 1, [Point(0, 0).buffer(1)]),
        (Point(0, 0), -1, [Point(0, 0).buffer(1).buffer(-2)]),
        (
            MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(5, 5), (5, 6), (6, 6), (5, 6)])]),
            -0.1,
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]).buffer(-0.1),
                Polygon([(5, 5), (5, 6), (6, 6), (5, 6)]).buffer(-0.1),
            ],
        ),
        (
            MultiPolygon([Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]), Point(7, 7).buffer(0.5)]),
            -1,
            [Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]).buffer(-1), Polygon()],
        ),
        (
            Polygon([(0, 0), (0, 3), (3, 3), (3, 0), (2, 0), (2, 2), (1, 2), (1, 1), (2, 1), (2, 0), (0, 0)]),
            -1,
            [Polygon()],
        ),
        (
            Point(-1, 0).buffer(1.5).union(Point(1, 0).buffer(1.5)),
            -1.2,
            Point(-1, 0).buffer(1.5).union(Point(1, 0).buffer(1.5)).buffer(-1.2).geoms,
        ),
    ],
)
def test_buffer_polygon(p, offset, p_expected) -> None:
    tc = Trench(block=Polygon())
    buff_polygon = tc.buffer_polygon(p, offset)

    for (p_calc, p_exp) in zip(buff_polygon, p_expected):
        assert almost_equal(p_calc, p_exp)


def test_toolpath() -> None:
    p = Point(0, 0).buffer(5)
    pp = deepcopy(p)
    tc = Trench(p, delta_floor=0.1)

    for path in tc.toolpath():
        assert pytest.approx(tc.wall_length) == pp.length
        np.testing.assert_array_equal(path, np.array(p.exterior.coords).T)
        p = p.buffer(-0.1)


def test_trenchcol_default() -> None:
    x, ymin, ymax = 1, 2, 5
    tcol = TrenchColumn(x, ymin, ymax)

    assert tcol.x_center == x
    assert tcol.y_min == ymin
    assert tcol.y_max == ymax
    assert tcol.bridge == float(0.026)
    assert tcol.length == float(1)
    assert tcol.nboxz == int(4)
    assert tcol.z_off == float(-0.020)
    assert tcol.h_box == float(0.075)
    assert tcol.base_folder == ''
    assert tcol.deltaz == float(0.0015)
    assert tcol.delta_floor == float(0.001)
    assert tcol.beam_waist == float(0.004)
    assert tcol.round_corner == float(0.010)
    assert tcol.u is None
    assert tcol.speed == float(4)
    assert tcol.speed_closed == float(5)
    assert tcol.speed_pos == float(0.5)

    assert tcol.CWD == Path.cwd()
    assert tcol._trench_list == []


def test_trenchcol_param(param) -> None:
    tcol = TrenchColumn(**param)

    assert tcol.x_center == float(6)
    assert tcol.y_min == float(1)
    assert tcol.y_max == float(2)
    assert tcol.bridge == float(0.050)
    assert tcol.length == float(3)
    assert tcol.nboxz == int(4)
    assert tcol.z_off == float(-0.020)
    assert tcol.h_box == float(0.080)
    assert tcol.base_folder == ''
    assert tcol.deltaz == float(0.002)
    assert tcol.delta_floor == float(0.0015)
    assert tcol.beam_waist == float(0.002)
    assert tcol.round_corner == float(0.010)
    assert tcol.u == [28, 29.47]
    assert tcol.speed == float(5)
    assert tcol.speed_closed == float(5)
    assert tcol.speed_pos == float(0.5)

    assert tcol.CWD == Path.cwd()
    assert tcol._trench_list == []


def test_trenchcol_adj_bridge(tc, param) -> None:
    assert tc.adj_bridge == param['bridge'] / 2 + param['beam_waist'] + param['round_corner']


@pytest.mark.parametrize(
    'h_box, z_off, deltaz, exp',
    [
        (0, 0.020, 0.001, 20),
        (0.300, 0.0, 0.001, 300),
        (0.100, 0.05, 0.002, 25),
        (-0.150, 0.0, 0.015, 10),
    ],
)
def test_trenchcol_n_repeat(h_box, z_off, deltaz, exp, param) -> None:
    param['h_box'] = h_box
    param['z_off'] = z_off
    param['deltaz'] = deltaz
    tcol = TrenchColumn(**param)
    assert tcol.n_repeat == exp


def test_trenchcol_fabrication_time_empty(tc) -> None:
    assert tc.fabrication_time == 0.0


def test_trenchcol_iterator(tc) -> None:
    t1 = Trench(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
    t2 = Trench(Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]))
    t3 = Trench(Polygon([(4, 4), (5, 4), (5, 5), (4, 5)]))
    tc._trench_list.extend([t1, t2, t3])

    tc_iter = tc.__iter__()

    assert next(tc_iter) == t1
    assert next(tc_iter) == t2
    assert next(tc_iter) == t3
    with pytest.raises(StopIteration):
        assert next(tc_iter)


def test_trenchcol_rect(tc, param) -> None:
    bb = tc.rect

    assert almost_equal(
        bb,
        box(
            param['x_center'] - param['length'] / 2,
            param['y_min'],
            param['x_center'] + param['length'] / 2,
            param['y_max'],
        ),
    )


def test_trenchcol_rect_empty() -> None:
    tc = TrenchColumn(1, 2, 3, length=None)
    bb = tc.rect
    assert bb == Polygon()


def test_trenchcol_fabtime(tc, param) -> None:
    t1 = Trench(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
    t2 = Trench(Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]))
    t3 = Trench(Polygon([(4, 4), (5, 4), (5, 5), (4, 5)]))
    tc._trench_list.extend([t1, t2, t3])

    assert pytest.approx(tc.fabrication_time) == sum(
        param['nboxz'] * (tc.n_repeat * t.wall_length + t.floor_length) for t in tc._trench_list
    )


def test_trenchcol_dig() -> None:
    p = {
        'x_center': 0.0,
        'y_min': 0.0,
        'y_max': 9.0,
        'length': 4.0,
        'bridge': 1.0,
        'beam_waist': 0.1,
        'round_corner': 0.0,
    }
    tc = TrenchColumn(**p)

    coords = [[(-5.0, 3.0), (5.0, 3.0)], [(-5.0, 6.0), (5.0, 6.0)]]
    comp_box = [box(-2.0, 0.0, 2.0, 2.4), box(-2.0, 3.6, 2.0, 5.4), box(-2.0, 6.6, 2.0, 9.0)]

    tc._dig(coords)

    for (t, c) in zip(tc._trench_list, comp_box):
        assert tc.normalize(c).equals_exact(t.block, tolerance=1e-8)
        assert almost_equal(tc.normalize(c), t.block)


def test_trenchcol_dig_remove() -> None:
    p = {
        'x_center': 0.0,
        'y_min': 0.0,
        'y_max': 9.0,
        'length': 4.0,
        'bridge': 1.0,
        'beam_waist': 0.1,
        'round_corner': 0.0,
    }
    tc = TrenchColumn(**p)

    coords = [[(-5.0, 3.0), (5.0, 3.0)], [(-5.0, 6.0), (5.0, 6.0)]]
    comp_box = [box(-2.0, 0.0, 2.0, 2.4), box(-2.0, 6.6, 2.0, 9.0)]

    tc._dig(coords, remove=[1])

    for (t, c) in zip(tc._trench_list, comp_box):
        assert tc.normalize(c).equals_exact(t.block, tolerance=1e-8)


def test_trenchcol_dig_remove_all() -> None:
    p = {
        'x_center': 0.0,
        'y_min': 0.0,
        'y_max': 9.0,
        'length': 4.0,
        'bridge': 1.0,
        'beam_waist': 0.1,
        'round_corner': 0.0,
    }
    tc = TrenchColumn(**p)

    coords = [[(-5.0, 3.0), (5.0, 3.0)], [(-5.0, 6.0), (5.0, 6.0)]]

    tc._dig(coords, remove=[0, 1, 2])
    assert tc._trench_list == []


def test_dig_from_waveguide(tc):
    from femto.helpers import dotdict
    from femto.waveguide import Waveguide

    wgs = []
    PARAM_WG = dotdict(speed=20, radius=25, pitch=0.080, int_dist=0.007)
    wg = Waveguide(**PARAM_WG)
    wg.start([0, 1.25, 0]).linear([9, 1.75, 0], mode='ABS').end()
    wgs.append(wg)

    wg = Waveguide(**PARAM_WG)
    wg.start([0, 1.25, 0]).linear([9, 1.75, 0], mode='ABS').end()
    wgs.append(wg)

    assert tc.dig_from_waveguide(wgs) is None
    assert bool(tc._trench_list)


def test_dig_from_waveguide_empty(tc):
    from femto.waveguide import Waveguide

    wg_list = [Waveguide() for _ in range(4)]
    assert tc.dig_from_waveguide(wg_list) is None
    assert not bool(tc._trench_list)


def test_dig_from_waveguide_raise(tc):
    from femto.waveguide import Waveguide

    wg_list = [Waveguide() for _ in range(4)]
    wg_list.append(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        tc.dig_from_waveguide(wg_list)


def test_dig_from_array(tc):
    arr_list = [
        np.array([[-5, 9], [1.2, 1.2]]).T,
        np.array([[-5, 9], [1.3, 1.3]]).T,
        np.array([[-5, 9], [1.4, 1.4]]).T,
        np.array([[-5, 9], [1.5, 1.5]]).T,
    ]
    assert tc.dig_from_array(arr_list) is None
    assert bool(tc._trench_list)

    arr_list = [
        np.array([[-5, 1.2], [0, 1.2], [5, 1.2], [9, 1.2]]).T,
        np.array([[-5, 1.3], [0, 1.3], [5, 1.3], [9, 1.3]]).T,
    ]
    print(arr_list[0])
    assert tc.dig_from_array(arr_list) is None
    assert bool(tc._trench_list)


def test_dig_from_array_empty(tc):
    arr_list = [np.array([[1, 2, 3, 4], [9, 9, 9, 9]]).T for _ in range(4)]
    assert tc.dig_from_array(arr_list) is None
    assert not bool(tc._trench_list)


def test_dig_from_array_raise(tc):
    from femto.waveguide import Waveguide

    arr_list = [np.array([1, 2, 3, 4]) for _ in range(4)]
    arr_list.append(Waveguide())
    with pytest.raises(ValueError):
        tc.dig_from_array(arr_list)


def test_normalize(tc):
    from itertools import combinations
    from femto.helpers import almost_equal

    poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    poly2 = Polygon([(0, 1), (1, 1), (1, 0), (0, 0)])
    poly3 = Polygon([(0, 1), (1, 1.00000001), (1, 0), (0, 0)])
    poly4 = Polygon([(0, 0), (0, 1), (1, 1.00000001), (1, 0)])
    poly = [poly1, poly2, poly3, poly4]

    for p1, p2 in combinations(poly, 2):
        assert almost_equal(p1, p2)
