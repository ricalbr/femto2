from __future__ import annotations

import os
import pathlib
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest
from femto.curves import sin
from femto.device import Layer
from femto.helpers import dotdict
from femto.helpers import flatten
from femto.marker import Marker
from femto.trench import Trench
from femto.trench import TrenchColumn
from femto.waveguide import Waveguide
from shapely.geometry import Polygon


@pytest.fixture
def ss_param() -> dict:
    return dict(
        book_name='custom_book_name.xlsx',
        sheet_name='custom_sheet_name',
        columns_names=['name', 'power', 'speed', 'scan', 'depth', 'int_dist', 'yin', 'yout', 'obs'],
    )


@pytest.fixture
def gc_param() -> dict:
    p = dict(
        filename='testCell.pgm',
        laser='PHAROS',
        shift_origin=(0.5, 0.5),
        samplesize=(25, 1),
    )
    return p


@pytest.fixture
def list_wg() -> list[Waveguide]:
    PARAM_WG = dotdict(speed=20, radius=25, pitch=0.080, int_dist=0.007, samplesize=(25, 3))

    coup = [Waveguide(**PARAM_WG) for _ in range(5)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch, 0.035])
        wg.linear([5, 0, 0])
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        wg.linear([5, 0, 0])
        wg.end()
    return coup


@pytest.fixture
def list_mk() -> list[Marker]:
    PARAM_MK = dotdict(scan=1, speed=2, speed_pos=5, speed_closed=5, depth=0.000, lx=1, ly=1)
    markers = []
    for (x, y) in zip(range(4, 8), range(3, 7)):
        m = Marker(**PARAM_MK)
        m.cross([x, y])
        markers.append(m)
    return markers


@pytest.fixture
def list_tcol(list_wg) -> list[TrenchColumn]:
    PARAM_TC = dotdict(
        length=1.0,
        base_folder='',
        y_min=-0.1,
        y_max=4 * 0.08 + 0.1,
        u=[30.339, 32.825],
    )
    x_c = [3, 7.5, 10.5]
    t_col = []
    for x in x_c:
        T = TrenchColumn(x_center=x, **PARAM_TC)
        T.dig_from_waveguide(flatten([list_wg]))
        t_col.append(T)
    return t_col


@pytest.fixture
def device(gc_param) -> Layer:
    return Layer(**gc_param)


def test_device_init(device, gc_param) -> None:
    assert device.unparsed_objects == []
    assert device._param == gc_param
    assert device.fig is None
    assert device.writers
    for wr in device.writers.values():
        assert wr.objs == []


def test_device_from_dict(device, gc_param) -> None:
    dev = Layer.from_dict(gc_param)
    assert dev.unparsed_objects == device.unparsed_objects
    assert dev._param == device._param
    assert dev.fig is device.fig
    assert dev.writers
    for (wr, wr_exp) in zip(dev.writers.values(), device.writers.values()):
        assert wr.objs == wr_exp.objs


@pytest.mark.parametrize(
    'inp, expectation',
    [
        ([Waveguide(), Waveguide()], does_not_raise()),
        ([Marker(), [Waveguide(), Waveguide()]], does_not_raise()),
        ([Trench(Polygon()), Waveguide()], pytest.raises(TypeError)),
        ([], does_not_raise()),
    ],
)
def test_device_parse_objects_raise(device, inp, expectation) -> None:
    with expectation:
        device.extend(inp)


def test_device_append(gc_param, list_wg, list_mk, list_tcol) -> None:
    device = Layer(**gc_param)
    for wg in list_wg:
        device.append(wg)

    assert device.writers[Waveguide]._obj_list == list_wg
    assert device.writers[TrenchColumn]._obj_list == []
    assert device.writers[Marker]._obj_list == []
    del device

    device = Layer(**gc_param)
    for wg in list_wg:
        device.append(wg)

    assert device.writers[Waveguide]._obj_list == list_wg
    assert device.writers[TrenchColumn]._obj_list == []
    assert device.writers[Marker]._obj_list == []

    for mk in list_mk:
        device.append(mk)

    assert device.writers[Waveguide]._obj_list == list_wg
    assert device.writers[TrenchColumn]._obj_list == []
    assert device.writers[Marker]._obj_list == list_mk

    for tc in list_tcol:
        device.append(tc)

    assert device.writers[Waveguide]._obj_list == list_wg
    assert device.writers[TrenchColumn]._obj_list == list_tcol
    assert device.writers[Marker]._obj_list == list_mk
    del device

    device = Layer(**gc_param)
    for wg in list_wg:
        device.append(wg)
    for mk in list_mk:
        device.append(mk)
    for tc in list_tcol:
        device.append(tc)

    assert device.writers[Waveguide]._obj_list == list_wg
    assert device.writers[TrenchColumn]._obj_list == list_tcol
    assert device.writers[Marker]._obj_list == list_mk
    del device

    device = Layer(**gc_param)
    for wg in list_wg:
        device.append(wg)
    for mk in list_mk:
        device.append(mk)
    for tc in list_tcol:
        device.append(tc)

    for tc in list_tcol:
        device.append(tc)

    assert device.writers[Waveguide]._obj_list == list_wg
    assert device.writers[TrenchColumn]._obj_list == list_tcol * 2
    assert device.writers[Marker]._obj_list == list_mk
    del device


def test_device_extend(gc_param, list_wg, list_mk, list_tcol) -> None:
    device = Layer(**gc_param)
    device.extend(list_wg)

    assert device.writers[Waveguide]._obj_list == list_wg
    assert device.writers[TrenchColumn]._obj_list == []
    assert device.writers[Marker]._obj_list == []
    del device

    device = Layer(**gc_param)
    device.extend(list_wg)

    assert device.writers[Waveguide]._obj_list == list_wg
    assert device.writers[TrenchColumn]._obj_list == []
    assert device.writers[Marker]._obj_list == []

    device.extend(list_mk)

    assert device.writers[Waveguide]._obj_list == list_wg
    assert device.writers[TrenchColumn]._obj_list == []
    assert device.writers[Marker]._obj_list == list_mk

    device.extend(list_tcol)

    assert device.writers[Waveguide]._obj_list == list_wg
    assert device.writers[TrenchColumn]._obj_list == list_tcol
    assert device.writers[Marker]._obj_list == list_mk
    del device

    device = Layer(**gc_param)
    device.extend(list_tcol)
    device.extend(list_wg)
    device.extend(list_mk)

    assert device.writers[Waveguide]._obj_list == list_wg
    assert device.writers[TrenchColumn]._obj_list == list_tcol
    assert device.writers[Marker]._obj_list == list_mk
    del device

    device = Layer(**gc_param)
    device.extend(list_tcol)
    device.extend(list_wg)
    device.extend(list_mk)
    device.extend(list_wg)
    device.extend(list_mk)
    device.extend(list_wg)
    device.extend(list_mk)

    assert device.writers[Waveguide]._obj_list == list_wg * 3
    assert device.writers[TrenchColumn]._obj_list == list_tcol
    assert device.writers[Marker]._obj_list == list_mk * 3
    del device

    device = Layer(**gc_param)
    device.extend([])

    assert device.writers[Waveguide]._obj_list == []
    assert device.writers[TrenchColumn]._obj_list == []
    assert device.writers[Marker]._obj_list == []
    del device


@pytest.mark.parametrize(
    'inp, expectation',
    [
        ([Waveguide(), Waveguide()], does_not_raise()),
        ([Marker(), [Waveguide(), Waveguide()]], does_not_raise()),
        (
            [[[Waveguide(), Waveguide()], Waveguide()], Waveguide()],
            pytest.raises(TypeError),
        ),
        ([], does_not_raise()),
        (Waveguide(), pytest.raises(TypeError)),
    ],
)
def test_device_extend_raise(device, inp, expectation) -> None:
    with expectation:
        device.extend(inp)


def test_plot2d_save(device, list_wg, list_mk, list_tcol) -> None:
    device.extend(list_wg)
    device.extend(list_mk)
    device.extend(list_tcol)

    device.plot2d(save=False)
    assert not (Path('.').cwd() / 'scheme.html').is_file()
    assert device.fig is not None


def test_plot2d_save_true(device, list_wg, list_mk, list_tcol) -> None:
    device.extend(list_wg)
    device.extend(list_mk)
    device.extend(list_tcol)

    device.plot2d(show=False, save=True)
    assert (Path('.').cwd() / 'scheme.html').is_file()
    assert device.fig is not None
    (Path('.').cwd() / 'scheme.html').unlink()


def test_plot3d_save(device, list_wg, list_mk, list_tcol) -> None:
    device.extend(list_wg)
    device.extend(list_mk)
    device.extend(list_tcol)

    device.plot3d(save=False)
    assert not (Path('.').cwd() / 'scheme.html').is_file()
    assert device.fig is not None


def test_plot3d_save_true(device, list_wg, list_mk, list_tcol) -> None:
    device.extend(list_wg)
    device.extend(list_mk)
    device.extend(list_tcol)

    device.plot3d(show=False, save=True)
    assert (Path('.').cwd() / 'scheme.html').is_file()
    assert device.fig is not None
    (Path('.').cwd() / 'scheme.html').unlink()


def test_device_pgm(device, list_wg, list_mk) -> None:

    device.extend(list_wg)
    device.extend(list_mk)
    device.pgm()
    assert (Path().cwd() / 'testCell_WG.pgm').is_file()
    assert (Path().cwd() / 'testCell_MK.pgm').is_file()
    (Path().cwd() / 'testCell_WG.pgm').unlink()
    (Path().cwd() / 'testCell_MK.pgm').unlink()


def test_device_pgm_verbose(device, list_wg, list_mk) -> None:

    device.extend(list_wg)
    device.extend(list_mk)
    device.pgm(verbose=True)
    assert (Path().cwd() / 'testCell_WG.pgm').is_file()
    assert (Path().cwd() / 'testCell_MK.pgm').is_file()
    (Path().cwd() / 'testCell_WG.pgm').unlink()
    (Path().cwd() / 'testCell_MK.pgm').unlink()


def test_device_xlsx(device, list_wg, list_mk, ss_param) -> None:
    device.extend(list_wg)
    device.extend(list_mk)
    device.xlsx(**ss_param)
    assert (Path().cwd() / 'custom_book_name.xlsx').is_file()
    (Path().cwd() / 'custom_book_name.xlsx').unlink()


def test_device_save_empty(device) -> None:
    assert device.save() is None
    assert not (Path('.').cwd() / 'scheme.html').is_file()


@pytest.mark.parametrize(
    'fn, expected',
    [
        ('test.html', 'test.html'),
        ('TEST.HTML', 'TEST.html'),
        ('test', 'test.html'),
        ('test.svg', 'test.svg'),
        ('test.pdf', 'test.pdf'),
        ('test.jpeg', 'test.jpeg'),
        ('test.png', 'test.png'),
    ],
)
def test_device_save(device, fn, expected) -> None:
    device.plot2d(show=False)
    opt = {'width': 480, 'height': 320, 'scale': 2, 'engine': 'kaleido'}
    assert device.save(fn, opt) is None
    assert (Path('.').cwd() / expected).is_file()
    (Path('.').cwd() / expected).unlink()


def test_device_export(device, list_wg, list_mk) -> None:
    device.extend(list_wg)
    device.extend(list_mk)
    device.export()
    for i, _ in enumerate(list_wg):
        fn = Path().cwd() / 'EXPORT' / 'testCell' / f'WG_{i + 1:02}.pkl'
        assert fn.is_file()
        fn.unlink()
    for i, _ in enumerate(list_mk):
        fn = Path().cwd() / 'EXPORT' / 'testCell' / f'MK_{i + 1:02}.pkl'
        assert fn.is_file()
        fn.unlink()

    (Path().cwd() / 'EXPORT' / 'testCell').rmdir()
    (Path().cwd() / 'EXPORT').rmdir()


def test_device_export_verbose(device, list_wg, list_mk) -> None:
    device.extend(list_wg)
    device.extend(list_mk)
    device.export(verbose=True)
    for i, _ in enumerate(list_wg):
        fn = Path().cwd() / 'EXPORT' / 'testCell' / f'WG_{i + 1:02}.pkl'
        assert fn.is_file()
        fn.unlink()
    for i, _ in enumerate(list_mk):
        fn = Path().cwd() / 'EXPORT' / 'testCell' / f'MK_{i + 1:02}.pkl'
        assert fn.is_file()
        fn.unlink()

    (Path().cwd() / 'EXPORT' / 'testCell').rmdir()
    (Path().cwd() / 'EXPORT').rmdir()


def test_device_load_verbose(device, list_wg, list_mk, gc_param) -> None:
    device.extend(list_wg)
    device.extend(list_mk)
    device.export(verbose=True)

    fn = Path().cwd() / 'EXPORT' / 'testCell'

    d2 = Layer.load_objects(fn, gc_param, verbose=True)
    assert d2.writers[Waveguide].objs
    assert not d2.writers[TrenchColumn].objs
    assert d2.writers[Marker].objs

    objs = []
    objs.extend(list_wg)
    objs.extend(list_mk)
    for root, dirs, files in os.walk(fn):
        for file in files:
            (pathlib.Path(root) / file).unlink()

    (Path().cwd() / 'EXPORT' / 'testCell').rmdir()
    (Path().cwd() / 'EXPORT').rmdir()


def test_device_load_empty(list_wg, list_mk, list_tcol, gc_param) -> None:

    device = Layer(**gc_param)
    device.extend(list_tcol)
    device.extend(list_wg)
    device.extend(list_mk)
    device.export(verbose=True)
    del device

    fn = Path().cwd() / 'EXPORT' / 'testCell'

    d2 = Layer.load_objects(fn, gc_param, verbose=True)
    assert d2.writers[Waveguide].objs
    assert d2.writers[TrenchColumn].objs
    assert d2.writers[Marker].objs
    # assert d2.writers[Waveguide].objs == list_wg
    # assert d2.writers[TrenchColumn].objs == list_tcol
    # assert d2.writers[Marker].objs == list_mk

    for root, dirs, files in os.walk(fn):
        for file in files:
            (pathlib.Path(root) / file).unlink()

    (Path().cwd() / 'EXPORT' / 'testCell').rmdir()
    (Path().cwd() / 'EXPORT').rmdir()
