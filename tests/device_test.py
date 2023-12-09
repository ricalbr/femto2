from __future__ import annotations

import os
import pathlib
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest
from femto.curves import sin
from femto.device import Cell
from femto.device import Device
from femto.helpers import dotdict
from femto.helpers import flatten
from femto.marker import Marker
from femto.trench import Trench
from femto.trench import TrenchColumn
from femto.trench import UTrenchColumn
from femto.waveguide import NasuWaveguide
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
def device(gc_param) -> Device:
    return Device(**gc_param)


@pytest.fixture
def cell() -> Cell:
    return Cell()


# Cell
def test_cell_init(cell) -> None:
    assert cell.name == 'base'
    assert cell.description is None
    assert type(cell._objs) == dict
    assert cell._objs == {
        TrenchColumn: [],
        UTrenchColumn: [],
        Marker: [],
        Waveguide: [],
        NasuWaveguide: [],
    }


@pytest.mark.parametrize(
    'name, exp',
    [
        ('base', 'base'),
        ('base test', 'base-test'),
        ('TEST-TEST', 'test-test'),
        ('test_test', 'test_test'),
        ('A B C', 'a-b-c'),
        ('ab C', 'ab-c'),
    ],
)
def test_cell_rename(name, exp) -> None:
    cell = Cell(name=name)
    assert cell.name == exp


@pytest.mark.parametrize(
    'name, exp',
    [
        ('base', 'base'),
        ('base test', 'base-test'),
        ('TEST-TEST', 'test-test'),
        ('test_test', 'test_test'),
        ('A B C', 'a-b-c'),
        ('ab C', 'ab-c'),
    ],
)
def test_cell_repr(name, exp) -> None:
    cell = Cell(name=name)
    assert cell.__repr__() == f'Cell {exp}'


def test_cell_objects_empty(cell) -> None:
    assert type(cell.objects) == dict
    assert cell.objects == {
        TrenchColumn: [],
        UTrenchColumn: [],
        Marker: [],
        Waveguide: [],
        NasuWaveguide: [],
    }


def test_cell_objects(cell, list_wg, list_mk) -> None:
    cell.add(list_wg)
    cell.add(list_mk)
    cell.add([list_wg, list_mk])
    obj = cell.objects
    assert obj[UTrenchColumn] == []
    assert obj[TrenchColumn] == []
    assert obj[NasuWaveguide] == []
    assert obj[Waveguide] == flatten([list_wg, list_wg])
    assert obj[Marker] == flatten([list_mk, list_mk])


@pytest.mark.parametrize(
    'inp, expectation',
    [
        ([Waveguide(), Waveguide()], does_not_raise()),
        ([Marker(), [Waveguide(), Waveguide()]], does_not_raise()),
        ([Trench(Polygon()), Waveguide()], pytest.raises(TypeError)),
        ([], does_not_raise()),
    ],
)
def test_cell_parse_objects_raise(cell, inp, expectation) -> None:
    with expectation:
        cell.add(inp)


def test_cell_add(list_wg, list_mk, list_tcol) -> None:
    cell = Cell()
    cell.add(list_wg)
    assert cell.objects[Waveguide] == list_wg
    assert cell.objects[TrenchColumn] == []
    assert cell.objects[Marker] == []
    assert cell.objects[UTrenchColumn] == []
    assert cell.objects[TrenchColumn] == []
    assert cell.objects[NasuWaveguide] == []
    del cell

    cell = Cell()
    cell.add(list_wg)
    assert cell.objects[Waveguide] == list_wg
    assert cell.objects[TrenchColumn] == []
    assert cell.objects[Marker] == []
    assert cell.objects[UTrenchColumn] == []
    assert cell.objects[NasuWaveguide] == []

    cell.add(list_mk)
    assert cell.objects[Waveguide] == list_wg
    assert cell.objects[TrenchColumn] == []
    assert cell.objects[Marker] == list_mk
    assert cell.objects[UTrenchColumn] == []
    assert cell.objects[NasuWaveguide] == []

    cell.add(list_tcol)
    assert cell.objects[Waveguide] == list_wg
    assert cell.objects[TrenchColumn] == list_tcol
    assert cell.objects[Marker] == list_mk
    assert cell.objects[UTrenchColumn] == []
    assert cell.objects[NasuWaveguide] == []
    del cell

    cell = Cell()
    cell.add(list_wg)
    cell.add(list_mk)
    cell.add(list_tcol)
    assert cell.objects[Waveguide] == list_wg
    assert cell.objects[TrenchColumn] == list_tcol
    assert cell.objects[Marker] == list_mk
    assert cell.objects[UTrenchColumn] == []
    assert cell.objects[NasuWaveguide] == []
    del cell

    cell = Cell()
    cell.add(list_wg)
    cell.add(list_mk)
    cell.add(list_wg)
    cell.add(list_tcol)
    cell.add(list_tcol)
    cell.add(list_wg)
    assert cell.objects[Waveguide] == list_wg * 3
    assert cell.objects[TrenchColumn] == list_tcol * 2
    assert cell.objects[Marker] == list_mk
    assert cell.objects[UTrenchColumn] == []
    assert cell.objects[NasuWaveguide] == []
    del cell


# Device
def test_device_init(device, gc_param) -> None:
    assert device.cells == {}
    assert device.fig is None
    assert device.fabrication_time == 0.0

    assert device._param == gc_param
    assert device._print_angle_warning is True
    assert device._print_base_cell_warning is True


def test_device_from_dict(device, gc_param) -> None:
    dev = Device.from_dict(gc_param)
    assert dev.cells == device.cells
    assert dev._param == device._param
    assert dev._param == gc_param
    assert dev.fig is device.fig
    assert dev.fabrication_time == dev.fabrication_time


def test_device_add_cell(device, cell, list_tcol, list_wg) -> None:
    c2 = Cell(name='new')
    c2.add(list_wg)
    device.add(list_tcol)

    device.add_cell(c2)
    assert len(device.cells.values()) == 2
    assert all(keys in ['base', 'new'] for keys in device.cells.keys())

    assert device.cells['base'].objects[Waveguide] == []
    assert device.cells['base'].objects[NasuWaveguide] == []
    assert device.cells['base'].objects[Marker] == []
    assert device.cells['base'].objects[TrenchColumn] == list_tcol
    assert device.cells['base'].objects[UTrenchColumn] == []

    assert device.cells['new'].objects[Waveguide] == list_wg
    assert device.cells['new'].objects[NasuWaveguide] == []
    assert device.cells['new'].objects[Marker] == []
    assert device.cells['new'].objects[TrenchColumn] == []
    assert device.cells['new'].objects[UTrenchColumn] == []


def test_device_add_to_cell(device, cell, list_wg, list_mk) -> None:
    c2 = Cell(name='new')
    c2.add(list_wg)
    device.add(c2)

    assert device.cells['new'].objects[Waveguide] == list_wg
    assert device.cells['new'].objects[NasuWaveguide] == []
    assert device.cells['new'].objects[Marker] == []
    assert device.cells['new'].objects[TrenchColumn] == []
    assert device.cells['new'].objects[UTrenchColumn] == []

    device.add_to_cell(key=c2.name, obj=list_mk)
    assert len(device.cells.values()) == 1
    assert list(device.cells.keys()) == ['new']

    assert device.cells['new'].objects[Waveguide] == list_wg
    assert device.cells['new'].objects[NasuWaveguide] == []
    assert device.cells['new'].objects[Marker] == list_mk
    assert device.cells['new'].objects[TrenchColumn] == []
    assert device.cells['new'].objects[UTrenchColumn] == []


def test_device_add_to_non_existing_cell(device, cell, list_wg, list_mk) -> None:
    c2 = Cell(name='new')
    c2.add(list_wg)
    device.add(c2)

    assert device.cells['new'].objects[Waveguide] == list_wg
    assert device.cells['new'].objects[NasuWaveguide] == []
    assert device.cells['new'].objects[Marker] == []
    assert device.cells['new'].objects[TrenchColumn] == []
    assert device.cells['new'].objects[UTrenchColumn] == []

    device.add_to_cell(key='new2', obj=list_mk)
    assert len(device.cells.values()) == 2
    assert list(device.cells.keys()) == ['new', 'new2']

    assert device.cells['new2'].objects[Waveguide] == []
    assert device.cells['new2'].objects[NasuWaveguide] == []
    assert device.cells['new2'].objects[Marker] == list_mk
    assert device.cells['new2'].objects[TrenchColumn] == []
    assert device.cells['new2'].objects[UTrenchColumn] == []


def test_device_add_list_femtobjs(device, list_wg, list_mk, list_tcol) -> None:
    listone = flatten([list_wg, list_mk, list_tcol])
    device.add(listone)

    assert device.cells['base'].objects[Waveguide] == list_wg
    assert device.cells['base'].objects[NasuWaveguide] == []
    assert device.cells['base'].objects[Marker] == list_mk
    assert device.cells['base'].objects[TrenchColumn] == list_tcol
    assert device.cells['base'].objects[UTrenchColumn] == []


def test_device_add_list_femtobjs_multi(device, list_wg, list_mk, list_tcol) -> None:
    listone = flatten([list_wg, list_mk, list_tcol, list_wg, list_wg])
    device.add(listone)

    assert device.cells['base'].objects[Waveguide] == list_wg * 3
    assert device.cells['base'].objects[NasuWaveguide] == []
    assert device.cells['base'].objects[Marker] == list_mk
    assert device.cells['base'].objects[TrenchColumn] == list_tcol
    assert device.cells['base'].objects[UTrenchColumn] == []


def test_device_add_single_femtobjs(device, list_wg, list_mk, list_tcol) -> None:
    wg, mk, tc = list_wg[0], list_mk[0], list_tcol[0]
    device.add(wg)
    device.add(mk)
    device.add(tc)

    assert device.cells['base'].objects[Waveguide] == [wg]
    assert device.cells['base'].objects[NasuWaveguide] == []
    assert device.cells['base'].objects[Marker] == [mk]
    assert device.cells['base'].objects[TrenchColumn] == [tc]
    assert device.cells['base'].objects[UTrenchColumn] == []


def test_device_add_single_cell(device) -> None:
    cell = Cell(name='test')
    device.add(cell)

    assert len(device.cells.values()) == 1
    assert list(device.cells.keys()) == ['test']
    assert device.cells['test'].objects[Waveguide] == []
    assert device.cells['test'].objects[NasuWaveguide] == []
    assert device.cells['test'].objects[Marker] == []
    assert device.cells['test'].objects[TrenchColumn] == []
    assert device.cells['test'].objects[UTrenchColumn] == []


def test_device_add_single_cell_multi(device) -> None:
    cell = Cell(name='test')
    device.add(cell)
    cell = Cell(name='test1')
    device.add(cell)
    cell = Cell(name='test2')
    device.add(cell)

    assert len(device.cells.values()) == 3
    assert list(device.cells.keys()) == ['test', 'test1', 'test2']
    assert device.cells['test'].objects[Waveguide] == []
    assert device.cells['test'].objects[NasuWaveguide] == []
    assert device.cells['test'].objects[Marker] == []
    assert device.cells['test'].objects[TrenchColumn] == []
    assert device.cells['test'].objects[UTrenchColumn] == []

    assert device.cells['test1'].objects[Waveguide] == []
    assert device.cells['test1'].objects[NasuWaveguide] == []
    assert device.cells['test1'].objects[Marker] == []
    assert device.cells['test1'].objects[TrenchColumn] == []
    assert device.cells['test1'].objects[UTrenchColumn] == []

    assert device.cells['test2'].objects[Waveguide] == []
    assert device.cells['test2'].objects[NasuWaveguide] == []
    assert device.cells['test2'].objects[Marker] == []
    assert device.cells['test2'].objects[TrenchColumn] == []
    assert device.cells['test2'].objects[UTrenchColumn] == []


def test_device_add_list_cell_multi(device) -> None:
    cell1 = Cell(name='test')
    cell2 = Cell(name='test1')
    cell3 = Cell(name='test2')
    device.add([cell1, cell2, cell3])

    assert len(device.cells.values()) == 3
    assert list(device.cells.keys()) == ['test', 'test1', 'test2']
    assert device.cells['test'].objects[Waveguide] == []
    assert device.cells['test'].objects[NasuWaveguide] == []
    assert device.cells['test'].objects[Marker] == []
    assert device.cells['test'].objects[TrenchColumn] == []
    assert device.cells['test'].objects[UTrenchColumn] == []

    assert device.cells['test1'].objects[Waveguide] == []
    assert device.cells['test1'].objects[NasuWaveguide] == []
    assert device.cells['test1'].objects[Marker] == []
    assert device.cells['test1'].objects[TrenchColumn] == []
    assert device.cells['test1'].objects[UTrenchColumn] == []

    assert device.cells['test2'].objects[Waveguide] == []
    assert device.cells['test2'].objects[NasuWaveguide] == []
    assert device.cells['test2'].objects[Marker] == []
    assert device.cells['test2'].objects[TrenchColumn] == []
    assert device.cells['test2'].objects[UTrenchColumn] == []


@pytest.mark.parametrize(
    'elem, exp',
    [
        ([Waveguide(), 1, 2, 3], pytest.raises(ValueError)),
        ([Waveguide(), Marker(), Marker()], does_not_raise()),
        ([[[Waveguide(), Marker(), Marker()]]], does_not_raise()),
        (NasuWaveguide(), does_not_raise()),
        ([NasuWaveguide(), Waveguide()], does_not_raise()),
        ([Waveguide()], does_not_raise()),
        ([None], pytest.raises(ValueError)),
        (None, pytest.raises(ValueError)),
    ],
)
def test_device_add_raise(device, elem, exp) -> None:
    with exp:
        device.add(elem)


def test_plot2d_save(device, list_wg, list_mk, list_tcol) -> None:
    device.add([list_wg, list_mk, list_tcol])

    device.plot2d(save=False)
    assert not (Path('.').cwd() / 'scheme.html').is_file()
    assert device.fig is not None


def test_plot2d_save_true(device, list_wg, list_mk, list_tcol) -> None:
    device.add([list_wg, list_mk, list_tcol])

    device.plot2d(show=False, save=True)
    assert (Path('.').cwd() / 'scheme.html').is_file()
    assert device.fig is not None
    (Path('.').cwd() / 'scheme.html').unlink()


def test_plot3d_save(device, list_wg, list_mk, list_tcol) -> None:
    device.add([list_wg, list_mk, list_tcol])

    device.plot3d(save=False)
    assert not (Path('.').cwd() / 'scheme.html').is_file()
    assert device.fig is not None


def test_plot3d_save_true(device, list_wg, list_mk, list_tcol) -> None:
    device.add([list_wg, list_mk, list_tcol])

    device.plot3d(show=False, save=True)
    assert (Path('.').cwd() / 'scheme.html').is_file()
    assert device.fig is not None
    (Path('.').cwd() / 'scheme.html').unlink()


def test_device_pgm(device, list_wg, list_mk) -> None:
    device.add([list_wg, list_mk])
    device.pgm()
    assert (Path().cwd() / 'testCell_WG.pgm').is_file()
    assert (Path().cwd() / 'testCell_MK.pgm').is_file()
    (Path().cwd() / 'testCell_WG.pgm').unlink()
    (Path().cwd() / 'testCell_MK.pgm').unlink()


def test_device_pgm_verbose(device, list_wg, list_mk) -> None:

    device.add([list_wg, list_mk])
    device.pgm(verbose=True)
    assert (Path().cwd() / 'testCell_WG.pgm').is_file()
    assert (Path().cwd() / 'testCell_MK.pgm').is_file()
    (Path().cwd() / 'testCell_WG.pgm').unlink()
    (Path().cwd() / 'testCell_MK.pgm').unlink()


def test_device_xlsx(device, list_wg, list_mk, ss_param) -> None:
    device.add([list_wg, list_mk])
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
    device.add([list_wg, list_mk])
    device.export()
    for i, _ in enumerate(list_wg):
        fn = Path().cwd() / 'EXPORT' / 'base' / f'WG_{i + 1:02}.pkl'
        assert fn.is_file()
        fn.unlink()
    for i, _ in enumerate(list_mk):
        fn = Path().cwd() / 'EXPORT' / 'base' / f'MK_{i + 1:02}.pkl'
        assert fn.is_file()
        fn.unlink()

    (Path().cwd() / 'EXPORT' / 'base').rmdir()
    (Path().cwd() / 'EXPORT').rmdir()


def test_device_export_verbose(device, list_wg, list_mk) -> None:
    device.add([list_wg, list_mk])
    device.export()
    for i, _ in enumerate(list_wg):
        fn = Path().cwd() / 'EXPORT' / 'base' / f'WG_{i + 1:02}.pkl'
        assert fn.is_file()
        fn.unlink()
    for i, _ in enumerate(list_mk):
        fn = Path().cwd() / 'EXPORT' / 'base' / f'MK_{i + 1:02}.pkl'
        assert fn.is_file()
        fn.unlink()

    (Path().cwd() / 'EXPORT' / 'base').rmdir()
    (Path().cwd() / 'EXPORT').rmdir()


def test_device_load_verbose(device, list_wg, list_mk, gc_param) -> None:
    device.add([list_wg, list_mk])
    device.export()

    fn = Path().cwd() / 'EXPORT' / 'base'

    d2 = Device.load_objects(fn, gc_param, verbose=True)
    assert d2.cells['base'].objects[Waveguide]
    assert d2.cells['base'].objects[Marker]
    assert not d2.cells['base'].objects[TrenchColumn]
    assert not d2.cells['base'].objects[UTrenchColumn]
    assert not d2.cells['base'].objects[NasuWaveguide]

    objs = []
    objs.extend(list_wg)
    objs.extend(list_mk)
    for root, dirs, files in os.walk(fn):
        for file in files:
            (pathlib.Path(root) / file).unlink()

    (Path().cwd() / 'EXPORT' / 'base').rmdir()
    (Path().cwd() / 'EXPORT').rmdir()


def test_device_load_empty(list_wg, list_mk, list_tcol, gc_param) -> None:

    device = Device(**gc_param)
    device.add([list_tcol, list_wg, list_mk])
    device.export()
    del device

    fn = Path().cwd() / 'EXPORT' / 'base'

    d2 = Device.load_objects(fn, gc_param, verbose=True)
    assert d2.cells['base'].objects[Waveguide]
    assert d2.cells['base'].objects[Marker]
    assert d2.cells['base'].objects[TrenchColumn]
    assert not d2.cells['base'].objects[UTrenchColumn]
    assert not d2.cells['base'].objects[NasuWaveguide]
    # assert d2.writers[Waveguide].objs == list_wg
    # assert d2.writers[TrenchColumn].objs == list_tcol
    # assert d2.writers[Marker].objs == list_mk

    for root, dirs, files in os.walk(fn):
        for file in files:
            (pathlib.Path(root) / file).unlink()

    (Path().cwd() / 'EXPORT' / 'base').rmdir()
    (Path().cwd() / 'EXPORT').rmdir()


def test_device_export_non_base_cell(device, list_wg, list_mk) -> None:
    cell = Cell(name='test')
    cell.add([list_wg, list_mk])
    device.add(cell)
    device.export()
    for i, _ in enumerate(list_wg):
        fn = Path().cwd() / 'EXPORT' / 'test' / f'WG_{i + 1:02}.pkl'
        assert fn.is_file()
        fn.unlink()
    for i, _ in enumerate(list_mk):
        fn = Path().cwd() / 'EXPORT' / 'test' / f'MK_{i + 1:02}.pkl'
        assert fn.is_file()
        fn.unlink()

    (Path().cwd() / 'EXPORT' / 'test').rmdir()
    (Path().cwd() / 'EXPORT').rmdir()


def test_device_load_non_base_cell(device, gc_param, list_wg, list_mk, list_tcol) -> None:
    cell = Cell(name='test')
    cell.add([list_wg, list_mk, list_tcol])
    device.add(cell)
    device.export()
    del device

    fn = Path().cwd() / 'EXPORT' / 'test'

    d2 = Device.load_objects(fn, gc_param, verbose=True)
    assert d2.cells['test'].objects[Waveguide]
    assert d2.cells['test'].objects[Marker]
    assert d2.cells['test'].objects[TrenchColumn]
    assert not d2.cells['test'].objects[UTrenchColumn]
    assert not d2.cells['test'].objects[NasuWaveguide]
    # assert d2.writers[Waveguide].objs == list_wg
    # assert d2.writers[TrenchColumn].objs == list_tcol
    # assert d2.writers[Marker].objs == list_mk

    for root, dirs, files in os.walk(fn):
        for file in files:
            (pathlib.Path(root) / file).unlink()

    (Path().cwd() / 'EXPORT' / 'test').rmdir()
    (Path().cwd() / 'EXPORT').rmdir()
