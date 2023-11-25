from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from itertools import product
from pathlib import Path

import numpy as np
import openpyxl
import pytest
from femto import __file__ as fpath
from femto.curves import sin
from femto.marker import Marker
from femto.spreadsheet import Spreadsheet
from femto.waveguide import Waveguide

src_path = Path(fpath).parent
dot_path = Path('.').cwd()


@pytest.fixture
def all_cols():
    all_cols = np.genfromtxt(
        src_path / 'utils' / 'spreadsheet_columns.txt',
        delimiter=', ',
        dtype=[('tagname', 'U20'), ('fullname', 'U20'), ('unit', 'U20'), ('width', int), ('format', 'U20')],
    )
    return all_cols


@pytest.fixture
def list_wg() -> list[Waveguide]:
    PARAM_WG = dict(speed=20, radius=25, pitch=0.080, int_dist=0.007, samplesize=(25, 3))

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
    PARAM_MK = dict(scan=1, speed=2, speed_pos=5, speed_closed=5, depth=0.000, lx=1, ly=1)
    markers = []
    for (x, y) in zip(range(4, 8), range(3, 7)):
        m = Marker(**PARAM_MK)
        m.cross([x, y])
        markers.append(m)
    return markers


@pytest.fixture
def ss_param():
    return dict(
        book_name='custom_book_name.xlsx',
        sheet_name='custom_sheet_name',
        columns_names=['name', 'power', 'speed', 'scan', 'depth', 'int_dist', 'yin', 'yout', 'obs'],
    )


@pytest.fixture
def wg_param():
    return dict(
        speed_closed=40,
        radius=40,
        depth=-0.860,
        pitch=0.080,
        samplesize=(25, 25),
    )


@pytest.fixture
def gc_param():
    return dict(
        filename='test_program.pgm',
        laser='PHAROS',
        n_glass=1.4625,
        n_environment=1.000,
        samplesize=(25, 25),
    )


# def device_redd_cols(redd_cols, non_redd_cols, gc_param):
#
#     # redd_cols: the reddundant columns, so all guides have this same attribute
#     dev = Device(**gc_param)
#     ints = iter(range(2, 500))
#     for i_guide in range(10):
#
#         non_rep_value = next(ints)
#
#         start_pt = [-2, 2 + i_guide * 0.08, 0]
#         wg = Waveguide()
#
#         for attribute in redd_cols:
#             setattr(wg, attribute, 50)  # set them all to the same value
#
#         for attribute in non_redd_cols:
#             setattr(wg, attribute, non_rep_value)
#
#         wg.start(start_pt)
#         wg.linear([27, 0, 0])
#         wg.end()
#
#         dev.append(wg)
#
#     return dev


def test_spsh_init_default():
    S = Spreadsheet()
    assert S.columns_names == ['name', 'power', 'speed', 'scan', 'radius', 'int_dist', 'depth', 'yin', 'yout', 'obs']
    assert S.description == ''
    assert S.book_name == 'FABRICATION.xlsx'
    assert S.sheet_name == 'Fabrication'
    assert S.font_name == 'DejaVu Sans Mono'
    assert S.font_size == 11
    assert S.redundant_cols is False
    assert S.new_columns == []
    assert S.metadata == {}

    assert S._workbook.default_format_properties['font_name'] == 'DejaVu Sans Mono'
    assert S._workbook.default_format_properties['font_size'] == 11
    assert S._workbook.filename == 'FABRICATION.xlsx'
    assert S._worksheet.name == 'Fabrication'


def test_extra_preamble_info(all_cols, ss_param):
    extra_preamble_info = {
        'wavelength': '1030 nm',
        'laboratory': 'CAPABLE',
        'reprate': '1 MHz',
        'material': 'EAGLE XG',
        'facet': 'bottom',
        'thickness': '900 um',
        'preset': '4',
        'attenuator': '33 %',
        'objective': '1000x',
        'samplename': 'My sample 01',
    }

    ss_param['metadata'] = extra_preamble_info
    ss_param['book_name'] = 'extra_preamble_info.xlsx'

    with Spreadsheet(**ss_param) as S:
        S.write([])

    wb = openpyxl.load_workbook(dot_path / 'extra_preamble_info.xlsx')
    worksheet = wb['custom_sheet_name']

    for k, v in extra_preamble_info.items():
        found_extra_par = False
        for row in range(1, 50):
            if worksheet.cell(row=row, column=2).value == k.capitalize().replace('_', ' '):
                found_extra_par = True
                assert worksheet.cell(row=row, column=3).value == v
                break
        assert found_extra_par

    (dot_path / 'extra_preamble_info.xlsx').unlink()


# def test_add_custom_column(ss_param):
#
#     rep_rates = [20, 100, 1000]
#     powers = np.linspace(100, 500, 5)
#     scan_offsets = np.linspace(0.000100, 0.000500, 5)
#
#     dev = Device(filename='test_program.pgm', laser='PHAROS')
#
#     wg_param = dict(
#         speed_closed=40,
#         radius=40,
#         speed=2,
#         scan=10,
#         depth=-0.100,
#         samplesize=(25, 25),
#     )
#
#     for i_guide, (rr, p, so) in enumerate(product(rep_rates, powers, scan_offsets)):
#
#         start_pt = [-2, 2 + i_guide * 0.08, wg_param['depth']]
#
#         wg = Waveguide(**wg_param)
#         wg.power = p  # Can NOT be added inside of the arguments of Waveguide
#         wg.reprate = rr
#         wg.sco = so
#
#         wg.start(start_pt)
#         wg.linear([27, 0, 0])
#         wg.end()
#
#         dev.append(wg)
#
#     ss_param.book_name = 'test_new_column.xlsx'
#     ss_param.columns_names = 'name power reprate sco yin yout obs'
#     ss_param.new_columns = [
#         ('sco', 'Scan offset', 'um', 7, '0.0000'),
#         ('reprate', 'Rep. rate', 'kHz', 7, '0'),
#     ]
#     dev.xlsx(**ss_param)
#
#     wb = openpyxl.load_workbook(dot_path / 'test_new_column.xlsx')
#     worksheet = wb['custom_sheet_name']
#
#     assert worksheet.cell(row=8, column=9).value == 'Scan offset / um'
#     assert worksheet.cell(row=8, column=8).value == 'Rep. rate / kHz'
#
#     for row in range(1, 20):
#         assert worksheet.cell(row=row, column=2).value != 'Rep. rate / kHz'
#         assert worksheet.cell(row=row, column=2).value != 'Rep. rate / MHz'
#
#     (dot_path / 'test_new_column.xlsx').unlink()


def test_write_header(ss_param):
    with Spreadsheet(**ss_param) as S:
        print(S)

    wb = openpyxl.load_workbook(dot_path / 'custom_book_name.xlsx')
    worksheet = wb['custom_sheet_name']

    assert worksheet.cell(row=1, column=4).value == 'Description'
    (dot_path / 'custom_book_name.xlsx').unlink()


def test_write_preamble_default(ss_param):
    del ss_param['columns_names']
    with Spreadsheet(**ss_param) as S:
        print(S)

    wb = openpyxl.load_workbook(dot_path / 'custom_book_name.xlsx')
    worksheet = wb['custom_sheet_name']

    assert worksheet.cell(row=9, column=2).value == 'General'
    assert worksheet.cell(row=10, column=2).value == 'Laboratory'
    assert worksheet.cell(row=11, column=2).value == 'Temperature'
    assert worksheet.cell(row=12, column=2).value == 'Humidity'
    assert worksheet.cell(row=13, column=2).value == 'Date'
    assert worksheet.cell(row=14, column=2).value == 'Start'
    assert worksheet.cell(row=15, column=2).value == 'Samplename'

    assert worksheet.cell(row=18, column=2).value == 'Substrate'
    assert worksheet.cell(row=19, column=2).value == 'Material'
    assert worksheet.cell(row=20, column=2).value == 'Facet'
    assert worksheet.cell(row=21, column=2).value == 'Thickness'

    assert worksheet.cell(row=24, column=2).value == 'Laser'
    assert worksheet.cell(row=25, column=2).value == 'Lasername'
    assert worksheet.cell(row=26, column=2).value == 'Wavelength'
    assert worksheet.cell(row=27, column=2).value == 'Duration'
    assert worksheet.cell(row=28, column=2).value == 'Reprate'
    assert worksheet.cell(row=29, column=2).value == 'Attenuator'
    assert worksheet.cell(row=30, column=2).value == 'Preset'

    assert worksheet.cell(row=33, column=2).value == 'Irradiation'
    assert worksheet.cell(row=34, column=2).value == 'Objective'
    assert worksheet.cell(row=35, column=2).value == 'Power'
    assert worksheet.cell(row=36, column=2).value == 'Speed'
    assert worksheet.cell(row=37, column=2).value == 'Scan'
    assert worksheet.cell(row=38, column=2).value == 'Depth'

    for i in range(8, 39):
        assert worksheet.cell(row=i, column=3).value is None
    (dot_path / 'custom_book_name.xlsx').unlink()


# @pytest.mark.parametrize(
#     'cols',
#     [
#         'power speed scan radius int_dist, ',
#         'power, speed scan radius int_dist',
#         'power scan, speed radius int_dist',
#         'radius, power speed scan',
#     ],
# )
# def test_redd_cols(cols, ss_param, gc_param):
#
#     non_redd_cols, redd_cols = cols.split(', ')
#     non_redd_cols = non_redd_cols.split()
#     redd_cols = redd_cols.split()
#
#     d = device_redd_cols(redd_cols, non_redd_cols, gc_param)
#
#     ss_param.columns_names = cols.replace(',', '').strip()
#     ss_param.suppr_redd_cols = True
#     ss_param.device = d
#
#     spsh = Spreadsheet(**ss_param)
#     spsh._build_struct_list()
#     spsh.close()
#
#     columns_tnames = list(spsh.columns_data['tagname'])
#     columns_tnames.remove('name')
#
#     assert all([tn in non_redd_cols for tn in columns_tnames])
#     assert all([tn not in redd_cols for tn in columns_tnames])
#
#     (dot_path / 'custom_book_name.xlsx').unlink()


def test_create_structures(list_wg, gc_param, ss_param):
    with Spreadsheet(**ss_param) as spsh:
        spsh.write(list_wg)

    assert (dot_path / 'custom_book_name.xlsx').is_file()
    (dot_path / 'custom_book_name.xlsx').unlink()


@pytest.mark.parametrize(
    'init_dev, bsl_dev',
    list(product(*2 * [[True, False]])),
)
def test_device_init(device, ss_param, init_dev, bsl_dev):

    ss_pars = ss_param
    bsl_pars = {}

    if init_dev:
        ss_pars['device'] = device

    if bsl_dev:
        bsl_pars['structures'] = device.writers[Waveguide]._obj_list

    exp = does_not_raise() if init_dev else pytest.raises(TypeError)
    with exp:
        spsh = Spreadsheet(**ss_pars)
        spsh._build_struct_list()
        spsh.close()

    if (dot_path / 'custom_book_name.xlsx').exists():
        (dot_path / 'custom_book_name.xlsx').unlink()


@pytest.mark.parametrize('verbose', [True, False])
def test_build_structure_list(empty_device, list_wg, list_mk, ss_param, verbose):
    empty_device.extend(list_wg)

    obj_list = empty_device.writers[Waveguide]._obj_list

    spsh = Spreadsheet(device=empty_device, **ss_param)
    # Use the defaults for suppressing reddundant olumns and static preamble
    spsh._build_struct_list(obj_list, ss_param['columns_names'], verbose=verbose)
    spsh.close()

    assert (dot_path / 'custom_book_name.xlsx').is_file()
    assert len(spsh.struct_data) == len(obj_list)

    (dot_path / 'custom_book_name.xlsx').unlink()


@pytest.mark.parametrize('structures', [[], [[]], [[[]]], [[[[]]]]])
def test_write_empty(ss_param, structures):
    with Spreadsheet(**ss_param) as S:
        info, numdata = S._extract_data(structures)
    assert info == []
    assert numdata.size == 0
