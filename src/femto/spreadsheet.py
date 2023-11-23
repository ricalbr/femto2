from __future__ import annotations

import pathlib
from types import TracebackType
from typing import Any

import attrs
import numpy as np
import xlsxwriter
from femto import logger
from femto.helpers import flatten
from femto.helpers import listcast
from femto.marker import Marker
from femto.waveguide import Waveguide


@attrs.define(kw_only=True)
class Spreadsheet:
    """Class representing the spreadsheet with all entities to fabricate."""

    columns_names: list[str] = attrs.field(factory=list)
    description: str = ''
    book_name: str | pathlib.Path = 'FABRICATION.xlsx'
    sheet_name: str = 'Fabrication'
    font_name: str = 'DejaVu Sans Mono'
    font_size: int = 11
    redundant_cols: bool = True
    static_preamble: bool = False
    new_columns: list = attrs.field(factory=list)
    extra_preamble_info: dict = attrs.field(factory=dict)

    _workbook: xlsxwriter.Workbook = None
    _worksheet: xlsxwriter.Workbook.worksheets = None
    _all_cols: list[ColumnData] = attrs.field(alias='_all_cols', factory=list)
    _preamble_data: dict = attrs.field(alias='_preamble_data', factory=dict)
    _formats: dict = attrs.field(alias='_formats', factory=dict)

    def __attrs_post_init__(self) -> None:
        """Intitialization of the Spreadsheet object.

        Opens a new workbook with a default spreadsheet named ``Fabrication``. Creates the basic formats that will be
        used, determines the columns that really need to be used, and writed the map of the fabrication.

        Parameters
        ----------

        columns_names: str
            The columns to be written, separated by a single whitespace. The user must provide a string with tagnames
            separated by a whitespace and the tagnames must be contained in the first column of the text file
            ``columns.txt`` located in the utils folder.

        book_name: str
            Name of the Excel file, without the extension, which will be added automatically.
            Defaults to ``FABRICATION.xlsx``.

        sheet_name: str
            Name of the Excel spreadsheet. Defaults to ``Fabrication``.

        redundant_cols: bool
            If True, it will suppress all redundant columns, meaning that it will not include them in the final
            spreadsheet, even if they are in the sel_cols string. Redundant columns are columns that contain the same
            value for all of the lines (structures) in the file. Defaults to True.

        static_preamble: bool
            If True, the preamble contains always the same information. For instance, if power changes during
            fabrication, the preamble should not contain this information, and a dedicated column would appear.
            However, with static_preamble, a preamble row appears with the in formation ``variable``. Defaults to False.

        Returns
        -------
        None

        """

        if not self.columns_names:
            default_cols = ['name', 'power', 'speed', 'scan', 'radius', 'int_dist', 'depth', 'yin', 'yout', 'obs']
            self.columns_names = default_cols
            self.redundant_cols = True
            logger.debug(
                'Columns_names not given in Spreadsheet initialization. Will proceed with standard columns names '
                f'"{default_cols}" and activate the redundant_cols flag to deal with reddundant columns.'
            )

        # Prepend 'name' column as default one
        if 'name' not in self.columns_names:
            self.columns_names = ['name'] + self.columns_names

        self._workbook = xlsxwriter.Workbook(
            self.book_name,
            options={'default_format_properties': {'font_name': self.font_name, 'font_size': self.font_size}},
        )
        self._worksheet = self._workbook.add_worksheet(self.sheet_name)
        self._workbook.set_calc_mode('auto')

        # Create all the Parameters contained in the general preamble_info
        # Add them to a dictionary with the key equal to their tagname
        preamble_info: dict = {
            'General': ['laboratory', 'temperature', 'humidity', 'date', 'start', 'sample name'],
            'Substrate': ['material', 'facet', 'thickness'],
            'Laser': ['laser name', 'wl', 'duration', 'reprate', 'attenuator', 'preset'],
            'Irradiation': ['objective', 'power', 'speed', 'scan', 'depth'],
        }

        preamble_data = {}
        for k, val in preamble_info.items():
            subcat_data = {}
            for t in val:
                p = PreambleParameter(name=t)
                subcat_data[t] = p
            preamble_data[k] = subcat_data

        preamble_data = {**preamble_data, **self.extra_preamble_info}
        preamble_data['Laser']['laser name'].value = 'Pharos'
        preamble_data['General']['start'].format = 'time'
        preamble_data['General']['date'].format = 'date'

        self._preamble_data = preamble_data
        self._all_cols = self.generate_all_cols_data()
        self._formats = self._create_formats()

    def __enter__(self) -> Spreadsheet:
        """Context manager entry.

        Can be use like:
        ::
            with Spreadsheet(**SS_PARAMETERS) as ss:
                ...
        """
        self.header()
        self.fabbrication_info()
        return self

    def __exit__(
        self,
        exc_type: type(BaseException) | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.close()

    def header(self, desc_size: int = 8) -> None:

        # Set columns properties
        self._worksheet.set_column(first_col=1, last_col=1, width=15)
        self._worksheet.set_column(first_col=2, last_col=2, width=15)

        # Add the femto logo at top left of spreadsheet
        path_logo = pathlib.Path(__file__).parent / 'utils' / 'logo_excel.png'
        self._worksheet.insert_image('B2', filename=path_logo, options={'x_scale': 0.33, 'y_scale': 0.33})

        # Write the header and leave space for fabrication description
        self._worksheet.set_row(row=2, height=50)
        self._worksheet.merge_range(
            first_row=1,
            first_col=4,
            last_row=1,
            last_col=desc_size + 4,
            data='Description',
            cell_format=self._formats['title'],
        )
        self._worksheet.merge_range(
            first_row=2,
            first_col=4,
            last_row=2,
            last_col=desc_size + 4,
            data=self.description,
            cell_format=self._formats['parval'],
        )

    def fabbrication_info(self, row: int = 8) -> None:

        for pre_title, parameters in self._preamble_data.items():
            self._worksheet.merge_range(
                first_row=row,
                first_col=1,
                last_row=row,
                last_col=2,
                data=pre_title,
                cell_format=self._formats['title'],
            )
            row += 1

            for tname, p in parameters.items():
                p.set_location((row, 1))

                # Merge cell only if one of the preamble parameters have a size bigger than one cell.
                if any(p.size):
                    # Field titles
                    self._worksheet.merge_range(
                        first_row=p.row,
                        first_col=p.col,
                        last_row=p.row + p.size[0],
                        last_col=p.col + p.size[1],
                        data='',
                        cell_format=self._formats['parname'],
                    )

                    # Empty fields entry
                    self._worksheet.merge_range(
                        first_row=p.row,
                        first_col=p.col + 1,
                        last_row=p.row + p.size[0],
                        last_col=p.col + p.size[1] + 1,
                        data='',
                        cell_format=self._formats[p.format],
                    )
                row += 1

                self._add_line(row=p.row, col=p.col, data=[p.name.capitalize(), p.value], fmt=['parname', p.format])
            row += 2

    def close(self):
        """Close the workbook."""
        self._workbook.close()

    def write(self, obj_list: list[Waveguide, Marker], start: int = 5) -> None:
        """Write the structures to the spreadsheet.

        Builds the structures list, containing all the required information about the structures to fabricate. Then,
        prepares the columns in the spreadsheet, setting their width according to the ``columns.txt`` file. Finally,
        fills the spreadsheet with the contents of the structures list, one structure per line, and creates the
        preamble, which is the general information to the left of individual structure information.

        Returns
        -------
        None.

        """

        # TODO: assert data types
        cols_info, numerical_data = self._extract_data(obj_list)

        if not cols_info or numerical_data.size == 0:
            logger.debug('No column names or numerical data.')
            return

        # Set the correct width and create the data format.
        for i, col in enumerate(cols_info):
            self._worksheet.set_column(first_col=start + i, last_col=start + i, width=int(col.width))
            fmt = col.format
            if fmt not in self._formats.keys():
                self._formats[str(fmt)] = self._workbook.add_format(
                    {'align': 'center', 'valign': 'vcenter', 'num_format': str(fmt)}
                )

        # Fill SpreadSheet
        # Titles
        titles = [f'{col.name}\n[{col.unit}]' if col.unit != '' else f'{col.name}' for col in cols_info]
        self._worksheet.set_row(row=7, height=50)
        self._add_line(row=7, col=5, data=titles, fmt=len(titles) * ['title'])

        # Data
        for i, sdata in enumerate(numerical_data):
            sdata = [
                s
                if (isinstance(s, (np.int64, np.float64)) and s < 1e5) or (not isinstance(s, (np.int64, np.float64)))
                else ''
                for s in sdata
            ]
            self._add_line(row=i + 8, col=5, data=sdata, fmt=[col.format for col in cols_info])

    def generate_all_cols_data(self) -> list[ColumnData]:
        """Create the available columns array from a file.

        Gathers all data from the ``utils/spreadsheet_columns.txt`` file and creates a structured array with the
        information for all possible columns. The user can only select columns to add to the spreadsheet throught their
        tagname, which must be in the first column of the txt document.
        """

        default_cols = []
        new_cols = []
        with open(pathlib.Path(__file__).parent / 'utils' / 'spreadsheet_columns.txt') as f:
            next(f)
            for line in f:
                tag, name, unit, width, fmt = line.strip().split(', ')
                default_cols.append(ColumnData(tagname=tag, name=name, unit=unit, width=width, format=fmt))

        if self.new_columns:
            for elem in self.new_columns:
                try:
                    tag, name, unit, width, fmt = elem
                    new_cols.append(ColumnData(tagname=tag, name=name, unit=unit, width=width, format=fmt))
                except ValueError:
                    logger.error(
                        'Wrong format. Elements of new_columns should be of the type: (tag, name, unit, width, fmt).'
                    )
                    raise ValueError(
                        'Wrong format. Elements of new_columns should be of the type: (tag, name, unit, width, fmt).'
                    )

        # Merge default columns and new columns
        def_dict = dict(zip([col.tagname for col in default_cols], default_cols))
        new_dict = dict(zip([col.tagname for col in new_cols], new_cols))
        def_dict.update(new_dict)
        return list(def_dict.values())

    def _extract_data(
        self,
        structures: list[Waveguide | Marker] | None = None,
        redundant_cols: bool | None = None,
        verbose: bool = False,
    ) -> tuple[list[ColumnData], np.ndarray]:
        """Build a table with all of the structures.

        The table has as lines the several structures, and for each of them, all of the fields given as columns_names
        as column data. Determines the columns that will be effectively added to the sheet, according to the specified
        redundant_cols and static_preamble.

        Parameters
        ----------
        structures: list
            Contains the waveguides and the markers to be added to the table.

        redundant_cols: bool, optional
            If True, it will suppress all redundant columns, meaning that it will not include them in the final
            spreadsheet, even if they are in the sel_cols string. Redundant columns are columns that contain the same
            value for all of the lines (structures) in the file. Defaults to the given instation value (otherwise True).

        verbose: bool
            If True, prints the columns, selected by the user, that will be excluded from the spreadsheet because they
            are reddundant (in the case that redundant_cols is set to True).
        """

        def coords(x):
            return {
                Waveguide: {'yin': x.path3d[1][0], 'yout': x.path3d[1][-1]},
                Marker: {
                    'yin': (max(x.path3d[0][:]) + min(x.path3d[0][:])) / 2,
                    'yout': (max(x.path3d[1][:]) + min(x.path3d[1][:])) / 2,
                },
            }

        if not structures:
            return [], np.ndarray([])

        suppr_redd_cols = redundant_cols if redundant_cols is not None else self.redundant_cols
        structures = flatten(structures)

        # Select with tagname
        column_name_info = [col for col in self._all_cols if col.tagname in self.columns_names]
        dtype = [(t, self._dtype(t)) for t in self.columns_names]

        # Create data table
        table_lines = np.zeros_like(structures, dtype=dtype)

        # Extract all data from structures (either attributes of metadata)
        for i, ent in enumerate(structures):
            data_line = []
            for tag, typ in dtype:
                if tag in ['yin', 'yout']:
                    item = coords(ent)[type(ent)][tag]
                else:
                    item = getattr(ent, tag, None) or ent.metadata.get(tag)

                if item is None:
                    item = 1.1e5 if typ in [np.float64, np.int64] else ''
                data_line.append(item)
            table_lines[i] = tuple(data_line)

        # Select
        keep = []
        ignored_fields = []
        for i, t in enumerate(self.columns_names):

            if table_lines.dtype.fields[t][0].char in 'ld' and np.all(table_lines[t] > 1e5):
                ignored_fields.append(t)
                continue

            if np.all(table_lines[t] == table_lines[t][0]) and suppr_redd_cols and table_lines[t][0] != '':
                # eliminate reddundancies if explicitly requested
                ignored_fields.append(t)
            keep.append(t)

        # Add 'name' field as first default value
        if 'name' not in keep:
            keep = ['name'] + keep

        if ignored_fields and verbose:
            fields_left_out = ', '.join(ignored_fields)
            logger.debug(
                f'For all entities, the fields {fields_left_out} were not defined, they will not be shown as columns.'
            )

        info = [col for col in column_name_info if col.tagname in keep]
        return info, table_lines[keep]

    def _add_line(self, row: int, col: int, data: list[str], fmt: list[str] | None = None) -> None:
        """Add a line to the spreadsheet.

        Takes a start cell and writes a sequence of data in that and the following cells in the same line. Also
        accepts a fmt kwarg that tells the cell_fmt of the data.

        Parameters
        ----------
        row: int
            Row of the starting cell, 1-indexing.
        col: int
            Column of the starting cell, 1-indexing.
        data: str
            List of strings with the several data to write, in the order which they should appear, column-wise.
        fmt: str, optional
            list of the formatting options to be used. They must be present in self._formats, so make sure to create the
            format prior to using it. defaults to the default text properties of the spreadsheet, given at the moment of
            instantiation.

        Returns
        -------
        None.
        """

        data = listcast(data)
        if fmt is None:
            cell_fmt = [self.wb.default_format_properties]
        else:
            cell_fmt = []
            for key in listcast(fmt):
                try:
                    cell_fmt.append(self._formats[str(key)])
                except KeyError:
                    logger.error(f'Found unknown key for formatting options. Given key {key}.')
                    raise KeyError(f'Found unknown key for formatting options. Given key {key}.')

        for i, (data_val, f) in enumerate(zip(data, cell_fmt)):
            if isinstance(data_val, str) and data_val.startswith('='):
                self._worksheet.write_formula(row, col + i, data_val, f)
            else:
                self._worksheet.write(row, col + i, data_val, f)

    def _create_formats(self) -> dict[str, Any]:
        """Prepare the basic formats that will be used.

        These are the following:
            - title: used for all section titles
            - parameter name: used for the name of variables in the preamble
            - parameter value: used for the preamble variables themselves
            - time: general time format HH:MM:SS for the fabrication time
            - date: general date format DD/MM/YYYY for the fabrication date

        They are inserted into a dictionary, becoming available through the respective abbreviated key.
        """

        al = {'align': 'center', 'valign': 'vcenter', 'border': 1}
        tit_specs = {'font_color': 'white', 'bg_color': '#6C5B7B'}
        titt = dict(**{'bold': True, 'text_wrap': True}, **al)

        title_fmt = self._workbook.add_format(dict(**tit_specs, **titt))
        parname_fmt = self._workbook.add_format(dict(**{'bg_color': '#D5CABD'}, **titt))
        parval_fmt = self._workbook.add_format(dict(**{'text_wrap': True}, **al))
        text_fmt = self._workbook.add_format({'align': 'center', 'valign': 'vcenter'})
        date_fmt = self._workbook.add_format(dict(**{'num_format': 'DD/MM/YYYY'}, **al))
        time_fmt = self._workbook.add_format(dict(**{'num_format': 'HH:MM:SS'}, **al))

        return {
            'title': title_fmt,
            'parname': parname_fmt,
            'parval': parval_fmt,
            'text': text_fmt,
            'date': date_fmt,
            'time': time_fmt,
        }

    def _dtype(self, tag: str):
        """Return the data type corresponding to a give column tagname.

        The data type is determined in the ``columns.txt`` file, under the column named ``format``. The dtypes are
        assigned according to the following possibilities for that field:

        - ``text`` or ``title`` -> dtype: 20-character str
        - sequence of zeros with a dot somewhere -> dtype: float
        - sequence of zeros with no dot -> dtype: int

        Parameters
        ----------
        tag: str
            The tagname of the column type. Must be contained in the ``columns.txt`` file under the ``utils`` folder.
            Not case sensitive.

        Returns
        -------
        dt: type
            Type of the data.

        """
        tag_columns = dict(zip([col.tagname for col in self._all_cols], self._all_cols))

        if tag_columns[tag].format in 'text title':
            return 'U20'
        elif '.' in tag_columns[tag].format:
            return np.float64
        else:
            return np.int64


@attrs.define
class ColumnData:
    """Class that handles column data."""

    tagname: str
    name: str
    unit: str
    width: str
    format: str


@attrs.define
class PreambleParameter:
    """Class that handles preamble parameters."""

    name: str  #: Full name.
    value: str = ''  #: Value.
    location: tuple[int, int] = (0, 0)  #: Location (1-indexing).
    size: tuple[int, int] = (0, 0)  #: Size (for merged cells).
    format: str = 'parval'  #: Format.
    row: int | None = None
    col: int | None = None

    def __attrs_post_init__(self):
        """Set row and column, from the location with Excel 1-indexing."""
        self.row = self.location[0]
        self.col = self.location[1]

    def set_location(self, loc: tuple[int, int]):
        """Set location of a cell parameter."""
        self.location = loc
        self.row = loc[0]
        self.col = loc[1]


def main() -> None:
    import numpy as np
    from itertools import product

    from femto.helpers import dotdict
    from femto.waveguide import Waveguide

    LS = -2
    LE = 27

    # GEOMETRICAL DATA
    PARS_WG = dotdict(
        speed_closed=40,
        radius=40,
        depth=-0.860,
        pitch=0.080,
        samplesize=(25, 25),
    )

    # SPREADSHEET PARAMETERS
    PARS_SS = dotdict(
        book_name='Fabbrication.xlsx',
        columns_names=['name', 'power', 'speed', 'scan', 'depth', 'int_dist', 'yin', 'yout', 'obs'],
    )

    powers = np.linspace(600, 800, 5)
    speeds = [20, 30, 40]
    scans = [3, 5, 7]

    all_fabb = []

    for i_guide, (p, v, ns) in enumerate(product(powers, speeds, scans)):
        start_pt = [LS, 2 + i_guide * 0.08, PARS_WG.depth]
        wg = Waveguide(**PARS_WG, speed=v, scan=ns, metadata={'power': p})
        wg.start(start_pt)
        wg.linear([LE - LS, 0, 0])
        wg.end()

        all_fabb.append(wg)

    with Spreadsheet(**PARS_SS) as S:
        S.write(all_fabb)


if __name__ == '__main__':
    main()
