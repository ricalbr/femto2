from __future__ import annotations

import collections
import copy
import pathlib
from typing import Any
from typing import get_args
from typing import Optional
from typing import Union

import attrs
import dill
import plotly.graph_objects as go
from femto import logger
from femto.curves import sin
from femto.helpers import flatten
from femto.helpers import walklevel
from femto.laserpath import LaserPath
from femto.marker import Marker
from femto.spreadsheet import Spreadsheet
from femto.trench import Trench
from femto.trench import TrenchColumn
from femto.trench import UTrenchColumn
from femto.waveguide import NasuWaveguide
from femto.waveguide import Waveguide
from femto.writer import MarkerWriter
from femto.writer import NasuWriter
from femto.writer import plot2d_base_layer
from femto.writer import plot3d_base_layer
from femto.writer import TrenchWriter
from femto.writer import UTrenchWriter
from femto.writer import WaveguideWriter


# List of femto objects
types = dict(
    WG=Waveguide,
    NWG=NasuWaveguide,
    MK=Marker,
    LP=LaserPath,
    TR=Trench,
    TC=TrenchColumn,
    UTC=UTrenchColumn,
)
femtobj = Union[Waveguide, NasuWaveguide, Marker, Trench, TrenchColumn, UTrenchColumn]

writers = {
    TrenchColumn: TrenchWriter,
    UTrenchColumn: UTrenchWriter,
    Marker: MarkerWriter,
    Waveguide: WaveguideWriter,
    NasuWaveguide: NasuWriter,
}


@attrs.define(kw_only=True, repr=False)
class Cell:
    """Cell object."""

    name: str = 'base'
    description: str | None = None

    _objs: dict[type[femtobj], list[femtobj]] = attrs.field(alias='_objs', factory=dict)

    def __attrs_post_init__(self) -> None:
        self._objs = {
            TrenchColumn: [],
            UTrenchColumn: [],
            Marker: [],
            Waveguide: [],
            NasuWaveguide: [],
        }
        self.name = self.name.lower()
        if ' ' in self.name:
            self.name = self.name.replace(' ', '-')
        logger.info(f'Cell {self.name.replace("-", " ").upper()}.')

    def __repr__(self) -> str:
        return f'Cell {self.name}'

    @property
    def objects(self) -> dict[type[Any], list[Any]]:
        return self._objs

    def parse_objects(self, unparsed_objects: Any | list[Any]) -> None:
        """Parse objects.

        The function takes a list of objects and parse all of them based on their types.
        If the type of the object matches one of the types of the ``Writer`` registered in the ``Cell`` class,
        the object is added to the ``Writer._obj_list``. If not, a ``TypeError`` is raised.

        Parameters
        ----------
        unparsed_objects : list
            List of object that can be stored in a ``Cell``.

        Returns
        -------
        None
        """

        # split the unparsed_object list based on the type of each element
        unparsed_objects = flatten([unparsed_objects])
        d = collections.defaultdict(list)
        for o in unparsed_objects:
            d[type(o)].append(o)

        # add each element to the type-matching keys in self._objs dictionary
        for k, e in d.items():
            try:
                logger.debug(f'Assign {e} to {self._objs[k]}.')
                self._objs[k].extend(e)
            except KeyError as err:
                logger.error(f'Found unexpected type {err.args}.')
                raise TypeError(f'Found unexpected type {err.args}.')

    def add(self, objs: femtobj | tuple[femtobj] | list[femtobj]) -> None:
        if all(isinstance(obj, get_args(femtobj)) for obj in flatten([objs])):
            self.parse_objects(objs)
        else:
            logger.error(
                'Given objects of the wrong type. Cell objects just accept laserpath- or trench-derived objects.'
            )
            raise ValueError(
                'Given objects of the wrong type. Cell objects just accept laserpath- or trench-derived objects.'
            )


class Device:
    def __init__(self, **param: dict[str, Any]) -> None:
        self.cells: dict[str, Cell] = dict()
        self.fig: go.Figure | None = None
        self.fabrication_time: float = 0.0

        self._param: dict[str, Any] = copy.deepcopy(param)
        self._print_angle_warning: bool = True
        self._print_base_cell_warning: bool = True
        logger.info(f'Instantiate device {self._param["filename"].rsplit(".", 1)[0].upper()}.')

    @classmethod
    def from_dict(cls, param: dict[str, Any], **kwargs: Any | None) -> Device:
        """Create an instance of the class from a dictionary.

        It takes a class and a dictionary, and returns an instance of the class with the dictionary's keys as the
        instance's attributes.

        Parameters
        ----------
        param: dict()
           Dictionary mapping values to class attributes.
        kwargs: optional
           Series of keyword arguments that will be used to update the param file before the instantiation of the
           class.

        Returns
        -------
        Instance of class.
        """
        # Update parameters with kwargs
        p = copy.deepcopy(param)
        if kwargs:
            p.update(kwargs)

        logger.debug(f'Create {cls.__name__} object from dictionary.')
        return cls(**p)

    def add(self, objs: Cell | list[Cell] | femtobj | list[femtobj]) -> None:
        objs = flatten([objs])
        for elem in objs:
            if isinstance(elem, Cell):
                self.add_cell(elem)
            elif isinstance(elem, get_args(femtobj)):
                if self._print_base_cell_warning:
                    logger.warning('femto objects added straight to a Device will be added to a common layer: BASE.')
                    self._print_base_cell_warning = False
                self.add_to_cell(key='base', obj=elem)
            else:
                logger.error(
                    'Objects can only be Cells or other femto objects (Waveguide, Markers, etc.). '
                    f'Given {type(elem)}.'
                )
                raise ValueError(
                    'Objects can only be Cells or other femto objects (Waveguide, Markers, etc.). '
                    f'Given {type(elem)}.'
                )

    def add_cell(self, cell: Cell) -> None:
        if cell.name.lower() in self.cells:
            logger.error(f'Cell ID "{cell.name}" already present in layer  dict, give another value.')
            raise ValueError(f'Cell ID "{cell.name}" already present in layer  dict, give another value.')
        self.cells[cell.name] = cell

    def add_to_cell(self, key: str, obj: femtobj | list[femtobj]) -> None:
        """Adds a femto object to a the cell.

        Parameters
        ----------
        key: str
            ID of the cell.
        obj: femto object
            Object (or list of objects) to add to cell.

        Returns
        -------
        None
        """
        key = key.lower()
        if key not in self.cells.keys():
            self.cells[key] = Cell(name=key)
        self.cells[key].add(obj)

    def plot2d(self, show: bool = True, save: bool = False, show_shutter_close: bool = True) -> None:
        logger.info('Plotting 2D objects...')
        self.fig = go.Figure()
        for layer in self.cells.values():
            logger.debug(f'2D plot of layer {layer}.')
            wrs = writers
            for typ, list_objs in layer.objects.items():
                wr = wrs[typ](self._param, objects=list_objs)
                self.fig = wr.plot2d(fig=self.fig, show_shutter_close=show_shutter_close)
        x0, y0, x1, y1 = writers[Waveguide](self._param)._get_glass_borders()
        self.fig = plot2d_base_layer(self.fig, x0=x0, y0=y0, x1=x1, y1=y1)
        if show:
            logger.debug('Show 2D plot.')
            self.fig.show()

        if save:
            self.save()

    def plot3d(self, show: bool = True, save: bool = False, show_shutter_close: bool = True) -> None:
        logger.info('Plotting 3D objects...')
        self.fig = go.Figure()
        for layer in self.cells.values():
            logger.debug(f'3D plot of layer {layer}.')
            wrs = writers
            for typ, list_objs in layer.objects.items():
                wr = wrs[typ](self._param, objects=list_objs)
                self.fig = wr.plot3d(fig=self.fig, show_shutter_close=show_shutter_close)
        self.fig = plot3d_base_layer(self.fig)
        if show:
            logger.debug('Show 3D plot.')
            self.fig.show()
        if save:
            self.save()

    def save(self, filename: str = 'scheme.html', opt: dict[str, Any] | None = None) -> None:
        """Save figure.

        Save the plot as a file.

        Parameters
        ----------
        filename: str, optional
            Filename of the output image file. The default name is "scheme.html".
        opt : dict, optional
            Dictionary with exporting options specifications.

        Returns
        -------
        None

        See Also
        --------
        go.Figure.write_image : Method to export a plotly image.
        """

        if opt is None:
            opt = dict()
        default_opt = {'width': 1980, 'height': 1080, 'scale': 2, 'engine': 'kaleido'}
        opt = {**default_opt, **opt}

        if self.fig is None:
            return None

        fn = pathlib.Path(filename)
        if fn.suffix.lower() in ['.html', '']:
            self.fig.write_html(str(fn.with_suffix('.html')))
        else:
            self.fig.write_image(str(fn), **opt)
        logger.info(f'Plot saved to "{fn}".')

    def pgm(self, verbose: bool = False) -> None:
        for layer in self.cells.values():
            logger.debug(f'Compile G-Code of {layer} layer.')
            wrs = writers
            for typ, list_objs in layer.objects.items():
                wr = wrs[typ](self._param, objects=list_objs)
                wr.pgm(verbose=verbose)
                self.fabrication_time += wr.fab_time

    def export(self, export_dir: str = 'EXPORT') -> None:
        """Export objects to pickle files.

        Export all the objects stored in ``Device`` class as a `pickle` file.

        Parameters
        ----------
        export_dir: str, optional
            Name of the directory inside which export objects.

        Returns
        -------
        None
        """

        logger.info('Exporting layer objects...')
        for layer in self.cells.values():
            wrs = writers
            for typ, list_objs in layer.objects.items():
                wr = wrs[typ](self._param, objects=list_objs)
                wr.export(filename=layer.name.upper(), export_dir=export_dir)
        logger.info('Export completed.')

    def xlsx(self, metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Generate the spreadsheet.

        Add all waveguides and markers of the ``Device`` to the spreadsheet.
        """

        # Case in which metadata is given as keyword argument, use it for the Spreadsheet generation
        if 'metadata' in kwargs.keys():
            mdata = kwargs.pop('metadata')
        elif not metadata:
            mdata = {
                'laser_name': self._param.get('laser') or '',
                'sample_name': pathlib.Path(self._param.get('filename') or '').stem,
            }

        # Fetch all objects from writers
        objs: list[Waveguide | NasuWaveguide] = []
        for layer in self.cells.values():
            objs.extend(layer.objects[Waveguide])
            objs.extend(layer.objects[NasuWaveguide])

        # Generate Spreadsheet
        logger.info('Generating spreadsheet...')
        with Spreadsheet(**kwargs, metadata=mdata) as S:
            S.write(objs)
        logger.info('Excel file created.')

    @staticmethod
    def load_objects(
        folder: str | pathlib.Path, param: dict[str, Any], level: int = 1, verbose: bool = False
    ) -> Device:
        """
        The load_objects method loads the objects from a folder.

        Parameters
        ----------
            folder: str | pathlib.Path
                Specify the folder where the objects are stored.
            param: dict
                Pass a dictionary of parameters to the load_objects function.
            level: int, optional
                Depth level of the directory/file tree for loading files. The default value is 1. ``level=0`` does
                not return anythin, ``level=-1`` traverse all of the subdirectories inside the ``folder``.
            verbose: bool, optional
                Flag for printing the progress of the loading process. The default values is False.

        Returns
        -------

            A list of objects that have been loaded from the given folder
        """

        dev = Device(**param)

        logger.info('Loading objects...')
        for root, dirs, files in walklevel(folder, level):
            if not files and pathlib.Path(root) != pathlib.Path(folder):
                logger.warning(f'No file is present in the given directory {root}.')
            for file in files:
                if verbose and file:
                    logger.info(f'Loading {file} object...')
                filename = pathlib.Path(root) / file
                with open(filename, 'rb') as f:
                    dev.add_to_cell(str(pathlib.Path(root).name), dill.load(f))

        logger.info('Loading complete.')
        return dev


def main() -> None:
    """The main function of the script."""

    from femto.trench import TrenchColumn
    from femto.waveguide import Waveguide

    # Parameters
    param_wg: dict[str, Any] = dict(speed=20, radius=25, pitch=0.080, int_dist=0.007, samplesize=(25, 3))
    param_tc: dict[str, Any] = dict(length=1.0, base_folder='', y_min=0.9, y_max=4 * 0.080 + 1.1, u=[30.0, 32.0])
    param_gc: dict[str, Any] = dict(
        filename='testCell.pgm', laser='PHAROS', new_origin=(0.5, 0.5), samplesize=(25, 3), aerotech_angle=-1.023
    )

    test = Device(**param_gc)
    waveguides = Cell(name='wgs')

    # Waveguides
    x_center = 0
    coup = [Waveguide(**param_wg) for _ in range(5)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch + 1, 0.035])
        wg.linear([9, None, None], mode='INC')
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        x_center = wg.x[-1]
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        wg.end()
        waveguides.add(wg)

    # Trench
    trenches = Cell(name='trenches')
    tcol = TrenchColumn(x_center=x_center, **param_tc)
    tcol.dig_from_waveguide(coup, remove=[0, 1])
    trenches.add(tcol)

    test.add(waveguides)
    test.add(tcol)
    test.add_cell(trenches)

    # test.plot2d()
    # test.save('scheme.pdf')
    # test.pgm()
    # test.xlsx()
    # test.export()

    # test2 = Device.load_objects('.\EXPORT', param_gc)
    # test2.plot2d()
    # test2.save()

    # Export
    # dev.plot2d()
    # dev.save('circuit_scheme.pdf')
    # dev.pgm()
    # dev.export()

    # data_xlsx = dict(
    #     laboratory='CAPABLE',
    #     samplename=pathlib.Path(param_gc['filename']).stem,
    #     material='Cornings Eagle-XG',
    #     facet='Bottom',
    #     thickness='1 mm',
    #     lasername=param_gc['laser'],
    #     wavelength='1030 nm',
    #     duration='180 fs',
    #     reprate='1 MHz',
    #     attenuator='15 %',
    #     preset='1',
    #     objective='20X WI',
    # )
    # dev.xlsx(metadata=data_xlsx)


if __name__ == '__main__':
    main()
