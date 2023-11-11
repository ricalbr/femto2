from __future__ import annotations

import collections
import copy
import os
import pathlib
from typing import Any
from typing import cast
from typing import TypeVar
from typing import Union

import dill
import plotly.graph_objects as go
from femto import logger
from femto.curves import sin
from femto.helpers import flatten
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
from femto.writer import TrenchWriter
from femto.writer import UTrenchWriter
from femto.writer import WaveguideWriter

# Create a generic variable that can be 'Device', or any subclass.
DV = TypeVar('DV', bound='Device')

types = dict(WG=Waveguide, NWG=NasuWaveguide, MK=Marker, LP=LaserPath, TR=Trench, TC=TrenchColumn, UTC=UTrenchColumn)


class Device:
    """Class representing a Device.

    A Device is a collection of ``Waveguide``, ``NasuWaveguide``, ``Marker``, ``Trench`` (or ``TrenchColumn``)
    objects.
    """

    def __init__(self, **param) -> None:
        self.unparsed_objects: list[Any] = []
        self.fig: go.Figure | None = None
        self.fabrication_time: float = 0.0
        self.writers = {
            TrenchColumn: TrenchWriter(param=param, objects=[]),
            UTrenchColumn: UTrenchWriter(param=param, objects=[]),
            Marker: MarkerWriter(param=param, objects=[]),
            Waveguide: WaveguideWriter(param=param, objects=[]),
            NasuWaveguide: NasuWriter(param=param, objects=[]),
        }

        self._param: dict[str, Any] = copy.deepcopy(param)
        logger.info(f'Instantiate device {self._param["filename"].rsplit(".", 1)[0]}.')

    @classmethod
    def from_dict(cls, param: dict[str, Any]):
        return cls(**param)

    def append(self, obj: Any) -> None:
        """Append object to Device.

        Parameters
        ----------
        obj: any object that can be stored in a ``Device`` class.

        Returns
        -------
        None
        """

        logger.debug(f'Parsing {obj}.')
        self.parse_objects(unparsed_objects=copy.copy(flatten([obj])))

    def extend(self, obj: list[Any]) -> None:
        """Exend Device objects list.

        Parameters
        ----------
        obj: List of any object that can be stored in a ``Device`` class.

        Returns
        -------
        None
        """

        if not isinstance(obj, list):
            raise TypeError(f'The object must be a list. {type(obj)} was given.')
        logger.debug(f'Parsing {obj}.')
        self.parse_objects(unparsed_objects=copy.copy(obj))

    def parse_objects(self, unparsed_objects: Any | list[Any]) -> None:
        """Parse objects.

        The function takes a list of objects and parse all of them based on their types.
        If the type of the object matches one of the types of the ``Writer`` registered in the ``Device`` class,
        the object is added to the ``Writer._obj_list``. If not, a ``TypeError`` is raised.

        Parameters
        ----------
        unparsed_objects : list
            List of object that can be stored in a ``Device`` class.

        Returns
        -------
        None
        """

        # split the unparsed_object list based on the type of each element
        d = collections.defaultdict(list)
        while unparsed_objects:
            obj = unparsed_objects.pop(0)
            if isinstance(obj, list):
                d[type(obj[0])].append(obj)
            else:
                d[type(obj)].append(obj)

        # add each element to the type-matching writer
        for k, e in d.items():
            try:
                logger.debug(f'Assign {e} to {self.writers[k]}.')
                self.writers[k].extend(e)
            except KeyError as err:
                logger.error(f'Found unexpected type {err.args}.')
                raise TypeError(f'Found unexpected type {err.args}.')

    def plot2d(self, show: bool = True, save: bool = False, show_shutter_close: bool = True) -> None:
        """Plot 2D.

        2D plot of all the objects stored in the ``Device`` class.

        Parameters
        ----------
        show : bool, optional
            Boolean flag to automatically show the plot. The default value is True.
        save : bool, optional
            Boolean flag to automatically save the plot. The default value is False.
        show_shutter_close : bool, optional
            Boolean flag to show the lines with closed shutter. The default value is True.

        Returns
        -------
        None
        """

        logger.info('Plotting 2D objects...')
        self.fig = go.Figure()
        for writer in self.writers.values():
            # TODO: fix standard fig update
            logger.debug(f'Plot 2D object from {writer}.')
            self.fig = writer.plot2d(self.fig, show_shutter_close=show_shutter_close)
            logger.debug('Update 2D figure.')
            self.fig = writer.standard_2d_figure_update(self.fig)
        if show:
            logger.debug('Show 2D plot.')
            self.fig.show()
        if save:
            self.save()

    def plot3d(self, show: bool = True, save: bool = False, show_shutter_close: bool = True) -> None:
        """Plot 3D.

        3D plot of all the objects stored in the ``Device`` class.

        Parameters
        ----------
        show : bool, optional
            Boolean flag to automatically show the plot. The default value is True.
        save : bool, optional
            Boolean flag to automatically save the plot. The default value is False.
        show_shutter_close : bool, optional
            Boolean flag to show the lines with closed shutter. The default value is True.

        Returns
        -------
        None
        """

        logger.info('Plotting 3D objects...')
        self.fig = go.Figure()
        for key, writer in self.writers.items():
            logger.debug(f'Plot 3D object from {writer}.')
            self.fig = writer.plot3d(self.fig, show_shutter_close=show_shutter_close)
            logger.debug('Update 3D figure.')
            self.fig = writer.standard_3d_figure_update(self.fig)
        if show:
            logger.debug('Show 3D plot.')
            self.fig.show()
        if save:
            self.save()

    def pgm(self, verbose: bool = False) -> None:
        """Export to PGM.

        Export all the objects stored in ``Device`` class as a `PGM` file.

        Parameters
        ----------
        verbose : bool, optional
            Boolean flag to print informations during the export operation. The default value is ``False``.

        Returns
        -------
        None
        """

        for key, writer in self.writers.items():
            if verbose and writer.objs:
                logger.info(f'Exporting {key.__name__} objects...')

            # writer = cast(Union[WaveguideWriter, NasuWriter, TrenchWriter, UTrenchWriter, MarkerWriter], writer)
            writer.pgm(verbose=verbose)

            self.fabrication_time += writer.fab_time
        if verbose:
            logger.info('Export .pgm files completed.\n')

    def export(self, export_dir: str = 'EXPORT', verbose: bool = False, **kwargs) -> None:
        """Export objects to pickle files.

        Export all the objects stored in ``Device`` class as a `pickle` file.

        Parameters
        ----------
        export_dir: str, optional
            Name of the directory inside which export objects.
        verbose : bool, optional
            Boolean flag to print informations during the export operation.

        Returns
        -------
        None
        """

        for key, writer in self.writers.items():
            if verbose and writer.objs:
                logger.info(f'Exporting {key.__name__} objects...')

            writer = cast(Union[WaveguideWriter, NasuWriter, TrenchWriter, UTrenchWriter, MarkerWriter], writer)
            writer.export(export_dir=export_dir)
        if verbose:
            logger.info('Export objects completed.\n')

    def xlsx(self, verbose: bool = True, **param) -> None:
        """Generate the spreadsheet.

        Add all waveguides and markers of the ``Device`` to the spreadsheet.
        """

        with Spreadsheet(device=self, **param) as spsh:
            if verbose:
                logger.info('Generating spreadsheet...')
            spsh.write_structures(verbose=verbose)
        if verbose:
            logger.info('Create .xlsx file completed.')

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
        logger.info(f'Saving plot to "{fn}".')

        if fn.suffix.lower() in ['.html', '']:
            self.fig.write_html(str(fn.with_suffix('.html')))
        else:
            self.fig.write_image(str(fn), **opt)

    @staticmethod
    def load_objects(folder: str | pathlib.Path, param: dict[str, Any], verbose: bool = False) -> Device:
        """
        The load_objects method loads the objects from a folder.

        Parameters
        ----------
            folder: str | pathlib.Path
                Specify the folder where the objects are stored
            param: dict
                Pass a dictionary of parameters to the load_objects function
            verbose: bool
                Print the progress of the function

        Returns
        -------

            A list of objects that have been loaded from the given folder
        """

        dev = Device(**param)
        objs = []

        for root, dirs, files in os.walk(folder):
            if not files:
                logger.warning(f'No file is present in the given directory {folder}.')
            for file in files:
                if verbose and file:
                    logger.info(f'Loading {file} object...')
                filename = pathlib.Path(root) / file
                with open(filename, 'rb') as f:
                    objs.append(dill.load(f))

        dev.append(objs)
        if verbose:
            logger.info('Loading objects completed.\n')
        return dev


def main() -> None:
    """The main function of the script."""

    from femto.trench import TrenchColumn
    from femto.waveguide import Waveguide

    # Parameters
    param_wg: dict[str, Any] = dict(speed=20, radius=25, pitch=0.080, int_dist=0.007, samplesize=(25, 3))
    param_tc: dict[str, Any] = dict(length=1.0, base_folder='', y_min=-0.1, y_max=4 * 0.080 + 0.1, u=[30.0, 32.0])
    param_gc: dict[str, Any] = dict(
        filename='testCell.pgm', laser='PHAROS', new_origin=(0.5, 0.5), samplesize=(25, 1), aerotech_angle=-1.023
    )

    dev = Device.from_dict(param_gc)

    # Waveguides
    x_center = 0
    coup = [Waveguide(**param_wg) for _ in range(5)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch, 0.035])
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        x_center = wg.x[-1]
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        wg.end()
        dev.append(wg)

    # Trench
    tcol = TrenchColumn(x_center=x_center, **param_tc)
    tcol.dig_from_waveguide(coup, remove=[0, 1])
    dev.append(tcol)

    # Export
    # dev.plot2d()
    # dev.save('circuit_scheme.pdf')
    dev.pgm()
    dev.export()


if __name__ == '__main__':
    main()
