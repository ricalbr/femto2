from __future__ import annotations

import collections
import copy
import os
import pathlib
from typing import Any
from typing import cast
from typing import Union

import dill
import plotly.graph_objects as go
from femto import logger
from femto.curves import sin
from femto.helpers import flatten, listcast
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
femtobj = LaserPath| Waveguide|NasuWaveguide|Marker|Trench|TrenchColumn|UTrenchColumn

class Device:
    def __init__(self, **param) -> None:
        self.layers: collections.OrderedDict[int, Layer] = collections.OrderedDict()
        self.fig: go.Figure | None = None
        self.fabrication_time: float = 0.0
        self._param: dict[str, Any] = dict(**param)
        self._print_angle_warning: bool = True
        logger.info(f'Layer {self._param["filename"].rsplit(".", 1)[0].upper()}.')

    def add_layer(self, layer_id:int, layer:Layer):
        if layer_id in self.layers:
            logger.error(f'Layer ID {layer_id} already present in layer  dict, give another value.')
            raise ValueError(f'Layer ID {layer_id} already present in layer  dict, give another value.')
        self.layers[layer_id] = layer


    def add_to_layer(self, layer_id:int, obj: femtobj|list[femtobj])->None:
        """Adds a femto object to a the layer

        Parameters
        ----------
        layer_id: int
            ID of the layer
        obj: femto object
            Object (or list of objects) to add to layer.

        Returns
        -------
        None
        """
        if layer_id not in self.layers:
            self.layers[layer_id] = Layer(**self._param)
        self.layers[layer_id].extend(listcast(obj))

    def plot2d(self, show: bool = True, save: bool = False, show_shutter_close: bool = True) -> None:
        logger.info('Plotting 2D objects...')
        self.fig = go.Figure()
        for layer in self.layers.values():
            logger.debug(f'2D plot of layer {layer}.')
            self.fig = layer.plot2d(figure=self.fig, show=False, save=False, show_shutter_close=show_shutter_close)
        if show:
            logger.debug('Show 2D plot.')
            self.fig.show()
        if save:
            self.save()

    def plot3d(self, show: bool = True, save: bool = False, show_shutter_close: bool = True) -> None:
        logger.info('Plotting 3D objects...')
        self.fig = go.Figure()
        for layer in self.layers.values():
            logger.debug(f'3D plot of layer {layer}.')
            self.fig = layer.plot3d(figure=self.fig, show=False, save=False, show_shutter_close=show_shutter_close)
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
        for layer in self.layers.values():
            logger.debug(f'Compile G-Code of {layer} layer.')
            layer.pgm(verbose=verbose)
            self.fabrication_time += layer.fabrication_time

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

        logger.info('Exporting objects of layer {}...')
        for layer in self.layers.values():
            layer.export(export_dir=export_dir)
        logger.info('Export completed.')

    def xlsx(self, metadata: dict[str, Any] | None = None, **kwargs) -> None:
        """Generate the spreadsheet.

        Add all waveguides and markers of the ``Device`` to the spreadsheet.
        """

        # Case in which metadata is given as keyword argument, use it for the Spreadsheet generation
        if 'metadata' in kwargs.keys():
            metadata = kwargs.pop('metadata')
        elif not metadata:
            metadata = {
                'laser_name': self._param.get('laser') or '',
                'sample_name': pathlib.Path(self._param.get('filename') or '').stem,
            }

        # Fetch all objects from writers
        objs = []
        for layer in self.layers.values():
            for key, writer in layer.writers.items():
                if isinstance(writer, (WaveguideWriter, NasuWriter, MarkerWriter)):
                    objs.extend(writer.objs)

        # Generate Spreadsheet
        logger.info('Generating spreadsheet...')
        with Spreadsheet(**kwargs, metadata=metadata) as S:
            S.write(objs)
        logger.info('Excel file created.')

class Layer:
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

        self._param: dict[str, Any] = dict(**param)
        self._print_angle_warning: bool = True
        logger.info(f'Instantiate device {self._param["filename"].rsplit(".", 1)[0]}.')

    @classmethod
    def from_dict(cls, param: dict[str, Any], **kwargs):
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
        Instance of class
        """
        # Update parameters with kwargs
        p = copy.deepcopy(param)
        if kwargs:
            p.update(kwargs)

        logger.debug(f'Create {cls.__name__} object from dictionary.')
        return cls(**p)

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

    def plot2d(self, figure:go.Figure | None = None, show: bool = False, save: bool = False, show_shutter_close: bool = True) ->  go.Figure:
        """Plot 2D.

        2D plot of all the objects stored in the ``Device`` class.

        Parameters
        ----------
        figure : go.Figure or None, optional
            Plotly figure, if None create a new one otherwise update the already existing one.
        show : bool, optional
            Boolean flag to automatically show the plot. The default value is False.
        save : bool, optional
            Boolean flag to automatically save the plot. The default value is False.
        show_shutter_close : bool, optional
            Boolean flag to show the lines with closed shutter. The default value is True.

        Returns
        -------
        None
        """

        logger.info('Plotting 2D objects...')
        if figure is None:
            self.fig = go.Figure()
        else:
            self.fig = figure
        for writer in self.writers.values():
            logger.debug(f'Plot 2D object from {writer}.')
            self.fig = writer.plot2d(self.fig, show_shutter_close=show_shutter_close)
            logger.debug('Update 2D figure.')
            self.fig = writer.standard_2d_figure_update(self.fig)
        if show:
            logger.debug('Show 2D plot.')
            self.fig.show()
        if save:
            self.save()
        return self.fig

    def plot3d(self, figure: go.Figure|None = None, show: bool = True, save: bool = False, show_shutter_close: bool = True) -> go.Figure:
        """Plot 3D.

        3D plot of all the objects stored in the ``Device`` class.

        Parameters
        ----------
        figure : go.Figure or None, optional
            Plotly figure, if None create a new one otherwise update the already existing one.
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
        if figure is None:
            self.fig = go.Figure()
        else:
            self.fig = figure
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
        return self.fig

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
                logger.info(f'Exporting G-Code of {key.__name__} objects...')

            # writer = cast(Union[WaveguideWriter, NasuWriter, TrenchWriter, UTrenchWriter, MarkerWriter], writer)
            if writer.objs:
                writer.pgm(verbose=self._print_angle_warning)
                self._print_angle_warning = False

            self.fabrication_time += writer.fab_time

        logger.info('\b' * 20)
        if verbose:
            logger.info('Export .pgm files completed.')

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

        logger.info('Exporting objects...')
        for key, writer in self.writers.items():
            if verbose and writer.objs:
                logger.info(f'Exporting {key.__name__} objects...')

            writer = cast(Union[WaveguideWriter, NasuWriter, TrenchWriter, UTrenchWriter, MarkerWriter], writer)
            writer.export(export_dir=export_dir)
        logger.info('Export completed.')

    def xlsx(self, metadata: dict[str, Any] | None = None, **kwargs) -> None:
        """Generate the spreadsheet.

        Add all waveguides and markers of the ``Device`` to the spreadsheet.
        """

        # Case in which metadata is given as keyword argument, use it for the Spreadsheet generation
        if 'metadata' in kwargs.keys():
            metadata = kwargs.pop('metadata')
        elif not metadata:
            metadata = {
                'laser_name': self._param.get('laser') or '',
                'sample_name': pathlib.Path(self._param.get('filename') or '').stem,
            }

        # Fetch all objects from writers
        objs = []
        for key, writer in self.writers.items():
            if isinstance(writer, (WaveguideWriter, NasuWriter, MarkerWriter)):
                objs.extend(writer.objs)

        # Generate Spreadsheet
        logger.info('Generating spreadsheet...')
        with Spreadsheet(**kwargs, metadata=metadata) as S:
            S.write(objs)
        logger.info('Excel file created.')

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

    @staticmethod
    def load_objects(folder: str | pathlib.Path, param: dict[str, Any], verbose: bool = False) -> Layer:
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

        dev = Layer(**param)
        objs = []

        logger.info('Loading objects...')
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
        logger.info('Loading complete.')
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

    test = Device(**param_gc)
    waveguides = Layer.from_dict(param_gc, filename='waveguides')

    # Waveguides
    x_center = 0
    coup = [Waveguide(**param_wg) for _ in range(5)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch, 0.035])
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        x_center = wg.x[-1]
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        wg.end()
        waveguides.append(wg)

    # Trench
    trenches = Layer.from_dict(param_gc, filename='trenches')
    tcol = TrenchColumn(x_center=x_center, **param_tc)
    tcol.dig_from_waveguide(coup, remove=[0, 1])
    trenches.append(tcol)

    test.add_layer(1, waveguides)
    test.add_layer(2, trenches)
    test.add_to_layer(2, wg)

    test.plot2d()
    test.plot3d()
    test.pgm()
    test.xlsx()

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
