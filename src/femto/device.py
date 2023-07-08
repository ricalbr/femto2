from __future__ import annotations

import collections
import copy
import pathlib
from typing import Any
from typing import cast
from typing import Union

import plotly.graph_objects as go
from femto.helpers import flatten
from femto.marker import Marker
from femto.trench import TrenchColumn, UTrenchColumn
from femto.waveguide import NasuWaveguide
from femto.waveguide import Waveguide
from femto.writer import MarkerWriter, UTrenchWriter
from femto.writer import NasuWriter
from femto.writer import TrenchWriter
from femto.writer import WaveguideWriter
from femto.spreadsheet import Spreadsheet


class Device:
    """Class representing a Device.

    A Device is a collection of ``Waveguide``, ``NasuWaveguide``, ``Marker``, ``Trench`` (or ``TrenchColumn``)
    objects.
    """

    def __init__(self, **param) -> None:
        self._param: dict[str, Any] = dict(**param)
        self.unparsed_objects: list[Any] = []
        self.fig: go.Figure | None = None
        self.fabrication_time: float = 0.0
        self.writers = {
            Waveguide: WaveguideWriter(wg_list=[], **param),
            NasuWaveguide: NasuWriter(nw_list=[], **param),
            TrenchColumn: TrenchWriter(tc_list=[], **param),
            UTrenchColumn: UTrenchWriter(utc_list=[], **param),
            Marker: MarkerWriter(mk_list=[], **param),
        }

    def append(self, obj: Any) -> None:
        """Append object to Device.

        Parameters
        ----------
        obj: any object that can be stored in a ``Device`` class.

        Returns
        -------
        None
        """

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
        self.parse_objects(unparsed_objects=copy.copy(obj))

    def parse_objects(self, unparsed_objects: Any | list[Any]) -> None:
        """Parse objects.

        The function takes a list of objects and parse all of them based on their types.
        If the type of the object matches one of the types of the ``Writer`` registered in the ``Device`` class,
        the object is added to the ``Writer.obj_list``. If not, a ``TypeError`` is raised.

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
                self.writers[k].extend(e)
            except KeyError as err:
                raise TypeError(f'Found unexpected type {err.args}.')

    def plot2d(self, show: bool = True, save: bool = False) -> None:
        """Plot 2D.

        2D plot of all the objects stored in the ``Device`` class.

        Parameters
        ----------
        show : bool, optional
            Boolean flag to automatically show the plot. The default value is True.
        save : bool, optional
            Boolean flag to automatically save the plot. The default value is False.

        Returns
        -------
        None
        """

        self.fig = go.Figure()
        for writer in self.writers.values():
            # TODO: fix standard fig update
            self.fig = writer.plot2d(self.fig)
            self.fig = writer.standard_2d_figure_update(self.fig)
        if show:
            self.fig.show()
        if save:
            self.save()

    def plot3d(self, show: bool = True, save: bool = False) -> None:
        """Plot 3D.

        3D plot of all the objects stored in the ``Device`` class.

        Parameters
        ----------
        show : bool, optional
            Boolean flag to automatically show the plot. The default value is True.
        save : bool, optional
            Boolean flag to automatically save the plot. The default value is False.

        Returns
        -------
        None
        """

        self.fig = go.Figure()
        for key, writer in self.writers.items():
            try:
                self.fig = writer.plot3d(self.fig)
                self.fig = writer.standard_3d_figure_update(self.fig)
            except NotImplementedError:
                print(f'3D {key} plot not yet implemented.\n')
        if show:
            self.fig.show()
        if save:
            self.save()

    def pgm(self, verbose: bool = True) -> None:
        """Export to PGM.

        Export all the objects stored in ``Device`` class as a `PGM` file.

        Parameters
        ----------
        verbose : bool, optional
            Boolean flag to print informations during the export operation.

        Returns
        -------
        None
        """

        for key, writer in self.writers.items():
            if verbose:
                print(f'Exporting {key.__name__} objects...')

            writer = cast(Union[WaveguideWriter, NasuWriter, TrenchWriter, UTrenchWriter, MarkerWriter], writer)
            writer.pgm(verbose=verbose)

            self.fabrication_time += writer._fabtime
        if verbose:
            print('Export .pgm files complete.\n')

    def xlsx(self, verbose: bool = True, **param) -> None:
        """Generate the spreadsheet.

        Add all waveguides and markers of the ``Device`` to the spreadsheet.
        """

        with Spreadsheet(device=self, **param) as spsh:
            if verbose:
                print('Generating spreadsheet...')
            spsh.write_structures(verbose=verbose)
        if verbose:
            print('Create .xlsx file complete.\n')

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


def main() -> None:
    from femto.trench import TrenchColumn
    from femto.waveguide import Waveguide

    # Parameters
    PARAM_WG: dict[str, Any] = dict(speed=20, radius=25, pitch=0.080, int_dist=0.007, samplesize=(25, 3))
    PARAM_TC: dict[str, Any] = dict(length=1.0, base_folder='', y_min=-0.1, y_max=4 * 0.080 + 0.1, u=[30.0, 32.0])
    PARAM_GC: dict[str, Any] = dict(filename='testCell.pgm', laser='PHAROS', new_origin=(0.5, 0.5), samplesize=(25, 1))

    dev = Device(**PARAM_GC)

    # Waveguides
    x_center = 0
    coup = [Waveguide(**PARAM_WG) for _ in range(5)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch, 0.035])
        wg.sin_coupler((-1) ** i * wg.dy_bend)
        x_center = wg.x[-1]
        wg.sin_coupler((-1) ** i * wg.dy_bend)
        wg.end()
        dev.append(wg)

    # Trench
    T = TrenchColumn(x_center=x_center, **PARAM_TC)
    T.dig_from_waveguide(coup, remove=[0, 1])
    dev.append(T)

    # Export
    dev.plot2d()
    dev.save('circuit_scheme.pdf')
    dev.pgm()


if __name__ == '__main__':
    main()
