from __future__ import annotations

import abc
import itertools
import pathlib
import time
from typing import Any

import numpy as np
import numpy.typing as npt
from shapely import geometry

from femto.helpers import flatten
from femto.helpers import listcast
from femto.helpers import nest_level
from femto.helpers import split_mask
from femto.marker import Marker
from femto.pgmcompiler import PGMCompiler
from femto.trench import Trench
from femto.trench import TrenchColumn
from femto.waveguide import NasuWaveguide
from femto.waveguide import Waveguide
from plotly import graph_objs as go


class Writer(PGMCompiler, abc.ABC):
    """Abstract class representing a G-Code Writer object.

    A Writer class is a super-object that can store homogeneous objects (Waveguides, Markers, Trenches,
    etc.) and provides methods to append objects, plot and export them as .pgm files.
    """

    @abc.abstractmethod
    def append(self, obj: Any) -> None:
        """Append objects.

        Abstract method for appending objects to a Writer.

        Parameters
        ----------
        obj : Any
            Object to be appended to current Writer.

        Returns
        -------
        None
        """
        pass

    @abc.abstractmethod
    def extend(self, obj: list[Any]) -> None:
        """Extend objects list.

        Abstract method for extending the object list of a Writer.

        Parameters
        ----------
        obj : list(any)
            List of objects to extend the Writer object list.

        Returns
        -------
        None
        """
        pass

    @abc.abstractmethod
    def plot2d(
        self,
        fig: go.Figure | None = None,
        show_shutter_close: bool = True,
        style: dict[str, Any] | None = None,
    ) -> go.Figure:
        """Plot 2D.

        Abstract method for plotting the object stored in Writer's object list.

        Parameters
        ----------
        fig : go.Figure, optional
            Optional plotly figure object, if no figure is provided a new one is created and updated with the object
            stored in the Writer class. If a figure is provided it is updated by adding the stored objects. The
            default behaviour is creating a new figure.
        show_shutter_close : bool, optional
            Boolean flag, if ``True`` the movements with closed shutter are represented. The default value is ``False``.
        style: dict(str, Any)
            Plotly compatible styles options for representing the objects.

        Returns
        -------
        go.Figure
            Plotly figure with the 2D plot of the stored objects.

        See Also
        --------
        go.Figure : Plotly figure object.
        """
        pass

    @abc.abstractmethod
    def plot3d(
        self,
        fig: go.Figure | None = None,
        show_shutter_close: bool = True,
        style: dict[str, Any] | None = None,
    ) -> go.Figure:
        """Plot 3D.

        Abstract method for plotting the object stored in Writer's object list.

        Parameters
        ----------
        fig : go.Figure, optional
            Optional plotly figure object, if no figure is provided a new one is created and updated with the object
            stored in the Writer class. If a figure is provided it is updated by adding the stored objects. The
            default behaviour is creating a new figure.
        show_shutter_close : bool, optional
            Boolean flag, if ``True`` the movements with closed shutter are represented. The default value is ``False``.
        style: dict(str, Any)
            Plotly compatible styles options for representing the objects.

        Returns
        -------
        go.Figure
            Plotly figure with the 3D plot of the stored objects.

        See Also
        --------
        go.Figure : Plotly figure object.
        """
        pass

    @abc.abstractmethod
    def pgm(self, verbose: bool = True) -> None:
        """Export to PGM file.

        Abstract method for exporting the objects stored in Writer object as PGM file.

        Parameters
        ----------
        verbose : bool, optional
            Boolean flag, if ``True`` some information about the exporting procedures are printed. The default value is
            ``False``.

        Returns
        -------
        None

        See Also
        --------
        ``femto.pgmcompiler.PGMCompiler`` : class that convert lists of points to PGM file.
        """
        pass

    def standard_2d_figure_update(self, fig: go.Figure) -> go.Figure:
        """2D plot update.

        Helper function that update a 2D plot by adding the rectangle representing the sample glass, the `(0, 0)`
        origin point and formats the axis.

        Parameters
        ----------
        fig : go.Figure
            Plotly Figure object to update.

        Returns
        -------
        go.Figure
            Input figure updated with sample glass shape, origin point and axis.
        """

        # GLASS
        fig.add_shape(
            type='rect',
            x0=self.new_origin[0] if self.flip_x else 0 - self.new_origin[0],
            y0=self.new_origin[1] if self.flip_y else 0 - self.new_origin[1],
            x1=self.new_origin[0] - self.xsample if self.flip_x else self.xsample - self.new_origin[0],
            y1=self.new_origin[1] - self.ysample if self.flip_y else self.ysample - self.new_origin[1],
            fillcolor='#D0FAF9',
            line_color='#000000',
            line_width=2,
            layer='below',
        )

        # ORIGIN
        fig.add_trace(
            go.Scattergl(
                x=[0],
                y=[0],
                marker=dict(color='red', size=12),
                hoverinfo='none',
                showlegend=False,
            )
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(pad=15),
            xaxis=dict(
                title='x [mm]',
                showgrid=False,
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='black',
                ticklen=10,
                tick0=0,
                ticks='outside',
                fixedrange=False,
                minor=dict(
                    ticklen=5,
                    tickmode='linear',
                    ticks='outside',
                ),
            ),
            yaxis=dict(
                title='y [mm]',
                showgrid=False,
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='black',
                ticklen=10,
                tick0=0,
                ticks='outside',
                fixedrange=False,
                minor=dict(
                    ticklen=5,
                    tickmode='linear',
                    ticks='outside',
                ),
            ),
            annotations=[
                dict(
                    x=0,
                    y=0,
                    text='(0,0)',
                    showarrow=False,
                    xanchor='left',
                    xshift=-25,
                    yshift=-20,
                    font=dict(color='red'),
                )
            ],
        )
        return fig

    @staticmethod
    def standard_3d_figure_update(fig: go.Figure) -> go.Figure:
        """3D plot update.

        Helper function that update a 3D plot by adding the sample glass, the `(0, 0, 0)` origin point and formats
        the axis.

        Parameters
        ----------
        fig : go.Figure
            Plotly Figure object to update.

        Returns
        -------
        go.Figure
            Input figure updated with sample glass shape, origin point and axis.
        """
        # ORIGIN
        fig.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                marker=dict(color='red', size=12),
                hoverinfo='none',
                showlegend=False,
            )
        )

        fig.update_layout(
            scene=dict(
                bgcolor='rgba(0,0,0,0)',
                aspectratio=dict(
                    x=2,
                    y=1,
                    z=0.25,
                ),
                xaxis=dict(
                    title='x [mm]',
                    showgrid=False,
                    zeroline=False,
                ),
                yaxis=dict(
                    title='y [mm]',
                    showgrid=False,
                    zeroline=False,
                ),
                zaxis=dict(
                    title='z [mm]',
                    showgrid=False,
                    zeroline=False,
                ),
                annotations=[
                    dict(
                        x=0,
                        y=0,
                        z=0,
                        text='(0,0,0)',
                        showarrow=False,
                        xanchor='left',
                        xshift=10,
                        font=dict(color='red'),
                    )
                ],
            )
        )
        return fig


class TrenchWriter(Writer):
    """Trench Writer class."""

    def __init__(self, tc_list: TrenchColumn | list[TrenchColumn], dirname: str = 'TRENCH', **param) -> None:
        super().__init__(**param)
        self.obj_list: list[TrenchColumn] = flatten(listcast(tc_list))
        self.trenches: list[Trench] = [tr for col in self.obj_list for tr in col]
        self.dirname: str = dirname

        self._param: dict[str, Any] = dict(**param)
        self._export_path = self.CWD / (self.export_dir or '') / (self.dirname or '')
        self._fabtime: float = 0.0

    def append(self, obj: TrenchColumn) -> None:
        """Append TrenchColumn objects.

        Parameters
        ----------
        obj: TrenchColumn
            TrenchColumn object to be added to object list.

        Returns
        -------
        None
        """

        if not isinstance(obj, TrenchColumn):
            raise TypeError(f'The object must be a TrenchColumn. {type(obj).__name__} was given.')
        self.obj_list.append(obj)
        self.trenches.extend(obj._trench_list)

    def extend(self, obj: list[TrenchColumn]) -> None:
        """Extend trench list.

        Extend the object list with a list of TrenchColumn objects.

        Parameters
        ----------
        obj : list(TrenchColumn)
            List of TrenchColumn objects to be added to object list.

        Returns
        -------
        None
        """

        if not isinstance(obj, list):
            raise TypeError(f'The object must be a list. {type(obj).__name__} was given.')
        for tr_col in flatten(obj):
            self.append(tr_col)

    def plot2d(
        self, fig: go.Figure | None = None, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """Plot 2D.

        2D plot of the Trench objects contained in ``self.obj_list``.

        Parameters
        ----------
        fig : go.Figure, optional
            Optional plotly figure object, if no figure is provided a new one is created and updated with the Trench
            objects stored in the Writer class. If a figure is provided it is updated by adding the stored objects. The
            default behaviour is creating a new figure.
        show_shutter_close : bool, optional
            Boolean flag, if ``True`` the movements with closed shutter are represented. The default value is ``False``.
        style: dict(str, Any)
            Plotly compatible styles options for representing the objects.

        Returns
        -------
        go.Figure
            Plotly figure with the 2D plot of the stored Trench.
        """

        # If existing figure is given as input parameter append to the figure and return it
        if fig is not None:
            return self._plot2d_trench(fig=fig, style=style)

        # If fig is None create a new figure from scratch
        fig = go.Figure()
        fig = self._plot2d_trench(fig=fig, style=style)
        fig = super().standard_2d_figure_update(fig)  # Add glass, origin and axis elements
        return fig

    def plot3d(
        self,
        fig: go.Figure | None = None,
        show_shutter_close: bool = True,
        style: dict[str, Any] | None = None,
    ) -> go.Figure:
        """Plot 3D - Not implemented."""
        raise NotImplementedError()

    def pgm(self, verbose: bool = True) -> None:
        """Export to PGM file.

        Function for the compilation of TrenchColumn objects.
        For each trench in the column, the function first compile a PGM file for border (or wall) and for the floor
        inside a directory given by the user (``self.base_folder``).
        Secondly, the function produce a `FARCALL.pgm` program to fabricate all the trenches in the column.

        Parameters
        ----------
        verbose : bool, optional
            Boolean flag, if ``True`` some information about the exporting procedures are printed. The default value is
            ``False``.

        Returns
        -------
        None

        See Also
        --------
        ``femto.pgmcompiler.PGMCompiler`` : class that convert lists of points to PGM file.
        """

        if not self.obj_list:
            return None

        # Export each Trench for each TrenchColumn
        for i_col, col in enumerate(self.obj_list):
            # Prepare PGM files export directory
            col_dir = self._export_path / f'trenchCol{i_col + 1:03}'
            col_dir.mkdir(parents=True, exist_ok=True)

            # Export walls and floors as .pgm files
            self._export_trench_column(column=col, column_path=col_dir)

            # Create FARCALL.pgm file for all trenches in current column
            self._farcall_trench_column(column=col, index=i_col)

        # Create a MAIN farcall file, calls all columns .pgm scripts
        main_param = dict(self._param.copy())
        main_param['filename'] = self._export_path / 'MAIN.pgm'
        main_param['aerotech_angle'] = None
        main_param['rotation_angle'] = None

        farcall_list = [
            str(pathlib.Path(col.base_folder) / f'FARCALL{i + 1:03}.pgm') for i, col in enumerate(self.obj_list)
        ]
        with PGMCompiler(**main_param) as G:
            G.farcall_list(farcall_list)

        if verbose:
            _tc_fab_time = 0.0
            for col in self.obj_list:
                _tc_fab_time += col.fabrication_time + 10

            print('=' * 79)
            print('G-code compilation completed.')
            print(
                'Estimated fabrication time of the isolation trenches: \t',
                time.strftime('%H:%M:%S', time.gmtime(_tc_fab_time)),
            )
            print('=' * 79, '\n')

            self._fabtime = _tc_fab_time

    def export_array2d(
        self,
        filename: pathlib.Path,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        speed: float,
        forced_deceleration: bool = False,
    ) -> None:
        """Export 2D path to PGM file.

        Helper function that produces a series of movements at given traslation speed and without shuttering
        operations for a 2D point matrix.
        The function parse the points input points, applies the rotation and homothety transformations and parse all
        the ``LINEAR`` instructions.

        Parameters
        ----------
        filename : pathlib.Path
            Filename of the output `PGM` file.
        x : numpy.ndarray
            `x` coordinates array [mm].
        y : numpy.ndarray
            `y` coordinates array [mm].
        speed : float
            Translation speed [mm/s].
        forced_deceleration: bool
            Add a `G9` command before `LINEAR` movements to reduce the acceleration to zero after the motion is
            completed.

        Returns
        -------
        None

        See Also
        --------
        pgmcompiler.transform_points : series of geometrical transformation on input points.
        pathlib.Path : class representing cross-system filepaths.
        """

        if filename is None:
            raise ValueError('No filename given.')

        # Transform points
        x_arr, y_arr, _ = self.transform_points(x, y, np.zeros_like(x))
        z_arr = [None]

        # Export points
        instr = [
            self._format_args(x, y, z, f)
            for (x, y, z, f) in itertools.zip_longest(x_arr, y_arr, z_arr, listcast(speed))
        ]
        if forced_deceleration:
            gcode_instr = [f'G9 LINEAR {line}\n' for line in instr]
        else:
            gcode_instr = [f'LINEAR {line}\n' for line in instr]

        with open(filename, 'w') as file:
            file.write(''.join(gcode_instr))

    # Private interface
    def _export_trench_column(self, column: TrenchColumn, column_path: pathlib.Path) -> None:
        """Export Trench columns to PGM file.

        Helper function that exports the wall and floor scripts for each trench in a Trench Column.

        Parameters
        ----------
        column : TrenchColumn
            TrenchColumn object containing the Trench blocks to export.
        column_path : pathlib.Path
            Directory for the floor and wall `PGM` files.

        Returns
        -------
        None

        See Also
        --------
        pathlib.Path : class representing cross-system filepaths.
        """

        for i, trench in enumerate(column):
            # Wall script
            x_wall, y_wall = trench.border
            self.export_array2d(
                filename=column_path / f'trench{i + 1:03}_WALL.pgm',
                x=x_wall,
                y=y_wall,
                speed=column.speed_wall,
            )
            del x_wall, y_wall

            # Floor script
            x_floor = np.array([])
            y_floor = np.array([])
            for x_temp, y_temp in trench.toolpath():
                x_floor = np.append(x_floor, x_temp)
                y_floor = np.append(y_floor, y_temp)

            self.export_array2d(
                filename=column_path / f'trench{i + 1:03}_FLOOR.pgm',
                x=x_floor,
                y=y_floor,
                speed=column.speed_floor,
            )
            del x_floor, y_floor

    def _farcall_trench_column(self, column: TrenchColumn, index: int) -> None:
        """Trench Column FARCALL generator

        The function compiles a Trench Column by loading the wall and floor scripts, and then calling them with the
        appropriate order and `U` parameters.
        It produces a `FARCALL` file calling all the trenches of the TrenchColumn.

        Parameters
        ----------
        column: TrenchColumn
            TrenchColumn object to fabricate (and export as `PGM` file).
        index: int
            Index of the trench column. It is used for automatic filename creation.

        Returns
        -------
        None
        """

        column_param = dict(self._param.copy())
        column_param['filename'] = self._export_path / f'FARCALL{index + 1:03}.pgm'
        with PGMCompiler(**column_param) as G:
            G.dvar(['ZCURR'])

            for nbox, (i_trc, trench) in list(itertools.product(range(column.nboxz), list(enumerate(column)))):
                # load filenames (wall/floor)
                wall_filename = f'trench{i_trc + 1:03}_wall.pgm'
                floor_filename = f'trench{i_trc + 1:03}_floor.pgm'
                wall_path = pathlib.Path(column.base_folder) / f'trenchCol{index + 1:03}' / wall_filename
                floor_path = pathlib.Path(column.base_folder) / f'trenchCol{index + 1:03}' / floor_filename

                # INIT POINT
                x0, y0, z0 = self.transform_points(
                    trench.xborder[0],
                    trench.yborder[0],
                    np.array(nbox * column.h_box + column.z_off),
                )
                G.comment(f'+--- COLUMN #{index + 1}, TRENCH #{i_trc + 1} LEVEL {nbox + 1} ---+')

                # WALL
                G.load_program(str(wall_path))
                G.instruction(f'MSGDISPLAY 1, "COL {index + 1:03}, TR {i_trc + 1:03}, LV {nbox + 1:03}, W"\n')
                G.shutter('OFF')
                if column.u:
                    G.instruction(f'LINEAR U{column.u[0]:.6f}')
                    G.dwell(self.long_pause)
                G.move_to([float(x0), float(y0), float(z0)], speed_pos=column.speed_closed)

                G.instruction(f'$ZCURR = {z0:.6f}')
                G.shutter('ON')
                with G.repeat(column.n_repeat):
                    G.farcall(wall_filename)
                    G.instruction(f'$ZCURR = $ZCURR + {column.deltaz / super().neff:.6f}')
                    G.instruction('LINEAR Z$ZCURR')
                G.remove_program(wall_filename)

                # FLOOR
                G.shutter(state='OFF')
                G.load_program(str(floor_path))
                G.instruction(f'MSGDISPLAY 1, "COL {index + 1:03}, TR {i_trc + 1:03}, LV {nbox + 1:03}, F"\n')
                if column.u:
                    G.instruction(f'LINEAR U{column.u[-1]:.6f}')
                    G.dwell(self.long_pause)
                G.shutter(state='ON')
                G.farcall(floor_filename)
                G.shutter('OFF')
                if column.u:
                    G.instruction(f'LINEAR U{column.u[0]:.6f}')
                G.remove_program(floor_filename)
            G.instruction('MSGCLEAR -1\n')

    def _plot2d_trench(self, fig: go.Figure, style: dict[str, Any] | None = None) -> go.Figure:
        """2D plot helper.

        The function takes a figure and a style dictionary as inputs, and adds a trace to the figure for each ``Trench``
        stored in the ``Writer`` object.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure object to add the trench traces.
        style : dict
            Dictionary containing all the styling parameters of the trench traces.

        Returns
        -------
        go.Figure
            Input figure with added trench traces.

        See Also
        --------
        go.Figure : Plotly's figure object.
        go.Scattergl : Plotly's method to trace paths and lines.
        """

        if style is None:
            style = dict()
        default_tcargs = {'fillcolor': '#7E7E7E', 'mode': 'none', 'hoverinfo': 'none'}
        tcargs = {**default_tcargs, **style}

        for tr in self.trenches:
            # get points and transform them
            xt, yt = tr.border
            xt, yt, *_ = self.transform_points(xt, yt, np.zeros_like(xt, dtype=np.float32))

            fig.add_trace(
                go.Scattergl(
                    x=xt,
                    y=yt,
                    fill='toself',
                    **tcargs,
                    showlegend=False,
                    hovertemplate='(%{x:.4f}, %{y:.4f})<extra>TR</extra>',
                )
            )
        return fig


class WaveguideWriter(Writer):
    """Waveguide Writer class."""

    def __init__(self, wg_list: list[Waveguide | list[Waveguide]], **param) -> None:
        super().__init__(**param)
        self.obj_list: list[Waveguide | list[Waveguide]] = wg_list

        self._param: dict[str, Any] = dict(**param)
        self._export_path = self.CWD / (self.export_dir or '')
        self._fabtime: float = 0.0

    def append(self, obj: Waveguide) -> None:
        """Append Waveguide objects.

        Parameters
        ----------
        obj: Waveguide
            Waveguide object to be added to object list.

        Returns
        -------
        None
        """

        if not isinstance(obj, Waveguide):
            raise TypeError(f'The object must be a Waveguide. {type(obj).__name__} was given.')
        self.obj_list.append(obj)

    def extend(self, obj: list[Waveguide] | list[list[Waveguide]]) -> None:
        """Extend waveguide list.

        Extend the object list with a list of Waveguide objects.

        Parameters
        ----------
        obj : list(Waveguide)
            List of Waveguide objects to be added to object list.

        Returns
        -------
        None
        """

        if not isinstance(obj, list):
            raise TypeError(f'The object must be a list. {type(obj).__name__} was given.')
        if nest_level(obj) > 2:
            raise ValueError(
                f'The waveguide list has too many nested levels ({nest_level(obj)}). The maximum value is 2.'
            )
        if all(isinstance(wg, Waveguide) for wg in flatten(obj)):
            self.obj_list.extend(obj)
        else:
            raise TypeError('All the objects must be of type Waveguide.')

    def plot2d(
        self, fig: go.Figure | None = None, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """Plot 2D.

        2D plot of the Waveguide objects contained in ``self.obj_list``.

        Parameters
        ----------
        fig : go.Figure, optional
            Optional plotly figure object, if no figure is provided a new one is created and updated with the Waveguide
            objects stored in the Writer class. If a figure is provided it is updated by adding the stored objects. The
            default behaviour is creating a new figure.
        show_shutter_close : bool, optional
            Boolean flag, if ``True`` the movements with closed shutter are represented. The default value is ``False``.
        style: dict(str, Any)
            Plotly compatible styles options for representing the objects.

        Returns
        -------
        go.Figure
            Plotly figure with the 2D plot of the stored Waveguide.
        """

        # If existing figure is given as input parameter append to the figure and return it
        if fig is not None:
            return self._plot2d_wg(fig=fig, show_shutter_close=show_shutter_close, style=style)

        # If fig is None create a new figure from scratch
        fig = go.Figure()
        fig = self._plot2d_wg(fig=fig, show_shutter_close=show_shutter_close, style=style)
        fig = super().standard_2d_figure_update(fig)  # Add glass, origin and axis elements
        return fig

    def plot3d(
        self, fig: go.Figure | None = None, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """Plot 3D.

        3D plot of the Waveguide objects contained in ``self.obj_list``.

        Parameters
        ----------
        fig : go.Figure, optional
            Optional plotly figure object, if no figure is provided a new one is created and updated with the Waveguide
            objects stored in the Writer class. If a figure is provided it is updated by adding the stored objects. The
            default behaviour is creating a new figure.
        show_shutter_close : bool, optional
            Boolean flag, if ``True`` the movements with closed shutter are represented. The default value is ``False``.
        style: dict(str, Any)
            Plotly compatible styles options for representing the objects.

        Returns
        -------
        go.Figure
            Plotly figure with the 2D plot of the stored Waveguide.
        """

        # If existing figure is given as input parameter append to the figure and return it
        if fig is not None:
            return self._plot3d_wg(fig=fig, show_shutter_close=show_shutter_close, style=style)

        # If fig is None create a new figure from scratch
        fig = go.Figure()
        fig = self._plot3d_wg(fig=fig, show_shutter_close=show_shutter_close, style=style)
        fig = super().standard_3d_figure_update(fig)  # Add glass, origin and axis elements
        return fig

    def pgm(self, verbose: bool = True) -> None:
        """Export to PGM file.

        Function for the compilation of Waveguide objects. The function produces a *single file* containing all the
        instructions of the waveguides.

        Parameters
        ----------
        verbose : bool, optional
            Boolean flag, if ``True`` some information about the exporting procedures are printed. The default value is
            ``False``.

        Returns
        -------
        None

        See Also
        --------
        ``femto.pgmcompiler.PGMCompiler`` : class that convert lists of points to PGM file.
        """

        if not self.obj_list:
            return

        _wg_fab_time = 0.0
        _wg_param = dict(self._param.copy())
        _wg_param['filename'] = pathlib.Path(self.filename).stem + '_WG.pgm'

        with PGMCompiler(**_wg_param) as G:
            for bunch in self.obj_list:
                with G.repeat(listcast(bunch)[0].scan):
                    for wg in listcast(bunch):
                        _wg_fab_time += wg.fabrication_time
                        G.write(wg.points)
            G.go_init()
            _wg_fab_time += G._total_dwell_time
        del G

        if verbose:
            print('=' * 79)
            print('G-code compilation completed.')
            print(
                'Estimated fabrication time of the waveguides: \t',
                time.strftime('%H:%M:%S', time.gmtime(_wg_fab_time)),
            )
            print('=' * 79, '\n')
            self._fabtime = _wg_fab_time
        self._instructions.clear()

    def _plot2d_wg(
        self, fig: go.Figure, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """2D plot helper.

        The function takes a figure and a style dictionary as inputs, and adds a trace to the figure for each Waveguide
        stored in the ``Writer`` object.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure object to add the waveguide traces.
        style : dict
            Dictionary containing all the styling parameters of the waveguide.

        Returns
        -------
        go.Figure
            Input figure with added waveguide traces.

        See Also
        --------
        go.Figure : Plotly's figure object.
        go.Scattergl : Plotly's method to trace paths and lines.
        """

        if style is None:
            style = dict()
        default_wgargs = {'dash': 'solid', 'color': '#0000ff', 'width': 1.5}
        wg_args = {**default_wgargs, **style}
        sc_args = {'dash': 'dot', 'color': '#0000ff', 'width': 0.5}

        for wg in listcast(flatten(self.obj_list)):
            x_wg, y_wg, z_wg, _, s = wg.points
            x, y, z = self.transform_points(x_wg, y_wg, z_wg)
            xo = split_mask(x, s.astype(bool))
            yo = split_mask(y, s.astype(bool))
            zo = split_mask(z, s.astype(bool))
            [
                fig.add_trace(
                    go.Scattergl(
                        x=list(xoo),
                        y=list(yoo),
                        mode='lines',
                        line=wg_args,
                        showlegend=False,
                        customdata=zoo,
                        hovertemplate='(%{x:.4f}, %{y:.4f}, %{customdata:.4f})<extra>WG</extra>',
                    )
                )
                for xoo, yoo, zoo in zip(xo, yo, zo)
            ]
            if show_shutter_close:
                xc = split_mask(x, ~s.astype(bool))
                yc = split_mask(y, ~s.astype(bool))
                [
                    fig.add_trace(
                        go.Scattergl(
                            x=x,
                            y=y,
                            mode='lines',
                            line=sc_args,
                            hoverinfo='none',
                            showlegend=False,
                        )
                    )
                    for x, y in zip(xc, yc)
                ]
        return fig

    def _plot3d_wg(
        self, fig: go.Figure, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """3D plot helper.

        The function takes a figure and a style dictionary as inputs, and adds a 3D trace to the figure for each
        Waveguide stored in the ``Writer`` object.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure object to add the waveguide traces.
        style : dict
            Dictionary containing all the styling parameters of the waveguide.

        Returns
        -------
        go.Figure
            Input figure with added waveguide traces.

        See Also
        --------
        go.Figure : Plotly's figure object.
        go.Scatter3d : Plotly's method to trace paths and lines.
        """

        if style is None:
            style = dict()
        default_wgargs = {'dash': 'solid', 'color': '#0000ff', 'width': 1.5}
        wg_args = {**default_wgargs, **style}
        sc_args = {'dash': 'dot', 'color': '#0000ff', 'width': 0.5}

        for wg in listcast(flatten(self.obj_list)):
            x_wg, y_wg, z_wg, _, s = wg.points
            x, y, z = self.transform_points(x_wg, y_wg, z_wg)
            xo = split_mask(x, s.astype(bool))
            yo = split_mask(y, s.astype(bool))
            zo = split_mask(z, s.astype(bool))
            [
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode='lines',
                        line=wg_args,
                        showlegend=False,
                        hovertemplate='(%{x:.4f}, %{y:.4f}, %{z:.4f})<extra>WG</extra>',
                    )
                )
                for x, y, z in zip(xo, yo, zo)
            ]
            if show_shutter_close:
                xc = split_mask(x, ~s.astype(bool))
                yc = split_mask(y, ~s.astype(bool))
                zc = split_mask(z, ~s.astype(bool))
                [
                    fig.add_trace(
                        go.Scatter3d(
                            x=x,
                            y=y,
                            z=z,
                            mode='lines',
                            line=sc_args,
                            hoverinfo='none',
                            showlegend=False,
                        )
                    )
                    for x, y, z in zip(xc, yc, zc)
                ]
        return fig


class NasuWriter(Writer):
    """NasuWaveguide Writer class."""

    def __init__(self, nw_list: list[NasuWaveguide], **param) -> None:
        super().__init__(**param)
        self.obj_list: list[NasuWaveguide] = nw_list

        self._param: dict[str, Any] = dict(**param)
        self._export_path = self.CWD / (self.export_dir or '')
        self._fabtime: float = 0.0

    def append(self, obj: NasuWaveguide) -> None:
        """Append NasuWaveguide objects.

        Parameters
        ----------
        obj: NasuWaveguide
            NasuWaveguide object to be added to object list.

        Returns
        -------
        None
        """

        if not isinstance(obj, NasuWaveguide):
            raise TypeError(f'The object must be a NasuWaveguide. {type(obj).__name__} was given.')
        self.obj_list.append(obj)

    def extend(self, obj: list[NasuWaveguide]) -> None:
        """Extend Nasu waveguide list.

        Extend the object list with a list of NasuWaveguide objects.

        Parameters
        ----------
        obj : list(NasuWaveguide)
            List of NasuWaveguide objects to be added to object list.

        Returns
        -------
        None
        """

        if not isinstance(obj, list):
            raise TypeError(f'The object must be a list. {type(obj).__name__} was given.')
        if all(isinstance(wg, NasuWaveguide) for wg in flatten(obj)):
            self.obj_list.extend(obj)
        else:
            raise TypeError('All the objects must be of type NasuWaveguide.')

    def plot2d(
        self, fig: go.Figure | None = None, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """Plot 2D.

        2D plot of the NasuWaveguide objects contained in ``self.obj_list``.

        Parameters
        ----------
        fig : go.Figure, optional
            Optional plotly figure object, if no figure is provided a new one is created and updated with the
            NasuWaveguide objects stored in the Writer class. If a figure is provided it is updated by adding the
            stored objects. The default behaviour is creating a new figure.
        show_shutter_close : bool, optional
            Boolean flag, if ``True`` the movements with closed shutter are represented. The default value is ``False``.
        style: dict(str, Any)
            Plotly compatible styles options for representing the objects.

        Returns
        -------
        go.Figure
            Plotly figure with the 2D plot of the stored NasuWaveguide.
        """

        # If existing figure is given as input parameter append to the figure and return it
        if fig is not None:
            return self._plot2d_nwg(fig=fig, show_shutter_close=show_shutter_close, style=style)

        # If fig is None create a new figure from scratch
        fig = go.Figure()
        fig = self._plot2d_nwg(fig=fig, show_shutter_close=show_shutter_close, style=style)
        fig = super().standard_2d_figure_update(fig)  # Add glass, origin and axis elements
        return fig

    def plot3d(
        self, fig: go.Figure | None = None, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """Plot 3D.

        3D plot of the NasuWaveguide objects contained in ``self.obj_list``.

        Parameters
        ----------
        fig : go.Figure, optional
            Optional plotly figure object, if no figure is provided a new one is created and updated with the
            NasuWaveguide objects stored in the Writer class. If a figure is provided it is updated by adding the
            stored objects. The default behaviour is creating a new figure.
        show_shutter_close : bool, optional
            Boolean flag, if ``True`` the movements with closed shutter are represented. The default value is ``False``.
        style: dict(str, Any)
            Plotly compatible styles options for representing the objects.

        Returns
        -------
        go.Figure
            Plotly figure with the 2D plot of the stored NasuWaveguide.
        """

        # If existing figure is given as input parameter append to the figure and return it
        if fig is not None:
            return self._plot3d_nwg(fig=fig, show_shutter_close=show_shutter_close, style=style)

        # If fig is None create a new figure from scratch
        fig = go.Figure()
        fig = self._plot3d_nwg(fig=fig, show_shutter_close=show_shutter_close, style=style)
        fig = super().standard_3d_figure_update(fig)  # Add glass, origin and axis elements
        return fig

    def pgm(self, verbose: bool = True) -> None:
        """Export to PGM file.

        Function for the compilation of NasuWaveguide objects. The function produces a *single file* containing all the
        instructions of the waveguides.

        Parameters
        ----------
        verbose : bool, optional
            Boolean flag, if ``True`` some information about the exporting procedures are printed. The default value is
            ``False``.

        Returns
        -------
        None

        See Also
        --------
        ``femto.pgmcompiler.PGMCompiler`` : class that convert lists of points to PGM file.
        """

        if not self.obj_list:
            return

        _nwg_fab_time = 0.0
        _nwg_param = dict(self._param.copy())
        _nwg_param['filename'] = pathlib.Path(self.filename).stem + '_NASU.pgm'

        with PGMCompiler(**_nwg_param) as G:
            for nwg in self.obj_list:
                for shift in nwg.adj_scan_order:
                    _nwg_fab_time += nwg.fabrication_time
                    dx, dy, dz = nwg.adj_scan_shift
                    coord_shift = np.array([dx, dy, dz, 0, 0]).reshape(-1, 1)
                    G.write(nwg.points + shift * coord_shift)
            G.go_init()
            _nwg_fab_time += G._total_dwell_time
        del G

        if verbose:
            print('=' * 79)
            print('G-code compilation completed.')
            print(
                'Estimated fabrication time for the Nasu waveguides: \t',
                time.strftime('%H:%M:%S', time.gmtime(_nwg_fab_time)),
            )
            print('=' * 79, '\n')
            self._fabtime = _nwg_fab_time
        self._instructions.clear()

    def _plot2d_nwg(
        self, fig: go.Figure, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """2D plot helper.

        The function takes a figure and a style dictionary as inputs, and adds a trace to the figure for each
        NasuWaveguide stored in the ``Writer`` object.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure object to add the Nasu waveguide traces.
        style : dict
            Dictionary containing all the styling parameters of the Nasu waveguide.

        Returns
        -------
        go.Figure
            Input figure with added Nasu waveguide traces.

        See Also
        --------
        go.Figure : Plotly's figure object.
        go.Scattergl : Plotly's method to trace paths and lines.
        """

        if style is None:
            style = dict()
        default_wgargs = {'dash': 'solid', 'color': '#0000ff', 'width': 1.5}
        wg_args = {**default_wgargs, **style}
        sc_args = {'dash': 'dot', 'color': '#0000ff', 'width': 0.5}

        for nwg in listcast(flatten(self.obj_list)):
            for shift in nwg.adj_scan_order:
                dx, dy, dz = nwg.adj_scan_shift
                coord_shift = np.array([dx, dy, dz, 0, 0]).reshape(-1, 1)
                x_wg, y_wg, z_wg, _, s = nwg.points + shift * coord_shift
                x, y, _ = self.transform_points(x_wg, y_wg, z_wg)
                xo = split_mask(x, s.astype(bool))
                yo = split_mask(y, s.astype(bool))
                [
                    fig.add_trace(
                        go.Scattergl(
                            x=list(xoo),
                            y=list(yoo),
                            mode='lines',
                            line=wg_args,
                            showlegend=False,
                            hovertemplate='(%{x:.4f}, %{y:.4f})<extra>WG</extra>',
                        )
                    )
                    for xoo, yoo in zip(xo, yo)
                ]
                if show_shutter_close:
                    xc = split_mask(x, ~s.astype(bool))
                    yc = split_mask(y, ~s.astype(bool))
                    [
                        fig.add_trace(
                            go.Scattergl(
                                x=x,
                                y=y,
                                mode='lines',
                                line=sc_args,
                                hoverinfo='none',
                                showlegend=False,
                            )
                        )
                        for x, y in zip(xc, yc)
                    ]
        return fig

    def _plot3d_nwg(
        self, fig: go.Figure, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """3D plot helper.

        The function takes a figure and a style dictionary as inputs, and adds a 3D trace to the figure for each
        NasuWaveguide stored in the ``Writer`` object.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure object to add the Nasu waveguide traces.
        style : dict
            Dictionary containing all the styling parameters of the Nasu waveguide.

        Returns
        -------
        go.Figure
            Input figure with added waveguide traces.

        See Also
        --------
        go.Figure : Plotly's figure object.
        go.Scatter3d : Plotly's method to trace paths and lines.
        """

        if style is None:
            style = dict()
        default_wgargs = {'dash': 'solid', 'color': '#0000ff', 'width': 1.5}
        wg_args = {**default_wgargs, **style}
        sc_args = {'dash': 'dot', 'color': '#0000ff', 'width': 0.5}

        for nwg in listcast(flatten(self.obj_list)):
            for shift in nwg.adj_scan_order:
                dx, dy, dz = nwg.adj_scan_shift
                coord_shift = np.array([dx, dy, dz, 0, 0]).reshape(-1, 1)
                x_wg, y_wg, z_wg, _, s = nwg.points + shift * coord_shift
                x, y, z = self.transform_points(x_wg, y_wg, z_wg)
                xo = split_mask(x, s.astype(bool))
                yo = split_mask(y, s.astype(bool))
                zo = split_mask(z, s.astype(bool))
                [
                    fig.add_trace(
                        go.Scatter3d(
                            x=x,
                            y=y,
                            z=z,
                            mode='lines',
                            line=wg_args,
                            showlegend=False,
                            hovertemplate='(%{x:.4f}, %{y:.4f}, %{z:.4f})<extra>WG</extra>',
                        )
                    )
                    for x, y, z in zip(xo, yo, zo)
                ]
                if show_shutter_close:
                    xc = split_mask(x, ~s.astype(bool))
                    yc = split_mask(y, ~s.astype(bool))
                    zc = split_mask(z, ~s.astype(bool))
                    [
                        fig.add_trace(
                            go.Scatter3d(
                                x=x,
                                y=y,
                                z=z,
                                mode='lines',
                                line=sc_args,
                                hoverinfo='none',
                                showlegend=False,
                            )
                        )
                        for x, y, z in zip(xc, yc, zc)
                    ]
        return fig


class MarkerWriter(Writer):
    """Marker Writer class."""

    def __init__(self, mk_list: list[Marker], **param) -> None:
        super().__init__(**param)
        self.obj_list: list[Marker] = flatten(mk_list)

        self._param: dict[str, Any] = dict(**param)
        self._export_path = self.CWD / (self.export_dir or '')
        self._fabtime: float = 0.0

    def append(self, obj: Marker) -> None:
        """Append Marker objects.

        Parameters
        ----------
        obj: Marker
            Marker object to be added to object list.

        Returns
        -------
        None
        """

        if not isinstance(obj, Marker):
            raise TypeError(f'The object must be a Marker. {type(obj).__name__} was given.')
        self.obj_list.append(obj)

    def extend(self, obj: list[Marker]) -> None:
        """Extend Marker list.

        Extend the object list with a list of Marker objects.

        Parameters
        ----------
        obj : list(Marker)
            List of Marker objects to be added to object list.

        Returns
        -------
        None
        """

        if not isinstance(obj, list):
            raise TypeError(f'The object must be a list. {type(obj).__name__} was given.')
        for mk in flatten(obj):
            self.append(mk)

    def plot2d(
        self, fig: go.Figure | None = None, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """Plot 2D.

        2D plot of the Marker objects contained in ``self.obj_list``.

        Parameters
        ----------
        fig : go.Figure, optional
            Optional plotly figure object, if no figure is provided a new one is created and updated with the Marker
            objects stored in the Writer class. If a figure is provided it is updated by adding the stored objects. The
            default behaviour is creating a new figure.
        show_shutter_close : bool, optional
            Boolean flag, if ``True`` the movements with closed shutter are represented. The default value is ``False``.
        style: dict(str, Any)
            Plotly compatible styles options for representing the objects.

        Returns
        -------
        go.Figure
            Plotly figure with the 2D plot of the stored Marker.
        """

        # If existing figure is given as input parameter append to the figure and return it
        if fig is not None:
            return self._plot2d_mk(fig=fig, style=style)

        # If fig is None create a new figure from scratch
        fig = go.Figure()
        fig = self._plot2d_mk(fig=fig, style=style)
        fig = super().standard_2d_figure_update(fig)  # Add glass, origin and axis elements
        return fig

    def plot3d(
        self, fig: go.Figure | None = None, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """Plot 3D.

        3D plot of the Marker objects contained in ``self.obj_list``.

        Parameters
        ----------
        fig : go.Figure, optional
            Optional plotly figure object, if no figure is provided a new one is created and updated with the Marker
            objects stored in the Writer class. If a figure is provided it is updated by adding the stored objects. The
            default behaviour is creating a new figure.
        show_shutter_close : bool, optional
            Boolean flag, if ``True`` the movements with closed shutter are represented. The default value is ``False``.
        style: dict(str, Any)
            Plotly compatible styles options for representing the objects.

        Returns
        -------
        go.Figure
            Plotly figure with the 2D plot of the stored Marker.
        """

        # If existing figure is given as input parameter append to the figure and return it
        if fig is not None:
            return self._plot3d_mk(fig=fig, style=style)

        # If fig is None create a new figure from scratch
        fig = go.Figure()
        fig = self._plot3d_mk(fig=fig, style=style)
        fig = super().standard_3d_figure_update(fig)  # Add glass, origin and axis elements
        return fig

    def pgm(self, verbose: bool = True) -> None:
        """Export to PGM file.

        Function for the compilation of Marker objects. The function produces a *single file* containing all the
        instructions of the markers and ablations.

        Parameters
        ----------
        verbose : bool, optional
            Boolean flag, if ``True`` some information about the exporting procedures are printed. The default value is
            ``False``.

        Returns
        -------
        None

        See Also
        --------
        ``femto.pgmcompiler.PGMCompiler`` : class that convert lists of points to PGM file.
        """

        if not self.obj_list:
            return

        _mk_fab_time = 0.0
        _mk_param = dict(self._param.copy())
        _mk_param['filename'] = pathlib.Path(self.filename).stem + '_MK.pgm'

        with PGMCompiler(**_mk_param) as G:
            for idx, mk in enumerate(flatten(self.obj_list)):
                with G.repeat(mk.scan):
                    _mk_fab_time += mk.fabrication_time
                    G.comment(f'MARKER {idx + 1}')
                    G.write(mk.points)
                    G.comment('')
            G.go_origin()
            _mk_fab_time += G._total_dwell_time
        del G

        if verbose:
            print('=' * 79)
            print('G-code compilation completed.')
            print(
                'Estimated fabrication time of the markers: \t',
                time.strftime('%H:%M:%S', time.gmtime(_mk_fab_time)),
            )
            print('=' * 79, '\n')
        self._instructions.clear()
        self._total_dwell_time = 0.0
        self._fabtime = _mk_fab_time

    def _plot2d_mk(self, fig: go.Figure, style: dict[str, Any] | None = None) -> go.Figure:
        """2D plot helper.

        The function takes a figure and a style dictionary as inputs, and adds a trace to the figure for each
        Marker stored in the ``Writer`` object.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure object to add the marker traces.
        style : dict
            Dictionary containing all the styling parameters of the marker.

        Returns
        -------
        go.Figure
            Input figure with added marker traces.

        See Also
        --------
        go.Figure : Plotly's figure object.
        go.Scattergl : Plotly's method to trace paths and lines.
        """

        if style is None:
            style = dict()
        default_mkargs = {'dash': 'solid', 'color': '#000000', 'width': 2.0}
        mk_args = {**default_mkargs, **style}

        for mk in listcast(flatten(self.obj_list)):
            x_wg, y_wg, z_wg, _, s = mk.points
            x, y, z = self.transform_points(x_wg, y_wg, z_wg)
            xo = split_mask(x, s.astype(bool))
            yo = split_mask(y, s.astype(bool))
            [
                fig.add_trace(
                    go.Scattergl(
                        x=x,
                        y=y,
                        mode='lines',
                        line=mk_args,
                        showlegend=False,
                        hovertemplate='(%{x:.4f}, %{y:.4f})<extra>MK</extra>',
                    )
                )
                for x, y in zip(xo, yo)
            ]
        return fig

    def _plot3d_mk(self, fig: go.Figure, style: dict[str, Any] | None = None) -> go.Figure:
        """3D plot helper.

        The function takes a figure and a style dictionary as inputs, and adds a 3D trace to the figure for each
        Marker stored in the ``Writer`` object.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure object to add the marker traces.
        style : dict
            Dictionary containing all the styling parameters of the marker.

        Returns
        -------
        go.Figure
            Input figure with added marker traces.

        See Also
        --------
        go.Figure : Plotly's figure object.
        go.Scatter3d : Plotly's method to trace paths and lines.
        """

        if style is None:
            style = dict()
        default_mkargs = {'dash': 'solid', 'color': '#000000', 'width': 2.0}
        mk_args = {**default_mkargs, **style}

        for mk in listcast(flatten(self.obj_list)):
            x_wg, y_wg, z_wg, _, s = mk.points
            x, y, z = self.transform_points(x_wg, y_wg, z_wg)
            xo = split_mask(x, s.astype(bool))
            yo = split_mask(y, s.astype(bool))
            zo = split_mask(z, s.astype(bool))
            [
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode='lines',
                        line=mk_args,
                        showlegend=False,
                        hovertemplate='<extra>MK</extra>',
                    )
                )
                for x, y, z in zip(xo, yo, zo)
            ]
        return fig


def main() -> None:
    from femto.waveguide import NasuWaveguide

    # Data
    PARAM_WG: dict[str, Any] = dict(scan=6, speed=20, radius=15, pitch=0.080, int_dist=0.007, lsamplesize=(10, 3))
    PARAM_GC: dict[str, Any] = dict(filename='testPGM.pgm', samplesize=PARAM_WG['samplesize'])

    increment = [5.0, 0, 0]

    # Calculations
    mzi = []
    for index in range(2):
        wg = NasuWaveguide(adj_scan_shift=(0, 0.004, 0), **PARAM_WG)
        wg.y_init = -wg.pitch / 2 + index * wg.pitch
        wg.start()
        wg.linear(increment)
        wg.sin_mzi((-1) ** index * wg.dy_bend)
        wg.sin_bridge((-1) ** index * 0.08, (-1) ** index * 0.015)
        wg.arc_bend((-1) ** (index + 1) * wg.dy_bend)
        wg.linear(increment)
        wg.end()
        mzi.append(wg)

    nwr = NasuWriter(mzi, **PARAM_GC)
    fig = nwr.plot3d()
    fig.show()


if __name__ == '__main__':
    main()
