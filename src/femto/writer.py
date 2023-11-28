from __future__ import annotations

import abc
import copy
import datetime
import itertools
import pathlib
from typing import Any

import dill
import numpy as np
import numpy.typing as npt
from femto import logger
from femto.helpers import flatten
from femto.helpers import listcast
from femto.helpers import nest_level
from femto.helpers import split_mask
from femto.marker import Marker
from femto.pgmcompiler import PGMCompiler
from femto.trench import Trench
from femto.trench import TrenchColumn
from femto.trench import UTrenchColumn
from femto.waveguide import NasuWaveguide
from femto.waveguide import Waveguide
from plotly import graph_objs as go


class Writer(PGMCompiler, abc.ABC):
    """Abstract class representing a G-Code Writer object.

    A Writer class is a super-object that can store homogeneous objects (Waveguides, Markers, Trenches,
    etc.) and provides methods to append objects, plot and export them as .pgm files.
    """

    @property
    @abc.abstractmethod
    def objs(self) -> list[Any]:
        """Objects.

        Abstract property for returning the objects contained inside a given Writer.

        Returns
        -------
        list
           List of objects contained in the current Writer.
        """
        pass

    @property
    @abc.abstractmethod
    def fab_time(self) -> float:
        """Fabrication time.

        Abstract property for returning the total fabrication time for objects in a given Writer.

        Returns
        -------
        float
           Total fabrication time [s].
        """
        pass

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
    def pgm(self, verbose: bool = False) -> None:
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
        shift_origin point and formats the axis.

        Parameters
        ----------
        fig : go.Figure
            Plotly Figure object to update.

        Returns
        -------
        go.Figure
            Input figure updated with sample glass shape, shift_origin point and axis.
        """

        # GLASS
        logger.debug('Add glass shape.')
        fig.add_shape(
            type='rect',
            x0=self.shift_origin[0] if self.flip_x else 0 - self.shift_origin[0],
            y0=self.shift_origin[1] if self.flip_y else 0 - self.shift_origin[1],
            x1=self.shift_origin[0] - self.xsample if self.flip_x else self.xsample - self.shift_origin[0],
            y1=self.shift_origin[1] - self.ysample if self.flip_y else self.ysample - self.shift_origin[1],
            fillcolor='#D0FAF9',
            line_color='#000000',
            line_width=2,
            layer='below',
        )

        # ORIGIN
        logger.debug('Add origin reference.')
        fig.add_trace(
            go.Scattergl(
                x=[0],
                y=[0],
                marker=dict(color='red', size=12),
                hoverinfo='none',
                showlegend=False,
            )
        )

        logger.debug('Add axis labels.')
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

        Helper function that update a 3D plot by adding the sample glass, the `(0, 0, 0)` shift_origin point and formats
        the axis.

        Parameters
        ----------
        fig : go.Figure
            Plotly Figure object to update.

        Returns
        -------
        go.Figure
            Input figure updated with sample glass shape, shift_origin point and axis.
        """
        # ORIGIN
        logger.debug('Add origin reference.')
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

        logger.debug('Add axis labels and fix camera angle.')
        fig.update_layout(
            scene_camera=dict(up=dict(x=0, y=np.cos(45), z=np.sin(45)), eye=dict(x=0.0, y=0.0, z=-2.5)),
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
            ),
        )
        return fig

    def export(
        self,
        filename: str | pathlib.Path | None = None,
        export_path: str | pathlib.Path | None = None,
        export_dir: str | pathlib.Path = 'EXPORT',
    ):
        """Export Nasu Waveguide objects.

        The export the list of Nasu Waveguide objects into a pickle file. The objects are exported in an ``EXPORT``
        directory.

        export_dir: str, optional
            Name of the directory inside which export objects.

        Returns
        -------
            None
        """

        fn = filename if filename is not None else self.filename
        exp_p = self.CWD if export_path is None else export_dir

        filepath = pathlib.Path(exp_p) / export_dir / pathlib.Path(fn).stem
        filepath.mkdir(exist_ok=True, parents=True)

        for i, el in enumerate(self.objs):
            objpath = filepath / f'{el.id}_{i + 1:02}.pkl'
            with open(objpath, 'wb') as f:
                dill.dump(el, f)


class TrenchWriter(Writer):
    """Trench Writer class."""

    def __init__(
        self,
        param: dict[str, Any],
        objects: TrenchColumn | list[TrenchColumn] | None = None,
        dirname: str = 'TRENCH',
        **kwargs,
    ) -> None:

        p = copy.deepcopy(param)
        p.update(kwargs)

        super().__init__(**p)
        self.dirname: str = dirname

        self._obj_list: list[TrenchColumn] = [] if objects is None else flatten(listcast(objects))
        self._trenches: list[Trench] = [tr for col in self._obj_list for tr in col]
        self._param: dict[str, Any] = p
        self._export_path = self.CWD / (self.export_dir or '') / (self.dirname or '')
        self._fabtime: float = 0.0

    @property
    def objs(self) -> list[Any]:
        """TrenchColumn objects.

        Property for returning the TrenchColumn objects contained inside a TrenchWriter.

        Returns
        -------
        list
           List of TrenchColumn objects.
        """
        return self._obj_list

    @property
    def trench_list(self) -> list[Any]:
        # TODO: test this
        """Trench objects.

        Property for returning the Trench objects contained inside a TrenchWriter.

        Returns
        -------
        list
           List of Trench objects.
        """
        return self._trenches

    @property
    def fab_time(self) -> float:
        """Trench fabrication time.

        Returns
        -------
        float
           Total trench fabrication time [s].
        """
        logger.debug(f'Return fabrication time = {self._fabtime}.')
        return self._fabtime

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
            logger.error(f'The object must be a TrenchColumn. {type(obj).__name__} was given.')
            raise TypeError(f'The object must be a TrenchColumn. {type(obj).__name__} was given.')
        self._obj_list.append(obj)
        self._trenches.extend(obj.trench_list)
        logger.debug('Append Trench to _obj_list.')

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
            logger.error(f'The object must be a list. {type(obj).__name__} was given.')
            raise TypeError(f'The object must be a list. {type(obj).__name__} was given.')
        for tr_col in flatten(obj):
            self.append(tr_col)

    def plot2d(
        self, fig: go.Figure | None = None, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """Plot 2D.

        2D plot of the Trench objects contained in ``self._obj_list``.

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
            logger.debug('Update figure.')
            return self._plot2d_trench(fig=fig, style=style)

        # If fig is None create a new figure from scratch
        logger.debug('Create figure.')
        fig = go.Figure()
        logger.debug('Update figure.')
        fig = self._plot2d_trench(fig=fig, style=style)
        fig = super().standard_2d_figure_update(fig)  # Add glass, shift_origin and axis elements
        return fig

    def plot3d(
        self,
        fig: go.Figure | None = None,
        show_shutter_close: bool = True,
        style: dict[str, Any] | None = None,
    ) -> go.Figure:
        """Plot 3D - Not implemented."""
        # raise NotImplementedError()
        if fig is not None:
            logger.debug('Update figure.')
            return self._plot3d_trench(fig=fig, style=style)

        # If fig is None create a new figure from scratch
        logger.debug('Create figure.')
        fig = go.Figure()
        logger.debug('Update figure.')
        fig = self._plot3d_trench(fig=fig, style=style)
        fig = super().standard_2d_figure_update(fig)  # Add glass, shift_origin and axis elements
        return fig

    def pgm(self, verbose: bool = False) -> None:
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

        if not self._obj_list:
            logger.debug('No object in current writer, no PGM file created.')
            return None

        # Export each Trench for each TrenchColumn
        for i_col, col in enumerate(self._obj_list):
            # Prepare PGM files export directory
            col_dir = self._export_path / f'trenchCol{i_col + 1:03}'
            col_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f'Export trench column to "{col_dir}".')

            # Export walls and floors as .pgm files
            self._export_trench_column(column=col, column_path=col_dir)

            # Create FARCALL.pgm file for all trenches in current column
            self._farcall_trench_column(column=col, index=i_col)

        if len(self._obj_list) > 1:
            # Create a MAIN farcall file, calls all columns .pgm scripts
            main_param = dict(self._param.copy())
            main_param['filename'] = self._export_path / 'MAIN.pgm'
            main_param['aerotech_angle'] = None
            main_param['rotation_angle'] = None
            main_param['verbose'] = verbose

            logger.debug('Generate MAIN.pgm file.')
            farcall_list = [
                str(pathlib.Path(col.base_folder) / f'FARCALL{i + 1:03}.pgm') for i, col in enumerate(self._obj_list)
            ]
            fn = self._export_path / 'MAIN.pgm'
            with PGMCompiler.from_dict(self._param, filename=fn, aerotech_angle=None, rotation_angle=None) as G:
                G.farcall_list(farcall_list)

        _tc_fab_time = self._fabtime
        for col in self._obj_list:
            _tc_fab_time += col.fabrication_time + 10
        self._fabtime = _tc_fab_time
        string = '{:.<49} {}'.format(
            'Estimated isolation trenches fabrication time: ', datetime.timedelta(seconds=int(self._fabtime))
        )
        logger.info(string)

    def export_array2d(
        self,
        filename: pathlib.Path,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        speed: float | list[float],
        forced_deceleration: bool | list[bool] | npt.NDArray[np.bool_] = False,
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
        speed : float | list[float]
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
            logger.error('No filename given.')
            raise ValueError('No filename given.')

        # Transform points
        x_arr, y_arr, _ = self.transform_points(x, y, np.zeros_like(x))
        z_arr = [None]

        # Export points
        instr = [
            self._format_args(x, y, z, f)
            for (x, y, z, f) in itertools.zip_longest(x_arr, y_arr, z_arr, listcast(speed))
        ]

        gcode_instr = []
        for (line, dec) in itertools.zip_longest(instr, listcast(forced_deceleration)):
            if bool(dec):
                gcode_instr.append(f'G9 G1 {line}\n')
            else:
                gcode_instr.append(f'G1 {line}\n')

        logger.debug('Export 2D path.')
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
            logger.debug(f'Export trench wall (border) for trench {trench}.')
            self.export_array2d(
                filename=column_path / f'trench{i + 1:03}_WALL.pgm',
                x=x_wall,
                y=y_wall,
                speed=column.speed_wall,
            )

            # Floor script
            x_floor = np.array([])
            y_floor = np.array([])
            f_decel = np.array([])

            for idx, (x_temp, y_temp) in enumerate(trench.toolpath()):
                if idx >= trench.num_insets:
                    f_temp = np.ones_like(x_temp, dtype=bool)
                else:
                    f_temp = np.empty_like(x_temp, dtype=object)
                    f_temp[0], f_temp[-1] = True, True

                x_floor = np.append(x_floor, x_temp)
                y_floor = np.append(y_floor, y_temp)
                f_decel = np.append(f_decel, f_temp)

            logger.debug(f'Export trench floor for trench {trench}.')
            self.export_array2d(
                filename=column_path / f'trench{i + 1:03}_FLOOR.pgm',
                x=x_floor,
                y=y_floor,
                speed=column.speed_floor,
                forced_deceleration=f_decel,
            )

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

        logger.debug('Generate Trench columns FARCALL.pgm file.')
        column_param = dict(self._param.copy())
        column_param['filename'] = self._export_path / f'FARCALL{index + 1:03}.pgm'
        with PGMCompiler.from_dict(column_param) as G:
            G.dvar(['ZCURR'])

            for nbox, (i_trc, trench) in list(itertools.product(range(column.nboxz), list(enumerate(column)))):
                # load filenames (wall/floor)
                wall_filename = f'trench{i_trc + 1:03}_WALL.pgm'
                floor_filename = f'trench{i_trc + 1:03}_FLOOR.pgm'
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
                    G.instruction(f'G1 U{column.u[0]:.6f}')
                    G.dwell(self.long_pause)
                G.move_to([float(x0), float(y0), float(z0)], speed_pos=column.speed_closed)

                G.instruction(f'$ZCURR = {z0:.6f}')
                G.shutter(state='ON')
                with G.repeat(column.n_repeat):
                    G.farcall(wall_filename)
                    G.instruction(f'$ZCURR = $ZCURR + {column.deltaz / super().neff:.6f}')
                    G.instruction('G1 Z$ZCURR')
                G.remove_program(wall_filename)

                # FLOOR
                G.shutter(state='OFF')
                G.load_program(str(floor_path))
                G.instruction(f'MSGDISPLAY 1, "COL {index + 1:03}, TR {i_trc + 1:03}, LV {nbox + 1:03}, F"\n')
                if column.u:
                    G.instruction(f'G1 U{column.u[-1]:.6f}')
                    G.dwell(self.long_pause)
                G.shutter(state='ON')
                G.farcall(floor_filename)
                G.shutter(state='OFF')
                if column.u:
                    G.instruction(f'G1 U{column.u[0]:.6f}')
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

        logger.debug('Add trenches shapes to figure.')
        for tr in self._trenches:
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

    def _plot3d_trench(self, fig: go.Figure, style: dict[str, Any] | None = None) -> go.Figure:
        if style is None:
            style = dict()
        default_tcargs = {'dash': 'solid', 'color': '#000000', 'width': 1.5}
        tcargs = {**default_tcargs, **style}

        logger.debug('Add trenches 3D profile to figure.')
        for tr in self._trenches:
            # get points and transform them
            xt, yt = tr.border
            xt, yt, *_ = self.transform_points(xt, yt, np.zeros_like(xt, dtype=np.float32))

            x = np.array([xt, xt])
            y = np.array([yt, yt])
            z = np.array([np.zeros_like(xt), tr.height * np.ones_like(xt)])
            fig.add_trace(
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    colorscale=[[0, 'grey'], [1, 'grey']],
                    showlegend=False,
                    hoverinfo='skip',
                    showscale=False,
                    opacity=0.6,
                )
            )
            for zval in [0.0, tr.height]:
                fig.add_trace(
                    go.Scatter3d(
                        x=xt,
                        y=yt,
                        z=zval * np.ones_like(xt),
                        mode='lines',
                        line=tcargs,
                        showlegend=False,
                        hovertemplate='(%{x:.4f}, %{y:.4f})<extra>TR</extra>',
                    )
                )
        return fig


class UTrenchWriter(TrenchWriter):
    """U-Trench Writer class."""

    def __init__(
        self,
        param: dict[str, Any],
        objects: UTrenchColumn | list[UTrenchColumn] | None = None,
        dirname: str = 'U-TRENCH',
        **kwargs,
    ) -> None:
        obj_list: list[UTrenchColumn] = flatten(listcast(objects)) if objects is not None else []
        super().__init__(param, objects=obj_list, dirname=dirname, **kwargs)
        self.beds: list[Trench] = [ubed for col in obj_list for ubed in col.trench_bed]

    @property
    def objs(self) -> list[Any]:
        """UTrenchColumn objects.

        Property for returning the UTrenchColumn objects contained inside a UTrenchWriter.

        Returns
        -------
        list
           List of UTrenchColumn objects.
        """
        return self._obj_list

    @property
    def fab_time(self) -> float:
        """U-Trench fabrication time.

        Returns
        -------
        float
           Total U-trench fabrication time [s].
        """
        logger.debug(f'Return fabrication time = {self._fabtime}.')
        return self._fabtime

    def append(self, obj: UTrenchColumn) -> None:
        """Append UTrenchColumn objects.

        Parameters
        ----------
        obj: UTrenchColumn
            UTrenchColumn object to be added to object list.

        Returns
        -------
        None
        """

        if not isinstance(obj, UTrenchColumn):
            logger.error(f'The object must be a UTrenchColumn. {type(obj).__name__} was given.')
            raise TypeError(f'The object must be a UTrenchColumn. {type(obj).__name__} was given.')
        self._obj_list.append(obj)
        self._trenches.extend(obj.trench_list)
        logger.debug('Append U-Trench to _obj_list.')
        self.beds.extend(obj.trench_bed)
        logger.debug('Append Trench bed to _trenchbed.')

    # Private interface
    def _export_trench_column(self, column: UTrenchColumn, column_path: pathlib.Path) -> None:
        """Export Trench columns to PGM file.

        Helper function that exports the wall and floor scripts for each trench in a Trench Column.

        Parameters
        ----------
        column : UTrenchColumn
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

        super()._export_trench_column(column=column, column_path=column_path)

        # Bed script
        for i, bed_block in enumerate(column.trench_bed):
            logger.debug(f'Export trench bed for {bed_block}.')

            x_bed_block = np.array([])
            y_bed_block = np.array([])
            f_decel = np.array([])

            for idx, (x_temp, y_temp) in enumerate(bed_block.toolpath()):
                if idx == bed_block.num_insets:
                    f_temp = np.ones_like(x_temp, dtype=bool)
                else:
                    f_temp = np.empty_like(x_temp, dtype=object)
                    f_temp[0], f_temp[-1] = True, True

                x_bed_block = np.append(x_bed_block, x_temp)
                y_bed_block = np.append(y_bed_block, y_temp)
                f_decel = np.append(f_decel, f_temp)

            self.export_array2d(
                filename=column_path / f'trench_BED_{i + 1:03}.pgm',
                x=x_bed_block,
                y=y_bed_block,
                speed=column.speed_floor,
                forced_deceleration=f_decel,
            )

    def _farcall_trench_column(self, column: UTrenchColumn, index: int) -> None:
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

        logger.debug('Generate U-Trench columns FARCALL.pgm file.')
        column_param = dict(self._param.copy())
        column_param['filename'] = self._export_path / f'FARCALL{index + 1:03}.pgm'
        with PGMCompiler.from_dict(column_param) as G:
            G.dvar(['ZCURR'])

            for nbox, (i_trc, trench) in list(itertools.product(range(column.nboxz), list(enumerate(column)))):
                # load filenames (wall/floor)
                wall_filename = f'trench{i_trc + 1:03}_WALL.pgm'
                floor_filename = f'trench{i_trc + 1:03}_FLOOR.pgm'
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
                G.shutter(state='OFF')
                if column.u:
                    G.instruction(f'G1 U{column.u[0]:.6f}')
                    G.dwell(self.long_pause)
                G.move_to([float(x0), float(y0), float(z0)], speed_pos=column.speed_closed)

                G.instruction(f'$ZCURR = {z0:.6f}')
                G.shutter(state='ON')
                with G.repeat(column.n_repeat):
                    G.farcall(wall_filename)
                    G.instruction(f'$ZCURR = $ZCURR + {column.deltaz / super().neff:.6f}')
                    G.instruction('G1 Z$ZCURR')
                G.remove_program(wall_filename)

                # FLOOR
                G.shutter(state='OFF')
                G.load_program(str(floor_path))
                G.instruction(f'MSGDISPLAY 1, "COL {index + 1:03}, TR {i_trc + 1:03}, LV {nbox + 1:03}, F"\n')
                if column.u:
                    G.instruction(f'G1 U{column.u[-1]:.6f}')
                    G.dwell(self.long_pause)
                G.shutter(state='ON')
                G.farcall(floor_filename)
                G.shutter(state='OFF')
                if column.u:
                    G.instruction(f'G1 U{column.u[0]:.6f}')
                G.remove_program(floor_filename)

            # BED
            for i_bed, bed_block in enumerate(column.trench_bed):
                x0, y0, z0 = self.transform_points(
                    np.array(bed_block.block.exterior.coords.xy[0])[0],
                    np.array(bed_block.block.exterior.coords.xy[1])[0],
                    np.array(nbox * column.h_box + column.z_off),
                )
                # load filenames (beds)
                bed_filename = f'trench_BED_{i_bed + 1:03}.pgm'
                bed_path = pathlib.Path(column.base_folder) / f'trenchCol{index + 1:03}' / bed_filename

                G.comment(f'+--- COLUMN #{index + 1}, TRENCH BED #{i_bed + 1} ---+')

                G.shutter(state='OFF')
                G.load_program(str(bed_path))
                G.instruction(f'MSGDISPLAY 1, "COL {index + 1:03}, BED {i_bed + 1:03}"\n')
                if column.u:
                    G.instruction(f'G1 U{column.u[-1]:.6f}')
                    G.dwell(self.long_pause)
                G.move_to([float(x0), float(y0), None], speed_pos=column.speed_closed)
                G.shutter(state='ON')
                G.farcall(bed_filename)
                G.shutter(state='OFF')
                if column.u:
                    G.instruction(f'G1 U{column.u[0]:.6f}')
                G.remove_program(bed_filename)

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
        default_utcargs = {'fillcolor': '#BEBEBE', 'mode': 'none', 'hoverinfo': 'none'}
        utcargs = {**default_utcargs, **style}

        logger.debug('Add trenches beds shapes to figure.')
        for bd in self.beds:
            xt, yt = bd.border
            xt, yt, *_ = self.transform_points(xt, yt, np.zeros_like(xt, dtype=np.float32))

            fig.add_trace(
                go.Scattergl(
                    x=xt,
                    y=yt,
                    fill='toself',
                    **utcargs,
                    showlegend=False,
                    hovertemplate='(%{x:.4f}, %{y:.4f})<extra>TR</extra>',
                )
            )
        super()._plot2d_trench(fig, style)
        return fig

    def _plot3d_trench(self, fig: go.Figure, style: dict[str, Any] | None = None) -> go.Figure:
        if style is None:
            style = dict()
        default_utcargs = {'dash': 'solid', 'color': '#000000', 'width': 1.5}
        utcargs = {**default_utcargs, **style}

        if not self._trenches:
            return fig

        for tr in self._trenches:
            # get points and transform them
            xt, yt = tr.border
            xt, yt, *_ = self.transform_points(xt, yt, np.zeros_like(xt, dtype=np.float32))

            x = np.array([xt, xt])
            y = np.array([yt, yt])
            z = np.array([np.zeros_like(xt), tr.height * np.ones_like(xt)])
            fig.add_trace(
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    colorscale=[[0, 'grey'], [1, 'grey']],
                    showlegend=False,
                    hoverinfo='skip',
                    showscale=False,
                    opacity=0.6,
                )
            )
            for zval in [0.0, tr.height]:
                fig.add_trace(
                    go.Scatter3d(
                        x=xt,
                        y=yt,
                        z=zval * np.ones_like(xt),
                        mode='lines',
                        line=utcargs,
                        showlegend=False,
                        hovertemplate='(%{x:.4f}, %{y:.4f})<extra>TR</extra>',
                    )
                )

        logger.debug('Add trenches beds 3D shapes to figure.')
        tr = self._trenches[0]
        for bd in self.beds:
            xt, yt = bd.border
            xt, yt, *_ = self.transform_points(xt, yt, np.zeros_like(xt, dtype=np.float32))

            x = np.array([xt, xt])
            y = np.array([yt, yt])
            z = np.array([tr.height * np.ones_like(xt), (tr.height + 0.01) * np.ones_like(xt)])
            fig.add_trace(
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    colorscale=[[0, 'grey'], [1, 'grey']],
                    showlegend=False,
                    hoverinfo='skip',
                    showscale=False,
                    opacity=0.6,
                )
            )
        return fig


class WaveguideWriter(Writer):
    """Waveguide Writer class."""

    def __init__(
        self, param: dict[str, Any], objects: list[Waveguide | list[Waveguide]] | None = None, **kwargs
    ) -> None:

        p = copy.deepcopy(param)
        p.update(kwargs)

        super().__init__(**p)

        self._obj_list: list[Waveguide | list[Waveguide]] = [] if objects is None else flatten(listcast(objects))
        self._param: dict[str, Any] = p
        self._export_path = self.CWD / (self.export_dir or '')
        self._fabtime: float = 0.0

    @property
    def objs(self) -> list[Any]:
        """Waveguide objects.

        Property for returning the Waveguide objects contained inside a WaveguideWriter.

        Returns
        -------
        list
           List of Waveguide objects.
        """
        return self._obj_list

    @property
    def fab_time(self) -> float:
        """Waveguide fabrication time.

        Returns
        -------
        float
           Total waveguide fabrication time [s].
        """
        logger.debug(f'Return fabrication time = {self._fabtime}.')
        return self._fabtime

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
            logger.error(f'The object must be a Waveguide. {type(obj).__name__} was given.')
            raise TypeError(f'The object must be a Waveguide. {type(obj).__name__} was given.')
        self._obj_list.append(obj)
        logger.debug('Append Waveguide to _obj_list.')

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
            logger.debug(f'The object must be a list. {type(obj).__name__} was given.')
            raise TypeError(f'The object must be a list. {type(obj).__name__} was given.')
        if nest_level(obj) > 2:
            logger.error(f'The waveguide list has too many nested levels ({nest_level(obj)}).')
            raise ValueError(
                f'The waveguide list has too many nested levels ({nest_level(obj)}). The maximum value is 2.'
            )
        if all(isinstance(wg, Waveguide) for wg in flatten(obj)):
            self._obj_list.extend(obj)
        else:
            logger.error('Not all the objects in the list are type Waveguides.')
            raise TypeError('All the objects must be of type Waveguide.')

    def plot2d(
        self, fig: go.Figure | None = None, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """Plot 2D.

        2D plot of the Waveguide objects contained in ``self._obj_list``.

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
            logger.debug('Update figure.')
            return self._plot2d_wg(fig=fig, show_shutter_close=show_shutter_close, style=style)

        # If fig is None create a new figure from scratch
        logger.debug('Update figure.')
        fig = go.Figure()
        logger.debug('Update figure.')
        fig = self._plot2d_wg(fig=fig, show_shutter_close=show_shutter_close, style=style)
        fig = super().standard_2d_figure_update(fig)  # Add glass, shift_origin and axis elements
        return fig

    def plot3d(
        self, fig: go.Figure | None = None, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """Plot 3D.

        3D plot of the Waveguide objects contained in ``self._obj_list``.

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
            logger.debug('Update figure.')
            return self._plot3d_wg(fig=fig, show_shutter_close=show_shutter_close, style=style)

        # If fig is None create a new figure from scratch
        logger.debug('Update figure.')
        fig = go.Figure()
        logger.debug('Update figure.')
        fig = self._plot3d_wg(fig=fig, show_shutter_close=show_shutter_close, style=style)
        fig = super().standard_3d_figure_update(fig)  # Add glass, shift_origin and axis elements
        return fig

    def pgm(self, verbose: bool = False) -> None:
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

        See Als*o
        --------
        ``femto.pgmcompiler.PGMCompiler`` : class that convert lists of points to PGM file.
        """

        if not self.objs:
            logger.debug('No object in current writer, no PGM file created.')
            return

        _wg_fab_time = 0.0
        fn = pathlib.Path(self.filename).stem + '_WG.pgm'

        with PGMCompiler.from_dict(self._param, filename=fn) as G:
            for bunch in self.objs:
                with G.repeat(listcast(bunch)[0].scan):
                    for wg in listcast(bunch):
                        _wg_fab_time += wg.fabrication_time
                        logger.debug(f'Export {wg}.')
                        G.write(wg.points)
            G.go_init()
            _wg_fab_time += G.total_dwell_time
        del G

        self._fabtime = _wg_fab_time
        string = '{:.<49} {}'.format(
            'Estimated waveguides fabrication time: ', datetime.timedelta(seconds=int(self._fabtime))
        )
        logger.info(string)
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

        logger.debug('Add waveguides to figure.')
        for wg in listcast(flatten(self._obj_list)):
            logger.debug('Add shutter open trace to figure.')
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
            logger.debug('Add shutter close trace to figure.')
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

        logger.debug('Add waveguides to figure.')
        for wg in listcast(flatten(self._obj_list)):
            x_wg, y_wg, z_wg, _, s = wg.points
            x, y, z = self.transform_points(x_wg, y_wg, z_wg)
            xo = split_mask(x, s.astype(bool))
            yo = split_mask(y, s.astype(bool))
            zo = split_mask(z, s.astype(bool))
            logger.debug('Add shutter open trace.')
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
                logger.debug('Add shutter close trace.')
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

    def __init__(self, param: dict[str, Any], objects: list[NasuWaveguide] | None = None, **kwargs) -> None:

        p = copy.deepcopy(param)
        p.update(kwargs)

        super().__init__(**p)

        self._obj_list: list[NasuWaveguide] = flatten(listcast(objects)) if objects is not None else []
        self._param: dict[str, Any] = p
        self._export_path = self.CWD / (self.export_dir or '')
        self._fabtime: float = 0.0

    @property
    def objs(self) -> list[Any]:
        """NasuWaveguide objects.

        Property for returning the NasuWaveguide objects contained inside a NasuWaveguideWriter.

        Returns
        -------
        list
           List of NasuWaveguide objects.
        """
        return self._obj_list

    @property
    def fab_time(self) -> float:
        """Nasu waveguides fabrication time.

        Returns
        -------
        float
           Total Nasu waveguides fabrication time [s].
        """
        logger.debug(f'Return fabrication time = {self._fabtime}.')
        return self._fabtime

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
            logger.error(f'The object must be a NasuWaveguide. {type(obj).__name__} was given.')
            raise TypeError(f'The object must be a NasuWaveguide. {type(obj).__name__} was given.')
        self._obj_list.append(obj)
        logger.debug('Append Nasu Waveguide to _obj_list.')

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
            logger.error(f'The object must be a list. {type(obj).__name__} was given.')
            raise TypeError(f'The object must be a list. {type(obj).__name__} was given.')
        if all(isinstance(wg, NasuWaveguide) for wg in flatten(obj)):
            self._obj_list.extend(obj)
        else:
            logger.error('Not all the objects of the list are type NasuWaveguide.')
            raise TypeError('All the objects must be of type NasuWaveguide.')

    def plot2d(
        self, fig: go.Figure | None = None, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """Plot 2D.

        2D plot of the NasuWaveguide objects contained in ``self._obj_list``.

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
            logger.debug('Update figure.')
            return self._plot2d_nwg(fig=fig, show_shutter_close=show_shutter_close, style=style)

        # If fig is None create a new figure from scratch
        logger.debug('Create figure.')
        fig = go.Figure()
        logger.debug('Update figure.')
        fig = self._plot2d_nwg(fig=fig, show_shutter_close=show_shutter_close, style=style)
        fig = super().standard_2d_figure_update(fig)  # Add glass, shift_origin and axis elements
        return fig

    def plot3d(
        self, fig: go.Figure | None = None, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """Plot 3D.

        3D plot of the NasuWaveguide objects contained in ``self._obj_list``.

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
            logger.debug('Update figure.')
            return self._plot3d_nwg(fig=fig, show_shutter_close=show_shutter_close, style=style)

        # If fig is None create a new figure from scratch
        logger.debug('Create figure.')
        fig = go.Figure()
        logger.debug('Update figure.')
        fig = self._plot3d_nwg(fig=fig, show_shutter_close=show_shutter_close, style=style)
        fig = super().standard_3d_figure_update(fig)  # Add glass, shift_origin and axis elements
        return fig

    def pgm(self, verbose: bool = False) -> None:
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

        if not self._obj_list:
            logger.debug('No object in current writer, no PGM file created.')
            return

        _nwg_fab_time = 0.0
        fn = pathlib.Path(self.filename).stem + '_NASU.pgm'

        with PGMCompiler.from_dict(self._param, filename=fn) as G:
            for nwg in self._obj_list:
                for shift in nwg.adj_scan_order:
                    _nwg_fab_time += nwg.fabrication_time
                    dx, dy, dz = nwg.adj_scan_shift
                    coord_shift = np.array([dx, dy, dz, 0, 0]).reshape(-1, 1)
                    logger.debug(f'Export {nwg}.')
                    G.write(nwg.points + shift * coord_shift)
            G.go_init()
            _nwg_fab_time += G.total_dwell_time
        del G

        self._fabtime = _nwg_fab_time
        string = '{:.<49} {}'.format(
            'Estimated Nasu waveguides fabrication time: ', datetime.timedelta(seconds=int(self._fabtime))
        )
        logger.info(string)
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

        logger.debug('Add Nasu waveguides to figure.')
        for nwg in listcast(flatten(self._obj_list)):
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

        logger.debug('Add waveguides to figure.')
        for nwg in listcast(flatten(self._obj_list)):
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

    def __init__(self, param: dict[str, Any], objects: list[Marker] | None = None, **kwargs) -> None:

        p = copy.deepcopy(param)
        p.update(kwargs)

        super().__init__(**p)

        self._obj_list: list[Marker] = flatten(listcast(objects)) if objects is not None else []
        self._param: dict[str, Any] = p
        self._export_path = self.CWD / (self.export_dir or '')
        self._fabtime: float = 0.0

    @property
    def objs(self) -> list[Any]:
        """Marker objects.

        Property for returning the Marker objects contained inside a MarkerWriter.

        Returns
        -------
        list
           List of Marker objects.
        """
        return self._obj_list

    @property
    def fab_time(self) -> float:
        """Marker fabrication time.

        Returns
        -------
        float
           Total marker fabrication time [s].
        """
        logger.debug(f'Return fabrication time = {self._fabtime}.')
        return self._fabtime

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
            logger.error(f'The object must be a Marker. {type(obj).__name__} was given.')
            raise TypeError(f'The object must be a Marker. {type(obj).__name__} was given.')
        self._obj_list.append(obj)
        logger.debug('Append Marker to _obj_list.')

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
            logger.error(f'The object must be a list. {type(obj).__name__} was given.')
            raise TypeError(f'The object must be a list. {type(obj).__name__} was given.')
        for mk in flatten(obj):
            self.append(mk)

    def plot2d(
        self, fig: go.Figure | None = None, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """Plot 2D.

        2D plot of the Marker objects contained in ``self._obj_list``.

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
            logger.debug('Update figure.')
            return self._plot2d_mk(fig=fig, style=style)

        # If fig is None create a new figure from scratch
        logger.debug('Create figure.')
        fig = go.Figure()
        logger.debug('Update figure.')
        fig = self._plot2d_mk(fig=fig, style=style)
        fig = super().standard_2d_figure_update(fig)  # Add glass, shift_origin and axis elements
        return fig

    def plot3d(
        self, fig: go.Figure | None = None, show_shutter_close: bool = True, style: dict[str, Any] | None = None
    ) -> go.Figure:
        """Plot 3D.

        3D plot of the Marker objects contained in ``self._obj_list``.

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
            logger.debug('Update figure.')
            return self._plot3d_mk(fig=fig, style=style)

        # If fig is None create a new figure from scratch
        logger.debug('Create figure.')
        fig = go.Figure()
        logger.debug('Update figure.')
        fig = self._plot3d_mk(fig=fig, style=style)
        fig = super().standard_3d_figure_update(fig)  # Add glass, shift_origin and axis elements
        return fig

    def pgm(self, verbose: bool = False) -> None:
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

        if not self._obj_list:
            logger.debug('No object in current writer, no PGM file created.')
            return

        _mk_fab_time = 0.0
        fn = pathlib.Path(self.filename).stem + '_MK.pgm'

        with PGMCompiler.from_dict(self._param, filename=fn) as G:
            for idx, mk in enumerate(flatten(self._obj_list)):
                logger.debug(f'Export {mk}.')
                with G.repeat(mk.scan):
                    _mk_fab_time += mk.fabrication_time
                    G.comment(f'MARKER {idx + 1}')
                    G.write(mk.points)
                    G.comment('')
            G.go_origin()
            _mk_fab_time += G.total_dwell_time
        del G

        self._fabtime = _mk_fab_time
        string = '{:.<49} {}'.format(
            'Estimated markers fabrication time: ', datetime.timedelta(seconds=int(self._fabtime))
        )
        logger.info(string)
        self._instructions.clear()

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

        logger.debug('Add marker trace to figure.')
        for mk in listcast(flatten(self._obj_list)):
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

        logger.debug('Add marker trace to figure.')
        for mk in listcast(flatten(self._obj_list)):
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
    """The main function of the script."""
    from femto.curves import sin, circ, spline_bridge
    from femto.waveguide import NasuWaveguide

    # Data
    param_wg: dict[str, Any] = dict(scan=6, speed=20, radius=15, pitch=0.080, int_dist=0.007, samplesize=(10, 3))
    param_gc: dict[str, Any] = dict(filename='testPGM.pgm', samplesize=param_wg['samplesize'])

    increment = [5.0, 0, 0]

    # Calculations
    mzi = []
    for index in range(2):
        wg = NasuWaveguide(adj_scan_shift=(0, 0.004, 0), **param_wg)
        wg.y_init = -wg.pitch / 2 + index * wg.pitch
        wg.start()
        wg.linear(increment)
        wg.bend(dy=(-1) ** index * 0.08, dz=(-1) ** index * 0.015, fx=spline_bridge)
        wg.bend(dy=(-1) ** (index + 1) * wg.dy_bend, dz=0, fx=circ)
        wg.bend(dy=(-1) ** index * wg.dy_bend, dz=0, fx=sin)
        wg.linear(increment)
        wg.end()
        mzi.append(wg)

    nwr = NasuWriter(param_gc, objects=mzi)
    fig = nwr.plot3d()
    fig.show()


if __name__ == '__main__':
    main()
