import os
from collections import deque
from collections.abc import Iterable
from contextlib import contextmanager
from copy import deepcopy
from itertools import zip_longest
from math import radians
from pathlib import Path

import numpy as np

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from typing import List

from femto import Trench, TrenchColumn
from femto.Parameters import GcodeParameters, WaveguideParameters


class PGMCompiler:
    """
    Class representing a PGM Compiler.
    """

    def __init__(self, param: GcodeParameters):

        if param.filename is None:
            raise ValueError('Filename is None, set GcodeParameters.filename.')
        else:
            self.filename = param.filename
        self.lab = param.lab
        self.cwd = param.CWD
        self.warp_flag = param.warp_flag
        self.fwarp = param.fwarp
        self.long_pause = param.long_pause
        self.short_pause = param.short_pause

        self.ind_rif = param.neff
        if param.angle != 0:
            print(' BEWARE ANGLES MUST BE IN DEGREE!! '.center(39, "*"))
            print(f' Given alpha = {param.angle % 360:.3f} deg. '.center(39, "*"))
        self.angle = radians(param.angle % 360)

        self.output_digits = param.output_digits

        self._instructions = deque()
        self._total_dwell_time = 0.0
        self._shutter_on = False
        self._loaded_files = []

    def __enter__(self):
        """
        Context manager entry

        :return: Self

        Can be use like:
        ::
            with femto.PGMCompiler(filename, ind_rif) as gc:
                <code block>
        """
        self.header()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit

        :return: None
        """
        self.homing()
        self.close()

    # Methods
    def header(self):
        """
        The function print the header file of the G-Code file. The user can specify the fabrication line to work in
        ``CAPABLE`` or ``FIRE LINE1`` as parameter when the G-Code Compiler obj is instantiated.

        :return: None
        """
        if self.lab.upper() not in ['CAPABLE', 'FIRE']:
            raise ValueError(f'Fabrication line should be CAPABLE or FIRE. Given {self.lab.upper()}.')

        if self.lab.upper() == 'CAPABLE':
            with open(os.path.join(self.cwd, 'utils', 'header_capable.txt')) as fd:
                self._instructions.extend(fd.readlines())
        else:
            with open(os.path.join(self.cwd, 'utils', 'header_fire.txt')) as fd:
                self._instructions.extend(fd.readlines())

    def dvar(self, variables: List[str]):
        """
        Adds the declaration of variables in a G-Code file.

        :param variables: List of G-Code variables
        :type variables: List[str]
        :return: None
        """
        args = ' '.join(["${}"] * len(variables)).format(*variables)
        self._instructions.appendleft(f'DVAR {args}\n')

    def comment(self, comstring: str):
        """
        Adds a comment to a G-Code file.

        :param comstring: Content of the comment (without line-break character).
        :type comstring: str
        :return: None
        """
        self._instructions.append(f'\n; {comstring}\n')

    def shutter(self, state: str):
        """
        Adds the instruction to open (close) the shutter to a G-Code file only when necessary.
        The user specifies the state and the function compare it to the current state of the shutter (which is
        tracked internally during the compilation of the .pgm file). The instruction is printed to file only if the
        new state differs from the current one.

        :param state: New state of the shutter. ``ON`` or ``OFF``
        :type state: str
        :return: None

        :raise ValueError: Shutter state not valid.
        """
        if state.upper() not in ['ON', 'OFF']:
            raise ValueError(f'Shutter state should be ON or OFF. Given {state.upper()}.')

        if state.upper() == 'ON' and self._shutter_on is False:
            self._shutter_on = True
            self._instructions.append('\nPSOCONTROL X ON\n')
        elif state.upper() == 'OFF' and self._shutter_on is True:
            self._shutter_on = False
            self._instructions.append('\nPSOCONTROL X OFF\n')
        else:
            pass

    def dwell(self, pause: float):
        """
        Adds pause instruction to a G-Code file.

        :param pause: Value of the pause time [s].
        :type pause: float
        :return: None
        """
        self._instructions.append(f'DWELL {pause}\n\n')
        self._total_dwell_time += float(pause)

    def set_home(self, home_pos: List[float]):
        """
        Defines a preset position or a software home position to the one specified in the input list.
        To exclude a variable set it to None.

        :param home_pos: Ordered coordinate list that specifies software home position [mm].
        ::
            ``home_pos[0]`` -> X
            ``home_pos[1]`` -> Y
            ``home_pos[2]`` -> Z
        :type home_pos: List[float]
        :return: None

        :raise ValueError: Final position is not valid.
        """
        if np.size(home_pos) != 3:
            raise ValueError(f'Given final position is not valid. 3 values required, given {np.size(home_pos)}.')

        args = self._format_args(*home_pos)
        self._instructions.append(f'G92 {args}\n')

    def homing(self):
        """
        Utility function to return to the origin (0,0,0) with shutter OFF.

        :return: None
        """
        self.comment('HOMING')
        self.move_to([0, 0, 0])

    def move_to(self, position: List[float], speedpos: float = 50):
        """
        Utility function to move to a given position with the shutter OFF. The user can specify the target position
        and the positioning speed.

        :param position: Ordered coordinate list that specifies the target position [mm].
            position[0] -> X
            position[1] -> Y
            position[2] -> Z
        :type position: List[float]
        :param speedpos: Positioning speed [mm/s]. The default is 50 mm/s.
        :type speedpos: float
        :return: None
        """
        if np.size(position) != 3:
            raise ValueError(f'Given final position is not valid. 3 values required, given {np.size(position)}.')

        if self._shutter_on is True:
            self.shutter('OFF')

        args = self._format_args(*position, speedpos)
        self._instructions.append(f'LINEAR {args}\n')
        self.dwell(self.long_pause)

    @contextmanager
    def for_loop(self, var: str, num: int):
        """
        Context manager that manages a FOR loop in a G-Code file.

        :param var: Name of the variable used for iteration.
        :type var: str
        :param num: Number of iterations.
        :type num: int
        :return: None
        """
        if num is None:
            raise ValueError('Number of iterations is None. Set the num_scan attribute in Waveguide obj.')
        if num == 0:
            raise ValueError('Number of iterations is 0. Set num_scan >= 1.')
        self._instructions.append(f'FOR ${var} = 0 TO {num - 1}\n')
        _temp_dt = self._total_dwell_time
        try:
            yield
        finally:
            self._instructions.append(f'NEXT ${var}\n\n')
            _dt_forloop = self._total_dwell_time - _temp_dt
            # pauses should be multiplied by number of cycles as well
            self._total_dwell_time += (num - 1) * _dt_forloop

    @contextmanager
    def repeat(self, num: int):
        """
        Context manager that manages a REPEAT loop in a G-Code file.

        :param num: Number of iterations.
        :type num: int
        :return: None
        """
        if num is None:
            raise ValueError('Number of iterations is None. Set the `scan` attribute in Waveguide obj.')
        if num == 0:
            raise ValueError('Number of iterations is 0. Set num_scan >= 1.')
        self._instructions.append(f'REPEAT {num}\n')
        _temp_dt = self._total_dwell_time
        try:
            yield
        finally:
            self._instructions.append('ENDREPEAT\n\n')
            _dt_repeat = self._total_dwell_time - _temp_dt
            # pauses should be multiplied by number of cycles as well
            self._total_dwell_time += (num - 1) * _dt_repeat

    def tic(self):
        """
        Print the current time (hh:mm:ss) in message panel. The function is intended to be used before the execution
        of an operation or script to measure its time performances.

        :return: None
        """
        self._instructions.append('MSGDISPLAY 1, "INIZIO #TS"\n\n')

    def toc(self):
        """
        Print the current time (hh:mm:ss) in message panel. The function is intended to be used after the execution
        of an operation or script to measure its time performances.

        :return: None
        """
        self._instructions.append('MSGDISPLAY 1, "FINE   #TS"\n')
        self._instructions.append('MSGDISPLAY 1, "---------------------"\n')
        self._instructions.append('MSGDISPLAY 1, " "\n\n')

    def load_program(self, filename: str):
        """
        Adds the instruction to LOAD a program in a G-Code file.

        :param filename: Name of the file that have to be loaded.
        :type filename: str
        :return: None
        """
        file = self._parse_filepath(filename, extension='pgm')
        self._instructions.append(f'PROGRAM 0 LOAD "{file}"\n')
        self._loaded_files.append(file.stem)

    def remove_program(self, filename: str):
        """
        Adds the instruction to REMOVE a program from memory buffer in a G-Code file.

        :param filename: Name of the file to remove.
        :type filename: str
        :return: None
        """
        file = self._parse_filepath(filename, extension='pgm')
        self._instructions.append(f'REMOVEPROGRAM "{file}"\n')
        self._loaded_files.remove(file.stem)

    def farcall(self, filename: str):
        """
        Adds the FARCALL instruction in a G-Code file.

        :param filename: Name of the file to call.
        :type filename: str
        :return: None
        """
        file = self._parse_filepath(filename)
        if file.stem not in self._loaded_files:
            raise FileNotFoundError(f'{file} not loaded. Cannot load it.')
        self._instructions.append(f'FARCALL "{file}"\n')
        self._instructions.append('PROGRAM 0 STOP\n')

    def write(self, points: np.ndarray):
        """
        The function convert the quintuple (X,Y,Z,F,S) to G-Code instructions. The (X,Y,Z) coordinates are
        transformed using the transformation matrix that takes into account the rotation of a given angle and the
        homothety to compensate the (effective) refractive index different from 1. Moreover, if the warp_flag is True
        the points are compensated along the z-direction.

        The transformed points are then parsed together with the feed rate and shutter state coordinate to produce
        the LINEAR movements.

        :param points: Numpy matrix containing the values of the tuple [X,Y,Z,F,S] coordinates.
        :type points: numpy.ndarray
        :return: None
        """
        x, y, z, f_c, s_c = points.T
        sub_points = np.stack((x, y, z), axis=-1).astype(np.float32)
        if self.warp_flag:
            sub_points = np.matmul(sub_points, self.t_matrix())
            x_c, y_c, z_c = self.compensate(sub_points).T
        else:
            x_c, y_c, z_c = np.matmul(sub_points, self.t_matrix()).T
        args = [self._format_args(x, y, z, f)
                for (x, y, z, f) in zip(x_c, y_c, z_c, f_c)]
        for (arg, s) in zip_longest(args, s_c):
            if s == 0 and self._shutter_on is False:
                pass
            elif s == 0 and self._shutter_on is True:
                self.shutter('OFF')
                self.dwell(self.long_pause)
            elif s == 1 and self._shutter_on is False:
                self.shutter('ON')
                self.dwell(self.long_pause)
            self._instructions.append(f'LINEAR {arg}\n')
        self.dwell(self.long_pause)

    def instruction(self, instr: str):
        """
        Adds a G-Code instruction passed as parameter to the PGM file.

        :param instr: Instruction line to be added to the PGM file. The ``\\n`` character is optional.
        :type instr: str
        :return: None
        """
        if instr.endswith('\n'):
            self._instructions.append(instr)
        else:
            self._instructions.append(instr + '\n')

    def close(self, filename: str = None, verbose: bool = False):
        """
        Dumps all the instruction in self._instruction in a .pgm file.
        The filename is specified during the class instatiation. If no extension is present, the proper one is
        automatically added.

        :param filename: Different filename. The default is None, using self.filename.
        :type filename: str
        :param verbose: Print when G-Code export is finished. The default is False.
        :type verbose: bool
        :return: None

        :raise ValueError: Missed filename.
        """

        # filename overrides self.filename. If not present, self.filename must not be None.
        if filename is None and self.filename is None:
            raise ValueError('No filename given.')

        if filename:
            pgm_filename = filename
        else:
            pgm_filename = self.filename
        # if not present in the filename, add the proper file extension
        if not pgm_filename.endswith('.pgm'):
            pgm_filename += '.pgm'

        # write instruction to file
        with open(pgm_filename, 'w') as f:
            f.write(''.join(self._instructions))
        if verbose:
            print('G-code compilation completed.')

    def trench(self, col: TrenchColumn, col_index: int = None, base_folder: str = r'C:\Users\Capable\Desktop',
               dirname: str = 's-trench', u: List = None, nboxz: int = 4, hbox: float = 0.075, zoff: float = 0.020,
               deltaz: float = 0.0015, tspeed: float = 4, speed_pos: float = 5, pause: float = 0.5):
        """
        See :mod:`make_trench() function femto.PGMCompiler.make_trench`
        """
        make_trench(self, col, col_index, base_folder, dirname, u, nboxz, hbox, zoff, deltaz, tspeed, speed_pos, pause)

    def compensate(self, pts: np.ndarray) -> np.ndarray:
        """
        Returns the points compensated along z-direction for the refractive index, the offset and the glass warp.

        :param pts: ``[X,Y,Z]`` matrix or just a single point
        :type pts: np.ndarray
        :return: ``[X,Y,Z]`` matrix of compensated points
        :rtype: np.ndarray
        """
        pts_comp = deepcopy(np.array(pts))

        if pts_comp.size > 3:
            zwarp = [float(self.fwarp(x, y)) for x, y in zip(pts_comp[:, 0], pts_comp[:, 1])]
            zwarp = np.array(zwarp)
            pts_comp[:, 2] = (pts_comp[:, 2] + zwarp / self.ind_rif)
        else:
            pts_comp[2] = (pts_comp[2] + self.fwarp(pts_comp[0], pts_comp[1]) / self.ind_rif)
        return pts_comp

    def t_matrix(self, dim: int = 3) -> np.ndarray:
        """
        Given the rotation angle and the rifraction index, the function compute the transformation matrix as
        composition of rotatio matrix (RM) and a homothety matrix (SM).

        :param dim: Dimension of the transformation matrix. The default is 3.
        :type dim: int
        :return: Transformation matrix, TM = SM*RM
        :rtype: np.array

        :raise ValueError: Dimension not valid
        """
        RM = np.array([[np.cos(self.angle), -np.sin(self.angle), 0],
                       [np.sin(self.angle), np.cos(self.angle), 0],
                       [0, 0, 1]])
        SM = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1 / self.ind_rif]])
        t_mat = SM @ RM
        if dim == 3:
            return t_mat
        elif dim == 2:
            # export xy-submatrix
            ixgrid = np.ix_([0, 1], [0, 1])
            return t_mat[ixgrid]
        else:
            raise ValueError(f'Dimension not valid. dim must be either 2 or 3. Given {dim}.')

    # Private interface
    def _format_args(self, x: float = None, y: float = None, z: float = None, f: float = None) -> str:
        """
        Utility function that creates a string prepending the coordinate name to the given value for all the given
        the coordinates ``[X,Y,Z]`` and feed rate ``F``.
        The decimal precision can be set by the user by setting the output_digits attribute.

        :param x: Value of the x-coordinate [mm]. The default is None.
        :type x: float
        :param y: Value of the y-coordinate [mm]. The default is None.
        :type y: float
        :param z: Value of the z-coordinate [mm]. The default is None.
        :type z: float
        :param f: Value of the f-coordinate [mm]. The default is None.
        :type f: float
        :return: Formatted string of the type: 'X<value> Y<value> Z<value> F<value>'.
        :rtype: str

        :raise ValueError: Try to move null speed.
        """
        args = []
        if x is not None:
            args.append(f'X{x:.{self.output_digits}f}')
        if y is not None:
            args.append(f'Y{y:.{self.output_digits}f}')
        if z is not None:
            args.append(f'Z{z:.{self.output_digits}f}')
        if f is not None:
            if f < 1e-6:
                raise ValueError('Try to move with F = 0.0 mm/s. Check speed parameter.')
            args.append(f'F{f:.{self.output_digits}f}')
        args = ' '.join(args)
        return args

    @staticmethod
    def _parse_filepath(filename: str, filepath: str = None, extension: str = None) -> Path:
        """
        The fuction takes a filename and (optional) filepath. It merges the two and check if the file exists in the
        system.
        An extension parameter can be given as input. In that case the function also checks weather the filename has
        the correct extension.

        :param filename: Name of the file that have to be loaded.
        :type filename: str
        :param filepath: Path of the folder containing the file. The default is None.
        :type filepath: str
        :param extension: File extension. The default is None.
        :type extension: str
        :return: Complete path of the file (filepath + filename).
        :rtype: pathlib.Path
        """
        if extension is not None and not filename.endswith(extension):
            raise ValueError(f'Given filename has wrong extension. Given {filename}, required .{extension}.')

        if filepath is not None:
            file = Path(filepath) / filename
        else:
            file = Path(filename)
        return file


def write_array(gc: PGMCompiler, points: np.ndarray, f_array: list = None):
    """
    Helper function that produces a PGM file for a 3D matrix of points at a given traslation speed,
    without shuttering operations.
    The function parse the points input matrix, applies the rotation and homothety transformations and parse all the
    LINEAR instructions.

    :param gc: Instance of a PGMCompiler for compilation of a G-Code file.
    :type gc: PGMCompiler
    :param points: 3D points matrix. If the points matrix is 2D it is intended as [x,y] coordinates.
    :type points: np.ndarray
    :param f_array: List of traslation speed values. The default is [].
    :type f_array: list
    :return: None
    """
    if points.shape[-1] == 2:
        x_array, y_array = np.matmul(points, gc.t_matrix(dim=2)).T
        z_array = [None]
    else:
        x_array, y_array, z_array = np.matmul(points, gc.t_matrix()).T

    if not isinstance(f_array, Iterable):
        f_array = [f_array]

    instructions = [gc._format_args(x, y, z, f) for (x, y, z, f) in zip_longest(x_array, y_array, z_array, f_array)]
    gc._instructions.extend([f'LINEAR {line}\n' for line in instructions])


def export_trench_path(trench: Trench, filename: str, ind_rif: float, angle: float, tspeed: float = 4):
    """
    Helper function for the export of the wall and floor instruction of a Trench obj.

    :param trench: Trench obj to export.
    :type trench: Trench
    :param filename: Base filename for the wall.pgm and floor.pgm files. If the filename ends with the '.pgm'
    extension, the latter it is stripped and replaced with '_wall.pgm' and '_floor.pgm' to differentiate the two paths.
    :type filename: str
    :param ind_rif: Refractive index.
    :type ind_rif: float
    :param angle: Rotation angle for the fabrication.
    :type angle: float
    :param tspeed: Traslation speed during fabrication [mm/s]. The default is 4 [mm/s].
    :type tspeed: float
    :return: None
    """
    if filename.endswith('.pgm'):
        filename = filename.split('.')[0]

    TEMP_GC = GcodeParameters(
        filename=filename + '_wall.pgm',
        n_glass=ind_rif,
        n_environment=1.0,
        angle=angle
    )
    G = PGMCompiler(TEMP_GC)
    write_array(G, np.array(trench.block.exterior.coords.xy).T, f_array=[tspeed])
    G.close()
    del G

    TEMP_GC.filename = filename + '_floor.pgm'
    G = PGMCompiler(TEMP_GC)
    for path in trench.trench_paths():
        write_array(G, np.stack(path, axis=-1), f_array=[tspeed])
    G.close()
    del G


def make_trench(gc: PGMCompiler, col: TrenchColumn, col_index: int = None,
                base_folder: str = r'C:\Users\Capable\Desktop', dirname: str = 's-trench', u: List = None,
                nboxz: int = 4, hbox: float = 0.075, zoff: float = 0.020, deltaz: float = 0.0015, tspeed: float = 4,
                speed_pos: float = 5, pause: float = 0.5):
    """
    Helper function for the compilation of trench columns.
    For each trench in the column, the function first compile a PGM file for border (or wall) and for the floor
    inside a directory given by the user (base_folder).
    Secondly, the function produce a FARCALL.pgm program to fabricate all the trenches in the column.

    :param gc: Instance of a PGMCompiler for compilation of a G-Code file.
    :type gc: PGMCompiler
    :param col: TrenchColumn obj containing the list of trench blocks to compile.
    :type col: List
    :param col_index: Index of the column, used for organize the code in folders. The default is None,
    trench directories will not be indexed.
    :type col_index: int
    :param base_folder: String of the full PATH (in the lab computer) of the directory containig all the scripts for
    the fabrication.
    :type base_folder: str
    :param dirname: Directory to export the trenches PGM files. The default is 's-trench'.
    :type dirname: str
    :param u: List of two values of U-coordinate for fabrication of wall and floor of the trench. The default is None.
            ``u[0]`` -> U-coordinate for the wall
            ``u[1]`` -> U-coordinate for the floor
    :type u: List
    :param nboxz: Number of sub-box along z-direction in which the trench is divided. The default is 4.
    :type nboxz: int
    :param hbox: Height along z-direction [mm] of the single sub-box. The default is 0.075 mm.
    :type hbox: float
    :param zoff: Offset [mm] in the z-direction for the starting the inscription of the trench wall. The default is
    0.020 [mm].
    :type zoff: float
    :param deltaz: Distance [mm] along z-direction between different wall planes. The default is 0.0015 [mm].
    :type deltaz: float
    :param tspeed: Traslation speed during fabrication [mm/s]. The default is 4 mm/s.
    :type tspeed: float
    :param speed_pos: Positioning speed [mm/s]. The default is 5 mm/s.
    :type speed_pos: float
    :param pause: float
    :type pause:  Value of pause. Units in [s]. The default is 0.5 s.
    :return: None
    """
    if col_index is None:
        trench_directory = os.path.join(dirname, 'trenchCol')
    else:
        trench_directory = os.path.join(dirname, f'trenchCol{col_index + 1:03}')

    col_dir = os.path.join(os.getcwd(), trench_directory)
    os.makedirs(col_dir, exist_ok=True)
    for i, trench in enumerate(col):
        filename = os.path.join(col_dir, f'trench{i + 1:03}_')
        export_trench_path(trench, filename, gc.ind_rif, gc.angle, tspeed)

    gc.dvar(['ZCURR'])

    for nbox in range(nboxz):
        for t_index, trench in enumerate(col):
            # load filenames (wall/floor)
            wall_filename = f'trench{t_index + 1:03}_wall.pgm'
            floor_filename = f'trench{t_index + 1:03}_floor.pgm'
            wall_path = os.path.join(base_folder, trench_directory, wall_filename)
            floor_path = os.path.join(base_folder, trench_directory, floor_filename)

            x0, y0 = trench.block.exterior.coords[0]
            z0 = (nbox * hbox - zoff) / gc.ind_rif
            gc.comment(f'+--- TRENCH #{t_index + 1}, LEVEL {nbox + 1} ---+')
            gc.load_program(wall_path)
            gc.load_program(floor_path)
            gc.shutter('OFF')
            gc.move_to([x0, y0, z0], speedpos=speed_pos)

            gc.instruction(f'$ZCURR = {z0:.6f}')
            gc.shutter('ON')
            with gc.repeat(int(np.ceil((hbox + zoff) / deltaz))):
                gc.farcall(wall_filename)
                gc.instruction(f'$ZCURR = $ZCURR + {deltaz / gc.ind_rif:.6f}')
                gc.instruction('LINEAR Z$ZCURR')

            if u is not None:
                gc.instruction(f'LINEAR U{u[-1]:.6f}')
            gc.dwell(pause)
            gc.farcall(floor_filename)
            gc.shutter('OFF')
            if u is not None:
                gc.instruction(f'LINEAR U{u[0]:.6f}')

            gc.remove_program(wall_path)
            gc.remove_program(floor_path)


def _example():
    from femto import Waveguide

    # Data
    PARAMETERS_WG = WaveguideParameters(
        scan=6,
        speed=20,
        radius=15,
        pitch=0.080,
        int_dist=0.007,
    )
    increment = [PARAMETERS_WG.lsafe, 0, 0]

    PARAMETERS_GC = GcodeParameters(
        filename='testPGMcompiler.pgm',
        lab='CAPABLE',
        samplesize=(25, 25),
        angle=0.0,
        warp_flag=True,
    )

    # Calculations
    coup = [Waveguide(PARAMETERS_WG) for _ in range(2)]
    for i, wg in enumerate(coup):
        wg.start([-2, -PARAMETERS_WG.pitch / 2 + i * PARAMETERS_WG.pitch, 0.035]) \
            .linear(increment) \
            .sin_mzi((-1) ** i * PARAMETERS_WG.dy_bend, arm_length=1.0) \
            .linear(increment)
        wg.end()

    # Compilation
    with PGMCompiler(PARAMETERS_GC) as G:
        G.set_home([0, 0, 0])
        with G.repeat(PARAMETERS_WG.scan):
            for i, wg in enumerate(coup):
                G.comment(f'Modo: {i}')
                G.write(wg.points)
        G.move_to([None, 0, 0.1])
        G.set_home([0, 0, 0])


if __name__ == '__main__':
    _example()
