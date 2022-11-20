import math
import os
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from itertools import product, zip_longest
from operator import add
from pathlib import Path
from typing import List, Optional, Tuple, TypeVar, Union

import dill
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp2d

from femto.helpers import Dotdict, flatten, listcast, pad
from femto.Trench import Trench, TrenchColumn

# Create a generic variable that can be 'LaserPath', or any subclass.
GC = TypeVar("GC", bound="PGMCompiler")


@dataclass
class PGMCompiler:
    """
    Class representing a PGM Compiler.
    """

    filename: str
    export_dir: str = ""
    samplesize: Tuple[float, float] = (None, None)
    laser: str = "PHAROS"
    home: bool = False
    new_origin: Tuple[float] = (0.0, 0.0)
    warp_flag: bool = False
    n_glass: float = 1.50
    n_environment: float = 1.33
    rotation_angle: Optional[float] = None
    aerotech_angle: Optional[float] = None
    long_pause: float = 0.5
    short_pause: float = 0.05
    output_digits: int = 6
    speed_pos: float = 5.0
    flip_x: bool = False
    flip_y: bool = False

    _total_dwell_time: float = 0.0
    _shutter_on: bool = False
    _mode_abs: bool = True

    def __post_init__(self: GC) -> None:
        if self.filename is None:
            raise ValueError("Filename is None, set 'filename' attribute")
        self.CWD = Path(__file__).parent
        self._instructions = deque()
        self._loaded_files: list = []
        self._dvars = []

        self.fwarp = self.antiwarp_management(self.warp_flag)

        # Set rotation angle in radians for matrix rotations
        if self.rotation_angle:
            self.rotation_angle = math.radians(self.rotation_angle % 360)
        else:
            self.rotation_angle = float(0.0)

        # Set AeroTech angle between 0 and 359 for G84 command
        if self.aerotech_angle:
            self.aerotech_angle = self.aerotech_angle % 360
        else:
            self.aerotech_angle = float(0.0)

    @classmethod
    def from_dict(cls: GC, param: Union[dict, Dotdict]) -> GC:
        return cls(**param)

    def __enter__(self: GC) -> GC:
        """
        Context manager entry

        :return: Self

        Can be use like:
        ::
            with femto.PGMCompiler(filename, ind_rif) as gc:
                <code block>
        """
        self.header()
        self.dwell(1.0)
        self.instruction("\n")
        if self.aerotech_angle:
            print(" BEWARE, G84 COMMAND WILL BE USED!!! ".center(39, "*"))
            print(" ANGLE MUST BE IN DEGREE! ".center(39, "*"))
            print(f" Rotation angle is {self.aerotech_angle:.3f} deg. ".center(39, "*"))
            print()
            self._enter_axis_rotation(angle=self.aerotech_angle)
        return self

    def __exit__(self: GC, exc_type, exc_value, traceback) -> None:
        """
        Context manager exit

        :return: None
        """

        if self.aerotech_angle:
            self._exit_axis_rotation()
            self._instructions.append("\n")
        if self.home:
            self.go_init()
        self.close()

    @property
    def xsample(self: GC) -> Optional[float]:
        if self.samplesize[0] is None:
            return None
        return np.fabs(self.samplesize[0])

    @property
    def ysample(self: GC) -> Optional[float]:
        if self.samplesize[1] is None:
            return None
        return np.fabs(self.samplesize[1])

    @property
    def neff(self: GC) -> Optional[float]:
        if self.n_glass is None or self.n_environment is None or math.isclose(self.n_environment, 0.0, abs_tol=1e-6):
            return None
        return self.n_glass / self.n_environment

    @property
    def tshutter(self: GC) -> float:
        """
        Function that set the shuttering time given the fabrication laboratory.

        :return: shuttering time
        :rtype: float
        """
        if self.laser.lower() not in ["pharos", "carbide", "uwe"]:
            raise ValueError(f"Laser can be only PHAROS, CARBIDE or UWE. Given {self.laser.upper()}.")
        if self.laser.lower() == "uwe":
            return 0.005
        return 0.000

    @property
    def dwell_time(self: GC) -> float:
        return self._total_dwell_time

    # G-Code Methods
    def header(self: GC) -> None:
        """
        The function print the header file of the G-Code file. The user can specify the fabrication line to work in
        ``CAPABLE``, ``CARBIDE`` or ``FIRE LINE1`` as parameter when the G-Code Compiler obj is instantiated.

        :return: None
        """
        if self.laser is None or self.laser.lower() not in ["pharos", "carbide", "uwe"]:
            raise ValueError(f"Fabrication line should be PHAROS, CARBIDE or UWE. Given {self.laser}.")

        header_name = f"header_{self.laser.lower()}.txt"
        with open(self.CWD / "utils" / header_name) as f:
            self._instructions.extend(f.readlines())

    def dvar(self: GC, variables: List[str]) -> None:
        """
        Adds the declaration of variables in a G-Code file.

        :param variables: List of G-Code variables
        :type variables: List[str]
        :return: None
        """
        # cast variables to a flattened list (no nested lists)
        variables = listcast(flatten(variables))

        args = " ".join(["${}"] * len(variables)).format(*variables)
        self._instructions.appendleft(f"DVAR {args}\n")

        # keep track of all variables
        self._dvars.extend([var.lower() for var in variables])

    def mode(self, mode: str = "abs") -> None:
        if mode is None or mode.lower() not in ["abs", "inc"]:
            raise ValueError(f"Mode should be either ABSOLUTE (ABS) or INCREMENTAL (INC). {mode} was given.")

        if mode.lower() == "abs":
            self._instructions.append("ABSOLUTE\n")
            self._mode_abs = True
        else:
            self._instructions.append("INCREMENTAL\n")
            self._mode_abs = False

    def comment(self: GC, comstring: str) -> None:
        """
        Adds a comment to a G-Code file.

        :param comstring: Content of the comment (without line-break character).
        :type comstring: str
        :return: None
        """
        if comstring:
            self._instructions.append(f"\n; {comstring}\n")
        else:
            self._instructions.append("\n")

    def shutter(self: GC, state: str) -> None:
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
        if state is None or state.lower() not in ["on", "off"]:
            raise ValueError(f"Shutter state should be ON or OFF. Given {state}.")

        if state.lower() == "on" and self._shutter_on is False:
            self._shutter_on = True
            self._instructions.append("\nPSOCONTROL X ON\n")
        elif state.lower() == "off" and self._shutter_on is True:
            self._shutter_on = False
            self._instructions.append("\nPSOCONTROL X OFF\n")
        else:
            pass

    def dwell(self: GC, pause: float) -> None:
        """
        Adds pause instruction to a G-Code file.

        :param pause: Value of the pause time [s].
        :type pause: float
        :return: None
        """
        if pause is None or pause == float(0.0):
            return None
        self._instructions.append(f"DWELL {np.fabs(pause)}\n")
        self._total_dwell_time += np.fabs(pause)

    def set_home(self: GC, home_pos: List[float]) -> None:
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
            raise ValueError(f"Given final position is not valid. 3 values required, given {np.size(home_pos)}.")

        if all(coord is None for coord in home_pos):
            raise ValueError(f"Given home position is (None, None, None). Give a valid home position.")

        args = self._format_args(*home_pos)
        self._instructions.append(f"G92 {args}\n")

    def move_to(self: GC, position: List[float], speed_pos: Optional[float] = None) -> None:
        """
        Utility function to move to a given position with the shutter OFF. The user can specify the target position
        and the positioning speed.

        :param position: Ordered coordinate list that specifies the target position [mm].
            position[0] -> X
            position[1] -> Y
            position[2] -> Z
        :type position: List[float]
        :param speed_pos: Positioning speed [mm/s]. The default is self.speed_pos.
        :type speed_pos: float
        :return: None
        """
        if np.size(position) != 3:
            raise ValueError(f"Given final position is not valid. 3 values required, given {np.size(position)}.")

        if speed_pos is None and self.speed_pos is None:
            raise ValueError(f"The positioning speed is None. Set the 'speed_pos' attribute or give a valid value.")
        speed_pos = self.speed_pos if speed_pos is None else speed_pos

        # close the shutter before the movements
        if self._shutter_on is True:
            self.shutter("OFF")

        args = self._format_args(*position, speed_pos)
        if all(coord is None for coord in position):
            self._instructions.append(f"{args}\n")
        else:
            self._instructions.append(f"LINEAR {args}\n")
        self.dwell(self.long_pause)
        self.instruction("\n")

    def homing(self: GC) -> None:
        """
        Utility function to return to the origin (0,0,0) with shutter OFF.

        :return: None
        """
        self.comment("HOMING")
        self.move_to([0.0, 0.0, 0.0])

    def go_init(self: GC) -> None:
        """
        Utility function to return to the initial point of fabrication (-2,0,0) with shutter OFF.

        :return: None
        """
        self.move_to([-2, 0, 0])

    @contextmanager
    def axis_rotation(self: GC, angle: Optional[float] = None) -> None:
        self._enter_axis_rotation(angle=angle)
        try:
            yield
        finally:
            self._exit_axis_rotation()

    @contextmanager
    def for_loop(self: GC, var: str, num: int) -> None:
        """
        Context manager that manages a FOR loop in a G-Code file.

        :param var: Name of the variable used for iteration.
        :type var: str
        :param num: Number of iterations.
        :type num: int
        :return: None
        """
        if num is None:
            raise ValueError("Number of iterations is None. Give a valid 'scan' attribute value.")
        if num <= 0:
            raise ValueError("Number of iterations is 0. Set 'scan'>= 1.")

        if var is None:
            raise ValueError("Given variable is None. Give a valid varible.")
        if var.lower() not in self._dvars:
            raise ValueError(f"Given variable has not beed declared. Use dvar() method to declare ${var} variable.")

        self._instructions.append(f"FOR ${var} = 0 TO {int(num) - 1}\n")
        _temp_dt = self._total_dwell_time
        try:
            yield
        finally:
            self._instructions.append(f"NEXT ${var}\n\n")

            # pauses should be multiplied by number of cycles as well
            self._total_dwell_time += int(num - 1) * (self._total_dwell_time - _temp_dt)

    @contextmanager
    def repeat(self: GC, num: int) -> None:
        """
        Context manager that manages a REPEAT loop in a G-Code file.

        :param num: Number of iterations.
        :type num: int
        :return: None
        """
        if num is None:
            raise ValueError("Number of iterations is None. Give a valid 'scan' attribute value.")
        if num <= 0:
            raise ValueError("Number of iterations is 0. Set 'scan'>= 1.")

        self._instructions.append(f"REPEAT {int(num)}\n")
        _temp_dt = self._total_dwell_time
        try:
            yield
        finally:
            self._instructions.append("ENDREPEAT\n\n")

            # pauses should be multiplied by number of cycles as well
            self._total_dwell_time += int(num - 1) * (self._total_dwell_time - _temp_dt)

    def tic(self: GC) -> None:
        """
        Print the current time (hh:mm:ss) in message panel. The function is intended to be used before the execution
        of an operation or script to measure its time performances.

        :return: None
        """
        self._instructions.append('MSGDISPLAY 1, "START #TS"\n\n')

    def toc(self: GC) -> None:
        """
        Print the current time (hh:mm:ss) in message panel. The function is intended to be used after the execution
        of an operation or script to measure its time performances.

        :return: None
        """
        self._instructions.append('MSGDISPLAY 1, "END   #TS"\n')
        self._instructions.append('MSGDISPLAY 1, "---------------------"\n')
        self._instructions.append('MSGDISPLAY 1, " "\n\n')

    def instruction(self: GC, instr: str) -> None:
        """
        Adds a G-Code instruction passed as parameter to the PGM file.

        :param instr: Instruction line to be added to the PGM file. The ``\\n`` character is optional.
        :type instr: str
        :return: None
        """
        if instr.endswith("\n"):
            self._instructions.append(instr)
        else:
            self._instructions.append(instr + "\n")

    def load_program(self: GC, filename: str, task_id: int = 0) -> None:
        """
        Adds the instruction to LOAD a program in a G-Code file.

        :param filename: Name of the file that have to be loaded.
        :type filename: str
        :param task_id: ID of the task associated to the process.
        :type task_id: int
        :return: None
        """
        if task_id is None:
            task_id = 0

        file = self._get_filepath(filename=filename, extension="pgm")
        self._instructions.append(f'PROGRAM {int(task_id)} LOAD "{file}"\n')
        self._loaded_files.append(file.stem)

    def programstop(self: GC, task_id: int = 0):
        self._instructions.append(f"PROGRAM {int(task_id)} STOP\n")
        self._instructions.append(f"WAIT (TASKSTATUS({int(task_id)}, DATAITEM_TaskState) == TASKSTATE_Idle) -1\n")

    def remove_program(self: GC, filename: str, task_id: int = 0) -> None:
        """
        Adds the instruction to REMOVE a program from memory buffer in a G-Code file.

        :param filename: Name of the file to remove.
        :type filename: str
        :param task_id: ID of the task associated to the process.
        :type task_id: int
        :return: None
        """
        file = self._get_filepath(filename=filename, extension="pgm")
        if file.stem not in self._loaded_files:
            raise FileNotFoundError(
                f"The program {file} is not loaded. Load the file with 'load_program' before removing it."
            )
        self.programstop(task_id)
        self._instructions.append(f'REMOVEPROGRAM "{file.name}"\n')
        self._loaded_files.remove(file.stem)

    def farcall(self: GC, filename: str) -> None:
        """
        Adds the FARCALL instruction in a G-Code file.

        :param filename: Name of the file to call.
        :type filename: str
        :return: None
        """
        file = self._get_filepath(filename=filename, extension=".pgm")
        if file.stem not in self._loaded_files:
            raise FileNotFoundError(
                f"The program {file} is not loaded. Load the file with 'load_program' before the call."
            )
        self.dwell(self.short_pause)
        self._instructions.append(f'FARCALL "{file}"\n')

    def buffercall(self: GC, filename: str, task_id: int = 0) -> None:
        """
        Adds the BUFFEREDRUN instruction in a G-Code file.

        :param filename: Name of the file to call.
        :type filename: str
        :param task_id: ID of the task associated to the process.
        :type task_id: int
        :return: None
        """
        file = self._get_filepath(filename=filename, extension=".pgm")
        if file.stem not in self._loaded_files:
            raise FileNotFoundError(
                f"The program {file} is not loaded. Load the file with 'load_program' before the call."
            )
        self.dwell(self.short_pause)
        self.instruction("\n")
        self._instructions.append(f'PROGRAM {task_id} BUFFEREDRUN "{file}"\n')

    def call_list(self: GC, filenames: List[str], task_id: List[int] = 0) -> None:
        # Remove None from task_id
        task_id = list(filter(None, task_id))

        # Ensure task_id and filenames have the same length. If task_id is longer take a slice, pad with 0 otherwise.
        if len(task_id) > len(filenames):
            task_id = task_id[: len(filenames)]
        else:
            task_id = list(pad(task_id, len(filenames), 0))

        for fpath, t_id in zip(filenames, task_id):
            fpath = Path(fpath)
            self.load_program(str(fpath.resolve()), t_id)
            self.farcall(fpath.name)
            self.dwell(self.short_pause)
            self.remove_program(fpath.name, t_id)
            self.dwell(self.short_pause)
            self.instruction("\n\n")

    def write(self: GC, points: npt.NDArray[np.float32]) -> None:
        """
        The function convert the quintuple (X,Y,Z,F,S) to G-Code instructions. The (X,Y,Z) coordinates are
        transformed using the transformation matrix that takes into account the rotation of a given rotation_angle
        and the homothety to compensate the (effective) refractive index different from 1. Moreover, if the warp_flag
        is True the points are compensated along the z-direction.

        The transformed points are then parsed together with the feed rate and shutter state coordinate to produce
        the LINEAR movements.

        :param points: Numpy matrix containing the values of the tuple [X,Y,Z,F,S] coordinates.
        :type points: numpy.ndarray
        :return: None
        """

        if self.rotation_angle:
            print(" BEWARE, ANGLE MUST BE IN DEGREE! ".center(39, "*"))
            print(f" Rotation angle is {self.rotation_angle:.3f} deg. ".center(39, "*"))
            print()

        x, y, z, f_gc, s_gc = points

        # Transform points (rotations, z-compensation and flipping)
        x_gc, y_gc, z_gc = self.transform_points(x, y, z)

        # Convert points if G-Code commands
        args = [self._format_args(x, y, z, f) for (x, y, z, f) in zip(x_gc, y_gc, z_gc, f_gc)]
        for (arg, s) in zip_longest(args, s_gc):
            if s == 0 and self._shutter_on is True:
                self.dwell(self.short_pause)
                self.shutter("OFF")
                self.dwell(self.long_pause)
                self.instruction("\n")
            elif s == 1 and self._shutter_on is False:
                self.dwell(self.short_pause)
                self.shutter("ON")
                self.dwell(self.long_pause)
                self.instruction("\n")
            else:
                self._instructions.append(f"LINEAR {arg}\n")
        self.dwell(self.long_pause)
        self.instruction("\n")

    def close(self: GC, filename: str = None, verbose: bool = False) -> None:
        """
        Dumps all the instruction in self._instruction in a .pgm file.
        The filename is specified during the class instatiation. If no extension is present, the proper one is
        automatically added.

        :param filename: Different filename. The default is None, using self.filename.
        :type filename: str
        :param verbose: Print when G-Code export is finished. The default is False.
        :type verbose: bool
        :return: None
        """

        # get filename and add the proper file extension
        pgm_filename = Path(self.filename) if filename is None else Path(filename)
        pgm_filename = pgm_filename.stem + ".pgm"

        # create export directory (mimicking the POSIX mkdir -p command)
        if self.export_dir:
            exp_dir = Path(self.export_dir)
            if not exp_dir.is_dir():
                exp_dir.mkdir(parents=True, exist_ok=True)
            pgm_filename = exp_dir / pgm_filename

        # write instructions to file
        with open(pgm_filename, "w") as f:
            f.write("".join(self._instructions))
        self._instructions.clear()
        if verbose:
            print("G-code compilation completed.")

    # Geometrical transformations
    def transform_points(
        self: GC,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        z: npt.NDArray[np.float32],
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:

        # flip x, y coordinates
        x, y = self.flip(x, y)

        # translate points to new origin
        x -= self.new_origin[0]
        y -= self.new_origin[1]

        # roatate points
        point_matrix = np.stack((x, y, z), axis=-1)
        x_t, y_t, z_t = np.matmul(point_matrix, self.t_matrix()).T

        # compensate for warp
        if self.warp_flag:
            return self.compensate(x_t, y_t, z_t)
        return x_t, y_t, z_t

    def flip(
        self: GC,
        xc: npt.NDArray[np.float32],
        yc: npt.NDArray[np.float32],
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Flip the laser path along the x-, y- and z-coordinates
        :return: None
        """
        flip_toggle = np.array([1, 1])

        # reverse the coordinates arrays to flip
        if self.flip_x:
            flip_toggle[0] = -1
            xc = np.flip(xc)
        if self.flip_y:
            flip_toggle[1] = -1
            yc = np.flip(yc)

        # create flip matrix (+1 -> no flip, -1 -> flip)
        flip_matrix = np.diag(flip_toggle)

        # create the displacement matrix to map the transformed min/max coordinates to the original min/max coordinates)
        points_matrix = np.array([xc, yc])
        displacements = np.array([self.samplesize[0] - self.new_origin[0], self.samplesize[1] - self.new_origin[1]])
        S = np.multiply((1 - flip_toggle) / 2, displacements)

        # matrix multiplication and sum element-wise, add the displacement only to the flipped coordinates
        flip_x, flip_y = map(add, flip_matrix @ points_matrix, S)

        # update coordinates
        xc = flip_x
        yc = np.flip(flip_y) if (self.flip_x ^ self.flip_y) else flip_y

        return xc, yc

    def compensate(
        self: GC,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        z: npt.NDArray[np.float32],
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Returns the points compensated along z-direction for the refractive index, the offset and the glass warp.

        :param x: array of the x-coordinates
        :type x: np.ndarray
        :param y: array of the y-coordinates
        :type y: np.ndarray
        :param z: array of the z-coordinates
        :type z: np.ndarray
        :return: (x, y, zc) tuple of x, y and compensated z points
        :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
        """

        x_comp = deepcopy(np.array(x))
        y_comp = deepcopy(np.array(y))
        z_comp = deepcopy(np.array(z))

        zwarp = np.array([float(self.fwarp(x, y)) for x, y in zip(x_comp, y_comp)])
        z_comp = z_comp + zwarp / self.neff
        return x_comp, y_comp, z_comp

    def t_matrix(self: GC, dim: int = 3) -> npt.NDArray[np.float32]:
        """
        Given the rotation rotation_angle and the rifraction index, the function compute the transformation matrix as
        composition of rotatio matrix (RM) and a homothety matrix (SM).

        :param dim: Dimension of the transformation matrix. The default is 3.
        :type dim: int
        :return: Transformation matrix, TM = SM*RM
        :rtype: np.array

        :raise ValueError: Dimension not valid
        """
        RM = np.array(
            [
                [np.cos(self.rotation_angle), -np.sin(self.rotation_angle), 0],
                [np.sin(self.rotation_angle), np.cos(self.rotation_angle), 0],
                [0, 0, 1],
            ]
        )
        SM = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1 / self.neff],
            ]
        )
        t_mat = SM @ RM
        if dim == 3:
            return t_mat.T
        elif dim == 2:
            # export xy-submatrix
            ixgrid = np.ix_([0, 1], [0, 1])
            return t_mat[ixgrid].T
        else:
            raise ValueError(f"Dimension not valid. dim must be either 2 or 3. Given {dim}.")

    def antiwarp_management(self: GC, opt: bool, num: int = 16) -> interp2d:
        """
        It fetches an antiwarp function in the current working direcoty. If it doesn't exist, it lets you create a new
        one. The number of sampling points can be specified.

        :param opt: if True apply antiwarp.
        :type opt: bool
        :param num: number of sampling points
        :type num: int
        :return: warp function, `f(x, y)`
        :rtype: scipy.interpolate.interp2d
        """

        if not opt:

            def fwarp(_x, _y):
                return 0

        else:
            if not all(self.samplesize):
                raise ValueError(f"Wrong sample size dimensions. Given ({self.samplesize[0]}, {self.samplesize[1]}).")
            function_pickle = self.CWD / "fwarp.pkl"
            if function_pickle.is_file():
                with open(function_pickle, "rb") as f:
                    fwarp = dill.load(f)
            else:
                fwarp = self.antiwarp_generation(self.samplesize, num)
                with open(function_pickle, "wb") as f:
                    dill.dump(fwarp, f)
        return fwarp

    @staticmethod
    def antiwarp_generation(samplesize: Tuple[float, float], num: int, margin: float = 2) -> interp2d:
        """
        Helper for the generation of antiwarp function.
        The minimum number of data points required is (k+1)**2, with k=1 for linear, k=3 for cubic and k=5 for quintic
        interpolation.

        :param samplesize: glass substrate dimensions, (x-dim, y-dim)
        :type samplesize: Tuple(float, float)
        :param num: number of sampling points
        :type num: int
        :param margin: margin [mm] from the borders of the glass samples
        :type margin: float
        :return: warp function, `f(x, y)`
        :rtype: scipy.interpolate.interp2d
        """

        if num is None or num < 4**2:
            raise ValueError("I need more values to compute the interpolation.")

        num_side = int(np.ceil(np.sqrt(num)))
        xpos = np.linspace(margin, samplesize[0] - margin, num_side)
        ypos = np.linspace(margin, samplesize[1] - margin, num_side)
        xlist = []
        ylist = []
        zlist = []

        print("Insert focus height [in µm!] at:")
        for (x, y) in product(xpos, ypos):
            z_temp = input(f"X={x:.3f} Y={y:.3f}: \t")
            if z_temp == "":
                raise ValueError("You missed the last value.")
            else:
                xlist.append(x)
                ylist.append(y)
                zlist.append(float(z_temp) * 1e-3)
        # surface interpolation
        func_antiwarp = interp2d(xlist, ylist, zlist, kind="cubic")

        # plot the surface
        xprobe = np.linspace(-3, samplesize[0] + 3)
        yprobe = np.linspace(-3, samplesize[1] + 3)
        zprobe = func_antiwarp(xprobe, yprobe)
        ax = plt.axes(projection="3d")
        ax.contour3D(xprobe, yprobe, zprobe, 200, cmap="viridis")
        ax.set_xlabel("X [mm]"), ax.set_ylabel("Y [mm]"), ax.set_zlabel("Z [mm]")
        # plt.show()
        return func_antiwarp

    # Private interface
    def _format_args(self: GC, x: float = None, y: float = None, z: float = None, f: float = None) -> str:
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
            args.append(f"X{x:.{self.output_digits}f}")
        if y is not None:
            args.append(f"Y{y:.{self.output_digits}f}")
        if z is not None:
            args.append(f"Z{z:.{self.output_digits}f}")
        if f is not None:
            if f < 10 ** (-self.output_digits):
                raise ValueError("Try to move with F <= 0.0 mm/s. Check speed parameter.")
            args.append(f"F{f:.{self.output_digits}f}")
        args = " ".join(args)
        return args

    @staticmethod
    def _get_filepath(filename: str, filepath: Optional[str] = None, extension: Optional[str] = None) -> Path:
        """
        The function takes a filename and (optional) filepath, it merges the two and return a filepath.
        An extension parameter can be given as input. In that case the function also checks if the filename has
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

        if filename is None:
            raise ValueError("Given filename is None. Give a valid filename.")

        path = Path(filename) if filepath is None else Path(filepath) / filename
        if extension is None:
            return path

        ext = "." + extension.split(".")[-1].lower()
        if path.suffix != ext:
            raise ValueError(f"Given filename has wrong extension. Given {filename}, required {ext}.")
        return path

    def _enter_axis_rotation(self: GC, angle: Optional[float] = None) -> None:
        self.comment("ACTIVATE AXIS ROTATION")
        self._instructions.append(f"LINEAR X{0.0:.6f} Y{0.0:.6f} Z{0.0:.6f} F{self.speed_pos:.6f}\n")
        self._instructions.append("G84 X Y\n")

        if angle is None and self.aerotech_angle == 0.0:
            return

        angle = self.aerotech_angle if angle is None else float(angle % 360)
        self._instructions.append(f"G84 X Y F{angle}\n\n")

    def _exit_axis_rotation(self: GC) -> None:
        self.comment("DEACTIVATE AXIS ROTATION")
        self._instructions.append(f"LINEAR X{0.0:.6f} Y{0.0:.6f} Z{0.0:.6f} F{self.speed_pos:.6f}\n")
        self._instructions.append("G84 X Y\n")


class PGMTrench(PGMCompiler):
    def __init__(self, param: dict, trench_columns: List[TrenchColumn]):
        super().__init__(param)
        self.trench_columns = listcast(trench_columns)
        self._param = param

    def write(self, dirname: str = "trench"):
        """
        Helper function for the compilation of trench columns.
        For each trench in the column, the function first compile a PGM file for border (or wall) and for the floor
        inside a directory given by the user (base_folder).
        Secondly, the function produce a FARCALL.pgm program to fabricate all the trenches in the column.

        :param dirname: Name of the directory in which the .pgm file will be written. The path is relative to the
        current file (.\\dirname\\)
        :type dirname: str
        :return: None
        """

        for col_idx, col in enumerate(self.trench_columns):
            # Prepare directories for export .pgm files
            col_dir = os.path.join(os.getcwd(), self.export_dir, dirname, f"trenchCol{col_idx + 1:03}")
            os.makedirs(col_dir, exist_ok=True)

            # Export walls and floors .pgm scripts
            for i, trench in enumerate(col):
                filename = os.path.join(col_dir, f"trench{i + 1:03}")
                self._export_path(filename, trench, speed=col.speed)

            # Export FARCALL for each trench column
            col_param = Dotdict(self._param.copy())
            col_param.filename = os.path.join(os.getcwd(), self.export_dir, dirname, f"FARCALL{col_idx + 1:03}.pgm")
            with PGMCompiler(**col_param) as G:
                G.dvar(["ZCURR"])

                for nbox in range(col.nboxz):
                    for t_index, trench in enumerate(col):
                        # load filenames (wall/floor)
                        wall_filename = f"trench{t_index + 1:03}_wall.pgm"
                        floor_filename = f"trench{t_index + 1:03}_floor.pgm"
                        wall_path = os.path.join(col.base_folder, f"trenchCol{col_idx + 1:03}", wall_filename)
                        floor_path = os.path.join(
                            col.base_folder,
                            f"trenchCol{col_idx + 1:03}",
                            floor_filename,
                        )

                        x0, y0 = trench.block.exterior.coords[0]
                        z0 = (nbox * col.h_box - col.z_off) / super().neff
                        G.comment(f"+--- COLUMN #{col_idx + 1}, TRENCH #{t_index + 1} LEVEL {nbox + 1} ---+")

                        # WALL
                        G.load_program(wall_path)
                        G.instruction(
                            f'MSGDISPLAY 1, "COL {col_idx + 1:03}, TR {t_index + 1:03}, LV {nbox + 1:03}, W"\n'
                        )
                        G.shutter("OFF")
                        if col.u:
                            G.instruction(f"LINEAR U{col.u[0]:.6f}")
                        G.move_to(
                            [x0 - self.new_origin[0], y0 - self.new_origin[1], z0],
                            speed_pos=col.speed_closed,
                        )

                        G.instruction(f"$ZCURR = {z0:.6f}")
                        G.shutter("ON")
                        with G.repeat(col.n_repeat):
                            G.farcall(wall_filename)
                            G.instruction(f"$ZCURR = $ZCURR + {col.deltaz / super().neff:.6f}")
                            G.instruction("LINEAR Z$ZCURR")
                        G.remove_program(wall_filename)

                        # FLOOR
                        G.shutter(state="OFF")
                        G.load_program(floor_path)
                        G.instruction(
                            f'MSGDISPLAY 1, "COL {col_idx + 1:03}, TR {t_index + 1:03}, LV {nbox + 1:03}, F"\n'
                        )
                        if col.u:
                            G.instruction(f"LINEAR U{col.u[-1]:.6f}")
                        G.shutter(state="ON")
                        G.farcall(floor_filename)
                        G.shutter("OFF")
                        if col.u:
                            G.instruction(f"LINEAR U{col.u[0]:.6f}")
                        G.remove_program(floor_filename)
                G.instruction("MSGCLEAR -1\n")
            del G

        # MAIN FARCALL, calls all columns .pgm scripts
        if self.trench_columns:
            # MAIN FARCALL parameters
            main_param = Dotdict(self._param.copy())
            main_param.filename = os.path.join(dirname, "MAIN.pgm")
            main_param.aerotech_angle = None

            t_list = [
                os.path.join(col.base_folder, f"FARCALL{idx + 1:03}.pgm") for idx, col in enumerate(self.trench_columns)
            ]
            with PGMCompiler(**main_param) as G:
                G.call_list(t_list)

    def _export_path(self, filename: str, trench: Trench, speed: float = 4):
        """
        Helper function for the export of the wall and floor instruction of a Trench obj.

        :param filename: Base filename for the wall.pgm and floor.pgm files. If the filename ends with the '.pgm'
        extension, the latter it is stripped and replaced with '_wall.pgm' and '_floor.pgm' to differentiate the two
        paths.
        :type filename: str
        :param trench: Trench obj to export.
        :type trench: Trench
        :param speed: Traslation speed during fabrication [mm/s]. The default is 4 [mm/s].
        :type speed: float
        :return: None
        """
        if filename is None:
            raise ValueError("No filename given.")
        if filename.endswith(".pgm"):
            filename = filename.split(".")[0]

        # Wall script
        points = np.array(trench.block.exterior.coords.xy).T
        self._write_array(filename + "_wall.pgm", points, speed)
        del points

        # Floor script
        points = []
        [points.extend(np.stack(path, axis=-1)) for path in trench.trench_paths()]
        self._write_array(filename + "_floor.pgm", np.array(points), speed)
        del points

    def _write_array(self, pgm_filename: str, points: np.ndarray, f_val: float):
        """
        Helper function that produces a PGM file for a 3D matrix of points at a given traslation speed,
        without shuttering operations.
        The function parse the points input matrix, applies the rotation and homothety transformations and parse all
        the LINEAR instructions.

        :param pgm_filename: Filename of the file in which the G-Code instructions will be written.
        :type pgm_filename: str
        :param points: 3D points matrix. If the points matrix is 2D it is intended as [x,y] coordinates.
        :type points: np.ndarray
        :param f_val: Traslation speed value.
        :type f_val: float, np.ndarray or list
        :return: None
        """

        points -= np.array([self.new_origin[0], self.new_origin[1]]).T

        if points.shape[-1] == 2:
            x_arr, y_arr = np.matmul(points, self.t_matrix(dim=2)).T
            z_arr = [None]
        else:
            x_arr, y_arr, z_arr = np.matmul(points, self.t_matrix()).T

        instr = [self._format_args(x, y, z, f) for (x, y, z, f) in zip_longest(x_arr, y_arr, z_arr, listcast(f_val))]
        gcode_instr = [f"LINEAR {line}\n" for line in instr]
        with open(pgm_filename, "w") as f:
            f.write("".join(gcode_instr))


def main():
    from femto.Waveguide import Waveguide

    # Parameters
    PARAM_WG = Dotdict(scan=6, speed=20, radius=15, pitch=0.080, int_dist=0.007, lsafe=3, samplesize=(25, 3))
    PARAM_GC = Dotdict(filename="testPGM.pgm", samplesize=PARAM_WG.samplesize, rotation_angle=2.0, flip_x=True)

    # Build paths
    chip = [Waveguide(**PARAM_WG) for _ in range(2)]
    for i, wg in enumerate(chip):
        wg.start([-2, -wg.pitch / 2 + i * wg.pitch, 0.035])
        wg.linear([wg.lsafe, 0, 0])
        wg.sin_mzi((-1) ** i * wg.dy_bend, arm_length=1.0)
        wg.linear([wg.x_end, wg.lasty, wg.lastz], mode="ABS")
        wg.end()

    # Compilation
    with PGMCompiler(**PARAM_GC) as G:
        G.set_home([0, 0, 0])
        with G.repeat(PARAM_WG["scan"]):
            for i, wg in enumerate(chip):
                G.comment(f"Modo: {i}")
                G.write(wg.points)
        G.move_to([None, 0, 0.1])
        G.set_home([0, 0, 0])


if __name__ == "__main__":
    main()
