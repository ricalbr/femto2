from __future__ import annotations

import collections
import contextlib
import copy
import itertools
import math
import pathlib
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Deque
from typing import Generator
from typing import TypeVar

import attrs
import dill
import numpy as np
import numpy.typing as npt
from femto import logger
from femto.helpers import flatten
from femto.helpers import listcast
from femto.helpers import pad
from femto.helpers import remove_repeated_coordinates
from scipy import interpolate

# Create a generic variable that can be 'PGMCompiler', or any subclass.
GC = TypeVar('GC', bound='PGMCompiler')
nparray = npt.NDArray[np.float32]


@attrs.define
class Laser:
    """Class representing a Laser.

    The class contains all the info (``name``, ``lab``) and configuration (``axis``, ``pin``, ``mode``) of the given
    laser for G-Code files.
    """

    name: str  #: name of the laser source
    lab: str  #: name of the fabrication line
    axis: str  #: Axis of the PSO card
    pin: int | None  #: Pin of the PSO card
    mode: int | None  #: Mode od the PSO card


@attrs.define(kw_only=True, repr=False)
class PGMCompiler:
    """Class representing a PGMCompiler.

    The class contains all the parameters and all the method for translating a series of 3D points to G-Code files.
    """

    filename: str  #: Filename of the .pgm file.
    n_glass: float = 1.50  #: Glass refractive index.
    n_environment: float = 1.33  #: Environment refrative index.
    export_dir: str = ''  #: Name of the directory onto which .pgm files will be exported. Default is current directoru.
    laser: str = 'PHAROS'  #: Name of the laser source.
    shift_origin: tuple[float, float] = (0.0, 0.0)  #: Shift the cooordinates of the origin to this new point. `[mm]`.
    samplesize: tuple[float, float] = (100, 50)  #: `(x, y)`-size of the substrate `[mm]`.
    rotation_angle: float = 0.0  #: Apply a rotation matrix of this angle to all the points `[deg]`.
    aerotech_angle: float = 0.0  #: Apply part rotation (G84) with this angle as parameter `[deg]`.
    long_pause: float = 0.5  #: Long pause value `[s]`.
    short_pause: float = 0.05  #: Short pause value `[s]`.
    speed_pos: float = 5.0  #: Positioning speed `[mm/s]`.
    output_digits: int = 6  #: Number of output digits for formatting G-Code instructinos.
    home: bool = False  #: Flag, if True the fabrication will finish in `(0,0,0)`.
    warp_flag: bool = False  #: Flag, toggle the warp compensation.
    flip_x: bool = False  #: Flag, if True the x-coordinates will be flipped
    flip_y: bool = False  #: Flag, if True the y-coordinates will be flipped
    minimal_gcode: bool = False  #: Flag, if True redundant movements are suppressed.
    verbose: bool = True  #: Flag, if True output informations during G-Code compilation.

    # Basic parameters
    CWD: pathlib.Path = attrs.field(default=pathlib.Path.cwd())
    # Load warp function
    fwarp: Callable[[nparray, nparray], nparray] = attrs.field(
        default=attrs.Factory(lambda self: self.warp_management(self.warp_flag), takes_self=True)
    )

    _total_dwell_time: float = attrs.field(alias='_total_dwell_time', default=0.0)
    _shutter_on: bool = attrs.field(alias='_shutter_on', default=False)
    _mode_abs: bool = attrs.field(alias='_mode_abs', default=True)
    _lasers: dict = attrs.field(factory=dict)
    _instructions: Deque[str] = attrs.field(factory=collections.deque)
    _loaded_files: list[str] = attrs.field(factory=list)
    _dvars: list[str] = attrs.field(factory=list)

    def __init__(self, **kwargs):
        filtered = {att.name: kwargs[att.name] for att in self.__attrs_attrs__ if att.name in kwargs}
        self.__attrs_init__(**filtered)

    def __attrs_post_init__(self) -> None:

        self._lasers = {
            'ant': Laser(name='ANT', lab='DIAMOND', axis='Z', pin=0, mode=1),
            'uwe': Laser(name='UWE', lab='FIRE', axis='X', pin=None, mode=None),
            'carbide': Laser(name='CARBIDE', lab='FIRE', axis='X', pin=2, mode=0),
            'pharos': Laser(name='PHAROS', lab='CAPABLE', axis='X', pin=3, mode=0),
        }

        # File initialization
        if self.filename is None:
            logger.error('Filename is None.')
            raise ValueError("Filename is None, set 'filename' attribute")
        self._instructions: Deque[str] = collections.deque()
        self._loaded_files: list[str] = []
        self._dvars: list[str] = []

        # Load warp function
        self.fwarp: Callable[
            [npt.NDArray[np.float32], npt.NDArray[np.float32]], npt.NDArray[np.float32]
        ] = self.warp_management(self.warp_flag)

        # Set rotation angle in radians for matrix rotations
        if self.rotation_angle:
            self.rotation_angle = math.radians(self.rotation_angle % 360)
            logger.debug(f'Rotation angle is set to {self.rotation_angle}.')
        else:
            self.rotation_angle = float(0.0)

        # Set AeroTech angle between 0 and 359 for G84 command
        if self.aerotech_angle:
            self.aerotech_angle = self.aerotech_angle % 360
            logger.debug(f'Axis rotation (G84) angle is set to {self.aerotech_angle}.')
        else:
            self.aerotech_angle = float(0.0)

    @classmethod
    def from_dict(cls: type(GC), param: dict[str, Any], **kwargs) -> GC:
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
        p.update(kwargs)

        logger.debug(f'Create {cls.__name__} object from dictionary.')
        return cls(**p)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}'

    def __enter__(self) -> GC:
        """Context manager entry.

        The context manager takes care to automatically add the proper header file (from the `self.laser` attribute,
        add the G84 activation instruction (if needed) and printing some warning info for rotations.

        It can be use like:

        >>>with femto.PGMCompiler(filename, ind_rif) as gc:
        >>>     <code block>

        Returns
        -------
        The object itself.
        """

        self.header()
        self.dwell(1.0)
        self.instruction('\n')

        if self.rotation_angle:
            msg = f' BEWARE, ANGLE MUST BE IN DEGREE! Rotation angle is {self.rotation_angle:.3f} deg. '.center(69, '*')
            if self.verbose:
                logger.warning(msg)
            else:
                logger.debug(msg)
        if self.aerotech_angle:
            msg = f' BEWARE, ANGLE MUST BE IN DEGREE! G84 angle is {self.aerotech_angle:.3f} deg. '.center(69, '*')
            if self.verbose:
                logger.warning(msg)
            else:
                logger.debug(msg)
            self._enter_axis_rotation(angle=self.aerotech_angle)
        return self

    def __exit__(
        self,
        exc_type: type(BaseException) | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Context manager exit.

        Returns
        -------
        None
        """

        if self.aerotech_angle:
            self._exit_axis_rotation()
            self._instructions.append('\n')
        if self.home:
            self.go_init()
        self.close()
        logger.debug('Close PGM file.')

    @property
    def xsample(self) -> float:
        """`x`-dimension of the sample

        Returns
        -------
        The absolute value of the `x` element of the samplesize array.
        """
        xsample = float(abs(self.samplesize[0]))
        logger.debug(f'Return xsample = {xsample}')
        return xsample

    @property
    def ysample(self) -> float:
        """`y`-dimension of the sample

        Returns
        -------
        The absolute value of the `y` element of the samplesize array.
        """
        ysample = float(abs(self.samplesize[1]))
        logger.debug(f'Return ysample = {ysample}')
        return ysample

    @property
    def neff(self) -> float:
        """Effective refractive index.

        Returns
        -------
        Effective refractive index of the waveguide.
        """
        logger.debug(f'Effective refractive index is {self.n_glass / self.n_environment}')
        return self.n_glass / self.n_environment

    @property
    def pso_axis(self) -> str:
        """PSO command axis.

        If the laser is ANT, return Z, otherwise return X.

        Returns
        -------
        Lable for the PSO commands.
        """
        try:
            ax = self._lasers[self.laser.lower()].axis
            logger.debug(f'Return the PSO axis {ax}.')
            return ax
        except (KeyError, AttributeError):
            logger.error(f'Laser can only be ANT, CARBIDE, PHAROS or UWE. Given {self.laser}.')
            raise ValueError(f'Laser can only be ANT, CARBIDE, PHAROS or UWE. Given {self.laser}.')

    @property
    def tshutter(self) -> float:
        """Shuttering delay.

        Function that gives the shuttering delay time given the fabrication laboratory.

        Returns
        -------
        Delay time [s].
        """
        if self.laser.lower() not in ['ant', 'carbide', 'pharos', 'uwe']:
            logger.error(f'Laser can be only ANT, CARBIDE, PHAROS or UWE. Given {self.laser}.')
            raise ValueError(f'Laser can be only ANT, CARBIDE, PHAROS or UWE. Given {self.laser}.')
        if self.laser.lower() == 'uwe':
            # mechanical shutter
            logger.debug('Shuttering time for mechanical shutter: 0.005.')
            return 0.005
        else:
            # pockels cell
            logger.debug('Shuttering time for pockels cell: 0.000.')
            return 0.000

    @property
    def dwell_time(self) -> float:
        """Total DWELL time.

        Returns
        -------
        Total pausing times in the G-code script.
        """
        logger.debug(f'Return total dwell time {self._total_dwell_time}.')
        return self._total_dwell_time

    def header(self) -> None:
        """Add header instructions.

        It reads the header file for the laser cutter and adds it to the instructions list.
        The user can specify the fabrication line to work in ``ANT``, ``CARBIDE``, ``PHAROS`` or ``UWE`` laser when
        the G-Code Compiler obj is instantiated.

        Returns
        -------
        None
        """

        try:
            par = attrs.asdict(self._lasers[self.laser.lower()])
            logger.debug(f'Load header for laser {self.laser.lower()}.')
        except (KeyError, AttributeError):
            logger.error(f'Laser can only be ANT, CARBIDE, PHAROS or UWE. Given {self.laser}.')
            raise ValueError(f'Laser can only be ANT, CARBIDE, PHAROS or UWE. Given {self.laser}.')

        with open(pathlib.Path(__file__).parent / 'utils' / 'header.txt') as f:
            for line in f:
                self._instructions.append(line.format(**par))
        if self.laser.lower() == 'uwe':
            for idx, instr in enumerate(self._instructions):
                if instr.startswith('PSOOUTPUT'):
                    # TODO: test questa cosa!!
                    del self._instructions[idx]
                    break
        self.instruction('\n')

    def dvar(self, variables: list[str]) -> None:
        """Add declared variable instructions.

        Adds the declaration of variables in a G-Code file.

        Parameters
        ----------
        variables : list(str)
            List of G-Code variables.

        Returns
        -------
        None
        """
        variables = listcast(flatten(variables))
        args = ' '.join(['${}'] * len(variables)).format(*variables)
        logger.debug(f'Add variables {variables}.')
        self._instructions.appendleft(f'DVAR {args}\n\n')

        # keep track of all variables
        self._dvars.extend([var.lower() for var in variables])

    def mode(self, mode: str = 'abs') -> None:
        """Movements mode.

        The function appends the mode string to the list of instructions. If the string is not 'abs' or 'inc',
        it will raise a ValueError.

        Parameters
        ----------
        mode: str, optional
            Operation mode of the movements commands. It can be ABSOLUTE (G90) or INCREMENTAL (G91). The default
            value is ABSOLUTE.

        Returns
        -------
        None
        """
        if mode is None or mode.lower() not in ['abs', 'inc']:
            logger.error(f'Mode should be either ABSOLUTE (ABS) or INCREMENTAL (INC). {mode} was given.')
            raise ValueError(f'Mode should be either ABSOLUTE (ABS) or INCREMENTAL (INC). {mode} was given.')

        if mode.lower() == 'abs':
            self._instructions.append('G90 ; ABSOLUTE\n')
            self._mode_abs = True
            logger.debug('Switch to ABSOLUTE mode.')
        else:
            self._instructions.append('G91 ; INCREMENTAL\n')
            self._mode_abs = False
            logger.debug('Switch to INCREMENTAL mode.')

    def comment(self, comstring: str) -> None:
        """Add a comment.

        Adds a comment to a G-Code file.

        Parameters
        ----------
        comstring : str
            Comment string.

        Returns
        -------
        None
        """

        if comstring:
            self._instructions.append(f'\n; {comstring}\n')
        else:
            self._instructions.append('\n')
        logger.debug('Add a comment.')

    def shutter(self, state: str) -> None:
        """Open and close shutter.

        Adds the instruction to open (or close) the shutter to a G-Code file.
        The user specifies the state and the function compare it to the current state of the shutter (which is
        tracked internally during the compilation of the .pgm file).

        Parameters
        ----------
        state: str
            State of the shutter (`ON` or `OFF`).

        Returns
        -------
        None
        """

        if state is None or state.lower() not in ['on', 'off']:
            logger.error(f'Shutter state should be ON or OFF. Given {state}.')
            raise ValueError(f'Shutter state should be ON or OFF. Given {state}.')

        if state.lower() == 'on' and self._shutter_on is False:
            self._shutter_on = True
            self._instructions.append(f'PSOCONTROL {self.pso_axis} ON\n')
            logger.debug('Open the shutter.')
        elif state.lower() == 'off' and self._shutter_on is True:
            self._shutter_on = False
            self._instructions.append(f'PSOCONTROL {self.pso_axis} OFF\n')
            logger.debug('Close the shutter.')
        else:
            pass

    def dwell(self, pause: float) -> None:
        """Add pause.

        Parameters
        ----------
        pause : float
            Pause duration [s].

        Returns
        -------
        None
        """

        if pause is None or pause == float(0.0):
            return None
        self._instructions.append(f'G4 P{np.fabs(pause)} ; DWELL\n')
        logger.debug(f'Add pause {np.fabs(pause)}')
        self._total_dwell_time += np.fabs(pause)

    def set_home(self, home_pos: list[float]) -> None:
        """Set coordinates of present position.

        The user can set the current Aerotech postition to a particular set of coordinates, given as an input list.
        A variable can be excluded if set to ``None``. The function can be used to set a user-defined home position.

        Parameters
        ----------
        home_pos: list(float)
            List of coordinates `(x, y, z)` of the new value for the current point [mm].

        Returns
        -------
        None
        """

        if np.size(home_pos) != 3:
            logger.error(f'Given final position is not valid. 3 values required, given {np.size(home_pos)}.')
            raise ValueError(f'Given final position is not valid. 3 values required, given {np.size(home_pos)}.')

        if all(coord is None for coord in home_pos):
            logger.error('Given home position is (None, None, None). Give a valid home position.')
            raise ValueError('Given home position is (None, None, None). Give a valid home position.')

        args = self._format_args(*home_pos)
        self._instructions.append(f'G92 {args}\n')
        logger.debug(f'Soft-reset of coordinates (G92) to {args}.')

    def move_to(self, position: list[float | None], speed_pos: float | None = None) -> None:
        """Move to target.

        Utility function to move to a given position with the shutter ``OFF``.
        The user can specify the target position and the positioning speed.

        Parameters
        ----------
        position: list(float, optional)
            List of target coordinates `(x, y, z)` [mm].
        speed_pos: float, optional
            Translation speed. The default value is ``self.speed_pos``.

        Returns
        -------
        None
        """
        if len(position) != 3:
            logger.error('Given final position is not valid.')
            raise ValueError(f'Given final position is not valid. 3 values required, given {len(position)}.')

        if speed_pos is None and self.speed_pos is None:
            logger.error('The positioning speed is None.')
            raise ValueError('The positioning speed is None. Set the "speed_pos" attribute or give a valid value.')
        speed_pos = self.speed_pos if speed_pos is None else speed_pos

        # close the shutter before the movements
        if self._shutter_on is True:
            self.shutter('OFF')

        xp, yp, zp = position
        args = self._format_args(xp, yp, zp, speed_pos)
        if all(coord is None for coord in position):
            self._instructions.append(f'{args}\n')
        else:
            self._instructions.append(f'G1 {args}\n')
            logger.debug(f'Move to {args}')
        self.dwell(self.long_pause)
        self.instruction('\n')

    def go_origin(self) -> None:
        """Return to origin.

        Utility function, returns to the origin `(0,0,0)` with shutter ``OFF``.

        Returns
        -------
        None
        """
        self.comment('HOMING')
        self.move_to([0.0, 0.0, 0.0])

    def go_init(self) -> None:
        """Return to initial point.

        Utility function to return to the initial point of fabrication `(-2,0,0)` with shutter ``OFF``.

        Returns
        -------
        None
        """
        self.move_to([-2, 0, 0])

    @contextlib.contextmanager
    def axis_rotation(self, angle: float | None = None) -> Generator[PGMCompiler, None, None]:
        """Aerotech axis rotation (G84).

        Context manager for the G84 command. The user can specify the angle (in degree) of the axis rotation.

        Parameters
        ----------
        angle : float
            Value [deg] of the rotation angle

        Yields
        ------
        Current PGMCompiler instance
        """
        self._enter_axis_rotation(angle=angle)
        try:
            yield self
        finally:
            self._exit_axis_rotation()

    @contextlib.contextmanager
    def for_loop(self, var: str, num: int) -> Generator[PGMCompiler, None, None]:
        """Foor loop instruction.

        Context manager that manages a ``FOR`` loops in a G-Code file.

        Parameters
        ----------
        var : str
            Iterating variable.
        num : int
            Number of iterations.

        Yields
        ------
        Current PGMCompiler instance
        """
        if num is None:
            logger.error("Number of iterations is None. Give a valid 'scan' attribute value.")
            raise ValueError("Number of iterations is None. Give a valid 'scan' attribute value.")
        if num <= 0:
            logger.error("Number of iterations is 0. Set 'scan'>= 1.")
            raise ValueError("Number of iterations is 0. Set 'scan'>= 1.")

        if var is None:
            logger.error('Given variable is None. Give a valid varible.')
            raise ValueError('Given variable is None. Give a valid varible.')
        if var.lower() not in self._dvars:
            logger.error(f'Given variable has not beed declared. Use dvar() method to declare ${var} variable.')
            raise ValueError(f'Given variable has not beed declared. Use dvar() method to declare ${var} variable.')

        self._instructions.append(f'FOR ${var} = 0 TO {int(num) - 1}\n')
        logger.debug(f'Init FOR loop with {num} iterations.')
        _temp_dt = self._total_dwell_time
        try:
            yield self
        finally:
            self._instructions.append(f'NEXT ${var}\n\n')
            logger.debug('End FOR loop.')

            # pauses should be multiplied by number of cycles as well
            self._total_dwell_time += int(num - 1) * (self._total_dwell_time - _temp_dt)

    @contextlib.contextmanager
    def repeat(self, num: int) -> Generator[PGMCompiler, None, None]:
        """Repeat loop instruction.

        Context manager that manages a ``REPEAT`` loops in a G-Code file.

        Parameters
        ----------
        num : int
            Number of iterations.

        Yields
        ------
        Current PGMCompiler instance
        """
        if num is None:
            logger.error("Number of iterations is None. Give a valid 'scan' attribute value.")
            raise ValueError("Number of iterations is None. Give a valid 'scan' attribute value.")
        if num <= 0:
            logger.error("Number of iterations is 0. Set 'scan'>= 1.")
            raise ValueError("Number of iterations is 0. Set 'scan'>= 1.")

        self._instructions.append(f'REPEAT {int(num)}\n')
        logger.debug(f'Init REPEAT loop with {num} iterations.')
        _temp_dt = self._total_dwell_time
        try:
            yield self
        finally:
            self._instructions.append('ENDREPEAT\n\n')
            logger.debug('End REPEAT loop.')

            # pauses should be multiplied by number of cycles as well
            self._total_dwell_time += int(num - 1) * (self._total_dwell_time - _temp_dt)

    def tic(self) -> None:
        """Start time measure.

        Print the current time (hh:mm:ss) in message panel. The function is intended to be used *before* the execution
        of an operation or script to measure its time performances.

        Returns
        -------
        None
        """
        self._instructions.append('MSGDISPLAY 1, "START #TS"\n\n')
        logger.debug('Add starting time string.')

    def toc(self) -> None:
        """Stop time measure.

        Print the current time (hh:mm:ss) in message panel. The function is intended to be used *after* the execution
        of an operation or script to measure its time performances.

        Returns
        -------
        None
        """
        self._instructions.append('MSGDISPLAY 1, "END   #TS"\n')
        self._instructions.append('MSGDISPLAY 1, "---------------------"\n')
        self._instructions.append('MSGDISPLAY 1, " "\n\n')
        logger.debug('Add ending time string.')

    def instruction(self, instr: str) -> None:
        """Add G-Code instruction.

        Adds a G-Code instruction passed as parameter to the PGM file.

        Parameters
        ----------
        instr : str
            G-Code instruction to add.

        Returns
        -------
        None
        """
        if instr.endswith('\n'):
            self._instructions.append(instr)
        else:
            self._instructions.append(instr + '\n')
        logger.debug(f'Add instruction: {instr.strip()}')

    def load_program(self, filename: str, task_id: int = 2) -> None:
        """Load G-code script.

        Adds the instruction to `LOAD` an external G-Code script in the driver memory. The function is used for
        `FARCALL` programs.

        Parameters
        ----------
        filename : str
            Filename of the G-code script.
        task_id : int, optional
            Task ID number onto which the program will be loaded (and executed). The default value is 2.

        Returns
        -------
        None
        """
        if task_id is None:
            logger.debug('Set task ID to 2.')
            task_id = 2

        file = self._get_filepath(filename=filename, extension='pgm')
        self._instructions.append(f'PROGRAM {int(task_id)} LOAD "{file}"\n')
        logger.debug(f'Load file {file}.')
        self._loaded_files.append(file.stem)

    def remove_program(self, filename: str, task_id: int = 2) -> None:
        """Remove program from memory buffer.

        Adds the instruction to `REMOVE` a program from memory buffer in a G-Code file.

        Parameters
        ----------
        filename : str
            Filename of the G-code script.
        task_id : int, optional
            Task ID number onto which the program will be loaded (and executed). The default value is 2.

        Returns
        -------
        None
        """
        file = self._get_filepath(filename=filename, extension='pgm')
        if file.stem not in self._loaded_files:
            logger.error(f'The program {file} is not loaded.')
            raise FileNotFoundError(
                f"The program {file} is not loaded. Load the file with 'load_program' before removing it."
            )
        self.programstop(task_id)
        self._instructions.append(f'REMOVEPROGRAM "{file.name}"\n')
        logger.debug(f'Remove file {file}.')
        self._loaded_files.remove(file.stem)

    def programstop(self, task_id: int = 2) -> None:
        """Program stop.

        Add the instruction to stop the execution of an external G-Code script and empty the Task in which the
        program was running.

        Parameters
        ----------
        task_id : int, optional
            Task ID number onto which the program will be loaded (and executed). The default value is 2.

        Returns
        -------
        None
        """
        self._instructions.append(f'PROGRAM {int(task_id)} STOP\n')
        self._instructions.append(f'WAIT (TASKSTATUS({int(task_id)}, DATAITEM_TaskState) == TASKSTATE_Idle) -1\n')

    def farcall(self, filename: str) -> None:
        """FARCALL instruction.

        Adds the instruction to call and execute an external G-Code script in the current G-Code file.

        Parameters
        ----------
        filename : str
            Filename of the G-code script.

        Returns
        -------
        None
        """
        file = self._get_filepath(filename=filename, extension='.pgm')
        if file.stem not in self._loaded_files:
            logger.error(f"The program {file} is not loaded. Load the file with 'load_program' before the call.")
            raise FileNotFoundError(
                f"The program {file} is not loaded. Load the file with 'load_program' before the call."
            )
        self.dwell(self.short_pause)
        self._instructions.append(f'FARCALL "{file}"\n')
        logger.debug(f'Call file {file}.')

    def bufferedcall(self, filename: str, task_id: int = 2) -> None:
        """BUFFEREDCALL instruction.

        Adds the instruction to run an external G-Code script in queue mode.

        Parameters
        ----------
        filename : str
            Filename of the G-code script.
        task_id : int, optional
            Task ID number onto which the program will be loaded (and executed). The default value is 2.

        Returns
        -------
        None
        """
        file = self._get_filepath(filename=filename, extension='.pgm')
        if file.stem not in self._loaded_files:
            logger.error(f"The program {file} is not loaded. Load the file with 'load_program' before the call.")
            raise FileNotFoundError(
                f"The program {file} is not loaded. Load the file with 'load_program' before the call."
            )
        self.dwell(self.short_pause)
        self.instruction('\n')
        self._instructions.append(f'PROGRAM {task_id} BUFFEREDRUN "{file}"\n')
        logger.debug(f'Call buffered file {file}.')

    def farcall_list(self, filenames: list[str], task_id: list[int] | int = 2) -> None:
        """Chiamatutto.

        Load and execute sequentially a list of G-Code scripts.

        Parameters
        ----------
        filenames : list(str)
            List of filenames of the G-code scripts to be executed.
        task_id : list(int), optional
            Task ID number onto which the program will be loaded (and executed). The default value is 2 for all the
            scripts in the filename list.

        Returns
        -------
        None
        """
        task_id = list(filter(None, listcast(task_id)))  # Remove None from task_id

        # Ensure task_id and filenames have the same length. If task_id is longer take a slice, pad with 0 otherwise.
        if len(task_id) > len(filenames):
            task_id = task_id[: len(filenames)]
        else:
            task_id = list(pad(task_id, len(filenames), 2))

        for fpath, t_id in zip(filenames, task_id):
            file = pathlib.Path(fpath)
            self.load_program(str(file), t_id)
            self.farcall(file.name)
            self.dwell(self.short_pause)
            self.remove_program(file.name, t_id)
            self.dwell(self.short_pause)
            self.instruction('\n\n')

    def write(self, points: nparray) -> None:
        """
        The function convert the quintuple (X,Y,Z,F,S) to G-Code instructions. The (X,Y,Z) coordinates are
        transformed using the transformation matrix that takes into account the rotation of a given rotation_angle
        and the homothety to compensate the (effective) refractive index different from 1. Moreover, if the warp_flag
        is True the points are compensated along the z-direction.

        The transformed points are then parsed together with the feed rate and shutter state coordinate to produce
        the LINEAR (G1) movements.

        :param points: Numpy matrix containing the values of the tuple [X,Y,Z,F,S] coordinates.
        :type points: numpy.ndarray
        :return: None
        """

        logger.debug('Start writing points to file...')
        x, y, z, f_gc, s_gc = points
        logger.debug('Fetch points.')

        # Transform points (rotations, z-compensation and flipping)
        x_gc, y_gc, z_gc = self.transform_points(x, y, z)
        logger.debug('Transform points (rotations, z-compensation and flipping).')

        if self.minimal_gcode:
            x_gc = remove_repeated_coordinates(x_gc)
            y_gc = remove_repeated_coordinates(y_gc)
            z_gc = remove_repeated_coordinates(z_gc)
            f_gc = remove_repeated_coordinates(f_gc)
            logger.debug('Reduce the G-Code commands.')

        # Convert points if G-Code commands
        args = [self._format_args(x, y, z, f) for (x, y, z, f) in zip(x_gc, y_gc, z_gc, f_gc)]
        for (arg, s) in itertools.zip_longest(args, s_gc):
            if s == 0 and self._shutter_on is True:
                self.instruction('\n')
                self.dwell(self.short_pause)
                self.shutter('OFF')
                self.dwell(self.long_pause)
                self.instruction('\n')
            elif s == 1 and self._shutter_on is False:
                self.instruction('\n')
                self.dwell(self.short_pause)
                self.shutter('ON')
                self.dwell(self.long_pause)
                self.instruction('\n')
            else:
                self._instructions.append(f'G1 {arg}\n')
        self.dwell(self.long_pause)
        self.instruction('\n')

    def close(self, filename: str | None = None) -> None:
        """Close and export a G-Code file.

        The functions writes all the instructions in a .pgm file. The filename is specified during the class
        instatiation. If no extension is present, the proper one is automatically added.

        Parameters
        ----------
        filename: str, optional
            Name of the .pgm file. The default value is ``self.filename``.

        Returns
        -------
        None
        """

        # get filename and add the proper file extension
        pgm_filename = pathlib.Path(self.filename) if filename is None else pathlib.Path(filename)
        pgm_filename = pgm_filename.with_suffix('.pgm')
        logger.debug(f'Export to {pgm_filename}.')

        # create export directory (mimicking the POSIX mkdir -p command)
        if self.export_dir:
            exp_dir = pathlib.Path(self.export_dir)
            if not exp_dir.is_dir():
                exp_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f'Created {exp_dir} directory.')
            pgm_filename = exp_dir / pgm_filename

        # write instructions to file
        with open(pgm_filename, 'w') as f:
            f.write(''.join(self._instructions))
        self._instructions.clear()
        if self.verbose:
            logger.info('G-code compilation completed.')
        logger.debug('G-code compilation completed.')

    # Geometrical transformations
    def transform_points(
        self,
        x: nparray,
        y: nparray,
        z: nparray,
    ) -> tuple[nparray, nparray, nparray]:
        """Transform points.

        The function takes in a set of points and apply a set of geometrical transformation (flip, translation,
        rotation and warp compensation).

        Parameters
        ----------
        x: numpy.ndarray
            Array of the `x`-coordinates.
        y: numpy.ndarray
            Array of the `y`-coordinates.
        z: numpy.ndarray
            Array of the `z`-coordinates.

        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
            Transformed `x`, `y` and `z` arrays.
        """

        # Normalize data
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
        logger.debug('Normalize x-, y-, z-arrays to numpy.ndarrys.')

        # Translate points to new origin
        x -= self.shift_origin[0]
        y -= self.shift_origin[1]
        logger.debug('Shift x-, y-arrays to new origin.')

        # Flip x, y coordinates
        x, y = self.flip(x, y)

        # Rotate points
        point_matrix = np.stack((x, y, z), axis=-1)
        x_t, y_t, z_t = np.matmul(point_matrix, self.t_matrix).T
        logger.debug('Applied 3D rotation matrix.')

        # Compensate for warp
        if self.warp_flag:
            logger.debug('Compensate for warp.')
            return self.compensate(x_t, y_t, z_t)
        return x_t, y_t, z_t

    def flip(
        self,
        xc: nparray,
        yc: nparray,
    ) -> tuple[nparray, nparray]:
        """Flip path.

        Flip the laser path along the `x` and `y` coordinates.

        Parameters
        ----------
        xc: numpy.ndarray
            Array of the `x`-coordinates.
        yc: numpy.ndarray
            Array of the `y`-coordinates.

        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
            Flipped `x` and `y` arrays.
        """

        # disp = np.array([self.shift_origin[0], self.shift_origin[1], 0])
        fx = int(self.flip_x) * 2 - 1
        fy = int(self.flip_y) * 2 - 1
        mirror_matrix = np.array([[-fx, 0], [0, -fy]])
        flip_x, flip_y = mirror_matrix @ np.array([xc, yc])

        return flip_x, flip_y

    def compensate(
        self,
        x: nparray,
        y: nparray,
        z: nparray,
    ) -> tuple[nparray, nparray, nparray]:
        """Warp compensation.

        Returns the `z`-compensated points for the glass warp using ``self.fwarp`` function.

        Parameters
        ----------
        x: numpy.ndarray
            Array of the `x`-coordinates.
        y: numpy.ndarray
            Array of the `y`-coordinates.
        z: numpy.ndarray
            Array of the `z`-coordinates.

        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
            Untouched `x`, `y` arrays and `z`-compensated array.
        """

        x_comp = copy.deepcopy(np.array(x))
        y_comp = copy.deepcopy(np.array(y))
        z_comp = copy.deepcopy(np.array(z))

        xy = np.column_stack([x_comp, y_comp])
        zwarp = np.array(self.fwarp(xy), dtype=np.float32)
        z_comp += zwarp

        return x_comp, y_comp, z_comp

    @property
    def t_matrix(self) -> nparray:
        """Composition of `xy` rotation matrix and `z` refractive index compensation.

        Given the rotation rotation_angle and the refractive index, the function compute the transformation matrix as
        composition of rotation matrix (RM) and a homothety matrix (SM).

        Returns
        -------
        numpy.ndarray
            Composition of `xy` rotation matrix and `z` compensation for the refractive change between air (or water)
            and glass interface.
        """

        rm = np.array(
            [
                [np.cos(self.rotation_angle), -np.sin(self.rotation_angle), 0],
                [np.sin(self.rotation_angle), np.cos(self.rotation_angle), 0],
                [0, 0, 1],
            ]
        )
        sm = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1 / self.neff],
            ]
        )
        tm = np.matmul(sm, rm).T
        return np.array(tm)

    def warp_management(self, opt: bool) -> interpolate.RBFInterpolator:
        """Warp Management

        Fetches warping function describing the surface of the sample.
        If ``opt`` is ``False``, the method load a dummy function representing a flat sample (no warp will be
        corrected).
        If ``opt`` is ``True``, the method look will load a function given by the interpolation of the points
        measured experimentally saved in a POS.txt file  containing a mapping of the surface of the sample. If a
        compensating function is already present in the current working direcoty, the method will just load that
        function without interpolating all the points from scratch.

        Notes
        -----
        Take care to input a POS.txt file.

        Parameters
        ----------
        opt: bool
            Flag to bypass the warp compensation

        Returns
        -------
        interpolate.RBFInterpolator
            interpolating function S(x, y) of the surface of the sample

        See Also
        --------
        femto.pgmcompiler.warp_generation: method that performs the surface interpolation given a POS.txt file
        """

        if not opt:

            def fwarp(_x: float, _y: float) -> float:
                """Dummy warp function."""
                return 0.0

        else:
            if not all(self.samplesize):
                raise ValueError(f'Wrong sample size dimensions. Given ({self.samplesize[0]}, {self.samplesize[1]}).')

            function_txt = self.CWD / 'POS.txt'
            function_pickle = self.CWD / 'fwarp.pkl'

            if function_pickle.is_file():
                with open(function_pickle, 'rb') as f_read:
                    fwarp = dill.load(f_read)
            else:
                # check for the existence of POS.txt in CWD. If not present, return dummy fwarp
                if not function_txt.is_file():
                    raise FileNotFoundError(
                        'Could not find surface mapping file. Add it to the current working directory'
                    )
                fwarp = self.warp_generation(surface_mapping_file=function_txt, gridsize=(100, 100), show=False)
                with open(function_pickle, 'wb') as f_write:
                    dill.dump(fwarp, f_write)
        return fwarp

    @staticmethod
    def warp_generation(
        surface_mapping_file: str | pathlib.Path = 'POS.txt',
        gridsize: tuple[int, int] = (100, 100),
        show: bool = False,
    ) -> interpolate.RBFInterpolator:
        """Warp Generation

        The method load the smapled points contained in the POS.txt file and finds a surface that interpolates them
        using the RBF interpolator.
        The ``show`` flag allows to plot the surface for debugging or inspection purposes.

        Parameters
        ----------
        surface_mapping_file: str | pathlib.Path
            File containing the warp coordinates of the sample.
        gridsize: tuple(int, int)
            Dimensions of the interpolation grid, (`x`-dim, `y`-dim). The default value is ``(100, 100)``.
        show: bool
            Flag to show the plot of the interpolated surface. The default value is ``False``.

        Returns
        -------
        scipy.interpolate.RBFInterpolator
            Warp function, `f(x, y)`

        See Also
        --------
        scipy.interpolate.RBFInterpolator: 2D interpolator function.

        """
        # Get data from POS.txt file
        warp_matrix = np.loadtxt(surface_mapping_file, dtype='f', delimiter=' ')

        x, y, z = warp_matrix.T
        f = interpolate.RBFInterpolator(np.column_stack([x, y]), z, kernel='cubic', smoothing=0)

        if show:
            # Plot the surface
            import matplotlib.pyplot as plt

            # Data generation for surface plotting
            x_f = np.linspace(np.min(x), np.max(x), int(gridsize[0]))
            y_f = np.linspace(np.min(y), np.max(y), int(gridsize[1]))

            # Interpolate
            X, Y = np.meshgrid(x_f, y_f)
            xy_points = np.stack([X.ravel(), Y.ravel()], -1)
            Z = f.__call__(xy_points).reshape(X.shape)

            ax = plt.axes(projection='3d')
            ax.contour3D(X, Y, Z, 200, cmap='viridis')
            ax.set_xlabel('X [mm]'), ax.set_ylabel('Y [mm]'), ax.set_zlabel('Z [mm]')
            ax.set_aspect('equalxz')
            plt.show()

        return f

    # Private interface
    def _format_args(
        self, x: float | None = None, y: float | None = None, z: float | None = None, f: float | None = None
    ) -> str:
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
            if f < 10 ** (-self.output_digits):
                logger.error(f'Try to move with F <= 0.0 mm/s. speed = {f}.')
                raise ValueError('Try to move with F <= 0.0 mm/s. Check speed parameter.')
            args.append(f'F{f:.{self.output_digits}f}')
        joined_args = ' '.join(args)
        return joined_args

    @staticmethod
    def _get_filepath(filename: str, filepath: str | None = None, extension: str | None = None) -> pathlib.Path:
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
            logger.error('Given filename is None. Give a valid filename.')
            raise ValueError('Given filename is None. Give a valid filename.')

        path = pathlib.Path(filename) if filepath is None else pathlib.Path(filepath) / filename
        if extension is None:
            logger.debug('Extension is None. Return Path without extension.')
            return path

        ext = '.' + extension.split('.')[-1].lower()
        if path.suffix != ext:
            logger.error(f'Given filename has wrong extension. Given {filename}, required {ext}.')
            raise ValueError(f'Given filename has wrong extension. Given {filename}, required {ext}.')
        logger.debug(f'Return path: {path}.')
        return path

    def _enter_axis_rotation(self, angle: float | None = None) -> None:

        if angle is None and self.aerotech_angle == 0.0:
            return
        angle = self.aerotech_angle if angle is None else float(angle % 360)

        self.comment('ACTIVATE AXIS ROTATION')
        self._instructions.append(f'G1 X{0.0:.6f} Y{0.0:.6f} Z{0.0:.6f} F{self.speed_pos:.6f}\n')
        self._instructions.append('G84 X Y\n')
        self.dwell(self.short_pause)
        self._instructions.append(f'G84 X Y F{angle}\n\n')
        self.dwell(self.short_pause)
        logger.debug(f'Activate axis rotation with angle = {angle}.')

    def _exit_axis_rotation(self) -> None:
        self.comment('DEACTIVATE AXIS ROTATION')
        self._instructions.append(f'G1 X{0.0:.6f} Y{0.0:.6f} Z{0.0:.6f} F{self.speed_pos:.6f}\n')
        self._instructions.append('G84 X Y\n')
        self.dwell(self.short_pause)
        logger.debug('Deactivate axis rotation.')


def sample_warp(pts_x: int, pts_y: int, margin: float, parameters: dict(str, Any)) -> None:
    """Generate sampling script

    The function compile a PGM file that automatically reads the G-Code parameters (namely, angle and sample size)
    and some user-input parameters for measuring the z-coordinate of the focused laser beam on the sample surface.

    The sampling points are part of a ``pts_x``x``pts_y`` grid. ``margin`` is the distance between the edges of the
    grid and the side of the sample.

    The G-Code script is intended to work for the bottom surface of the sample. The user must correct the
    ``z``-coordinate for each point of the grid. At the end of the script a `POS.txt` file is generated with all the
    coordinates of the points. It can be used to interpolate the surface of the sample.

    Parameters
    ----------
    pts_x: int
        Number of grid points along the `x`-direction
    pts_y: int
        Number of grid points along the `y`-direction
    margin: float
        Distance between the edge of the points-grid and the edges of the sample.
    parameters: dict(str, Any)
        Dictionary of the the parameters for a G-Code file.

    Returns
    -------
    None
    """

    parameters['filename'] = 'SAMPLE_WARP.pgm'
    size_x, size_y = parameters['samplesize']
    angle = parameters['aerotech_angle']
    warp_name = 'WARP.txt'

    G = PGMCompiler(**parameters)

    function_txt = G.CWD / 'POS.txt'
    if pathlib.Path.is_file(function_txt):
        pathlib.Path.unlink(function_txt)

    with open(pathlib.Path(__file__).parent / 'utils' / warp_name) as f:
        for line in f:
            if line.startswith('<HEADER>'):
                G.header()
            else:
                G.instruction(line.format_map(locals()))
    G.close()


def main() -> None:
    """The main function of the script."""
    from femto.waveguide import Waveguide
    from femto.curves import sin
    from femto.helpers import dotdict

    # Parameters
    param_wg = dotdict(scan=6, speed=20, radius=15, pitch=0.080, int_dist=0.007, lsafe=3, samplesize=(25, 3))
    param_gc = dotdict(
        filename='testPGM.pgm', samplesize=param_wg['samplesize'], rotation_angle=2.0, flip_x=True, minimal_gcode=True
    )

    # Build paths
    chip = [Waveguide(**param_wg) for _ in range(2)]
    for i, wg in enumerate(chip):
        wg.start([-2, -wg.pitch / 2 + i * wg.pitch, 0.035])
        wg.linear([wg.lsafe, 0, 0])
        wg.bend(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        wg.bend(dy=(-1) ** (i + 1) * wg.dy_bend, dz=0, fx=sin)
        wg.linear([wg.x_end, wg.lasty, wg.lastz], mode='ABS')
        wg.end()

    # Compilation
    with PGMCompiler(**param_gc) as G:
        G.set_home([0, 0, 0])
        with G.repeat(param_wg['scan']):
            for i, wg in enumerate(chip):
                G.comment(f'Modo: {i}')
                G.write(wg.points)
        G.move_to([None, 0, 0.1])
        G.set_home([0, 0, 0])

    # Test warp script
    sample_warp(ptsX=9, ptsY=9, margin=2, PARAM_GC=param_gc)


if __name__ == '__main__':
    main()
