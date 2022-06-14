import os
import pickle
import warnings
from itertools import product
from math import ceil
from math import radians
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry
from scipy.interpolate import interp2d

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from shapely.geometry import box


class WaveguideParameters:
    """
    Class containing the parameters for the waveguide fabrication.
    """

    def __init__(self,
                 scan: int,
                 speed: float,
                 depth: float = 0.035,
                 radius: float = 15,
                 pitch: float = 0.080,
                 pitch_fa: float = 0.127,
                 int_dist: float = None,
                 int_length: float = 0.0,
                 arm_length: float = 0.0,
                 speedpos: float = 40,
                 dwelltime: float = 0.5,
                 lsafe: float = 4.0,
                 ltrench: float = 1.0,
                 dz_bridge: float = 0.015,
                 margin: float = 1.0,
                 cmd_rate_max: float = 1200,
                 acc_max: float = 500,
                 samplesize: Tuple[float, float] = (None, None)):
        if not isinstance(scan, int):
            raise ValueError('Number of scan must be integer.')

        # input parameters:
        self.scan = scan
        self._speed = speed
        self.depth = depth
        self._radius = radius
        self._pitch = pitch
        self.pitch_fa = pitch_fa
        self._int_dist = int_dist
        self._int_length = int_length
        self._arm_length = arm_length
        self._dz_bridge = dz_bridge
        self.samplesize = samplesize

        self.lsafe = lsafe
        self.ltrench = ltrench
        self.margin = margin
        self.speedpos = speedpos
        self.dwelltime = dwelltime

        self.cmd_rate_max = cmd_rate_max
        self.acc_max = acc_max

        # Computed parameters
        self.lvelo = None
        self.dl = None

        # TODO: think of a @property for these quantities
        self.dy_bend = None
        self.dx_bend = None
        self.dx_acc = None
        self.dx_mzi = None

        self._compute_parameters()
        self._calc_bend()

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float):
        self._speed = value
        self._compute_parameters()

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float):
        self._radius = value
        self._calc_bend()
        # compute

    @property
    def int_dist(self) -> float:
        return self._int_dist

    @int_dist.setter
    def int_dist(self, value):
        self._int_dist = value
        self._calc_bend()

    @property
    def pitch(self) -> float:
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        self._pitch = value
        self._calc_bend()

    @property
    def int_length(self) -> float:
        return self._int_length

    @int_length.setter
    def int_length(self, value):
        self._int_length = value
        # compute cose

    @property
    def arm_length(self):
        return self._arm_length

    @arm_length.setter
    def arm_length(self, value):
        self._arm_length = value
        # compute cose

    @property
    def dz_bridge(self):
        return self._dz_bridge

    @dz_bridge.setter
    def dz_bridge(self, value):
        self._dz_bridge = value

    @staticmethod
    def get_sbend_parameter(dy: float, radius: float) -> tuple:
        """
        Computes the final angle, and x-displacement for a circular S-bend given the y-displacement dy and curvature
        radius.

        :param dy: Displacement along y-direction [mm].
        :type dy: float
        :param radius: Curvature radius of the S-bend [mm].
        :type radius: float
        :return: (final angle [radians], x-displacement [mm])
        :rtype: tuple
        """
        a = np.arccos(1 - (np.abs(dy / 2) / radius))
        dx = 2 * radius * np.sin(a)
        return a, dx

    @classmethod
    def sbend_length(cls, dy: float, radius: float) -> float:
        """
        Computes the x-displacement for a circular S-bend given the y-displacement dy and curvature radius.

        :param dy: Displacement along y-direction [mm].
        :type dy: float
        :param radius: Curvature radius of the S-bend [mm].
        :type radius: float
        :return: x-displacement [mm]
        :rtype: float
        """
        return cls.get_sbend_parameter(dy, radius)[1]

    @staticmethod
    def get_spline_parameter(init_pos: np.ndarray, dy: float, dz: float, radius: float = 20,
                             disp_x: float = 0) -> tuple:
        """
        Computes the delta displacements along x-, y- and z-direction and the total lenght of the curve.

        :param init_pos: Initial position of the curve.
        :type init_pos: np.ndarray
        :param dy: Displacement along y-direction [mm].
        :type dy: float
        :param dz: Displacement along z-direction [mm].
        :type dz: float
        :param radius: Curvature radius of the spline [mm]. The default is 20 mm.
        :type dz: radius
        :param disp_x: Displacement along x-direction [mm]. The default is 0 mm.
        :type disp_x: float
        :return: (deltax [mm], deltay [mm], deltaz [mm], curve length [mm]).
        :rtype: Tuple[float, float, float, float]
        """
        xl, yl, zl = init_pos
        final_pos = np.array([yl + dy, zl + dz])
        if disp_x != 0:
            final_pos = np.insert(final_pos, 0, xl + disp_x)
            pos_diff = np.subtract(final_pos, init_pos)
            l_curve = np.sqrt(np.sum(pos_diff ** 2))
        else:
            final_pos = np.insert(final_pos, 0, xl)
            pos_diff = np.subtract(final_pos, init_pos)
            ang = np.arccos(1 - np.sqrt(pos_diff[1] ** 2 + pos_diff[2] ** 2) / (2 * radius))
            pos_diff[0] = 2 * radius * np.sin(ang)
            l_curve = 2 * ang * radius
        return pos_diff[0], pos_diff[1], pos_diff[2], l_curve

    # Private interface
    def _calc_bend(self):
        if self._pitch is None:
            print(f'WARNING: Waveguide pitch is set to None.')
            self.dy_bend = None
        if self._int_dist is None:
            print(f'WARNING: Interaction distance is set to None.')
            self.dy_bend = None
        else:
            self.dy_bend = 0.5 * (self._pitch - self._int_dist)
            if self.radius is None:
                raise ValueError('Curvature radius is set to None.')
            _, self.dx_bend = self.get_sbend_parameter(self.dy_bend, self.radius)
            self.dx_acc = 2 * self.dx_bend + self._int_length
            self.dx_mzi = 4 * self.dx_bend + 2 * self._int_length + self._arm_length

    def _compute_parameters(self):
        self.lvelo = 3 * (0.5 * self.speed ** 2 / self.acc_max)  # length needed to acquire the writing speed [mm]
        self.dl = self.speed / self.cmd_rate_max  # minimum separation between two points


class TrenchParameters:
    """
    Class containing the parameters for trench fabrication.
    """

    def __init__(self,
                 x_center: float = None,
                 y_min: float = None,
                 y_max: float = None,
                 bridge: float = 0.026,
                 lenght: float = 1,
                 nboxz: int = 4,
                 z_off: float = 0.020,
                 h_box: float = 0.075,
                 base_folder: str = r'C:\Users\Capable\Desktop',
                 deltaz: float = 0.0015,
                 delta_floor: float = 0.001,
                 beam_waist: float = 0.004,
                 round_corner: float = 0.005,
                 speed: float = 4,
                 speedpos: float = 5):
        self.x_center = x_center
        self.y_min = y_min
        self.y_max = y_max
        self.bridge = bridge
        self.length = lenght
        self.nboxz = nboxz
        self.z_off = z_off
        self.h_box = h_box
        self.deltaz = deltaz
        self.delta_floor = delta_floor
        self.beam_waist = beam_waist
        self.round_corner = round_corner
        self.speed = speed
        self.speedpos = speedpos

        # adjust bridge size considering the size of the laser focus [mm]
        self.adj_bridge = self.bridge / 2 + self.beam_waist + self.round_corner
        self.n_repeat = int(ceil((self.h_box + self.z_off) / self.deltaz))

        # FARCALL directories
        self.base_folder = base_folder
        self.CWD = os.path.dirname(os.path.abspath(__file__))

        # self.rect = self._make_box()

    @property
    def rect(self) -> float:
        """
        Getter for the x-coordinate of the trench column center.

        :return: center x-coordinate of the trench block
        :rtype: float
        """
        return self._make_box()

    # Private interface
    def _make_box(self) -> shapely.geometry.box:
        """
        Create the rectangular box for the whole trench column. If the ``x_c``, ``y_min`` and ``y_max`` are set we
        create a rectangular polygon that will be used to create the single trench blocks.

        ::
            +-------+  -> y_max
            |       |
            |       |
            |       |
            +-------+  -> y_min
                x_c

        :return: Rectangular box centered in ``x_c`` and y-borders at ``y_min`` and ``y_max``.
        :rtype: shapely.geometry.box
        """
        if self.x_center is not None and self.y_min is not None and self.y_max is not None:
            return box(self.x_center - self.length / 2, self.y_min,
                       self.x_center + self.length / 2, self.y_max)
        else:
            return None


class GcodeParameters:
    """
    Class containing the parameters for the G-Code file compiler.
    """

    def __init__(self,
                 filename: str = None,
                 samplesize: Tuple[float, float] = (None, None),
                 lab: str = 'CAPABLE',
                 warp_flag: bool = False,
                 n_glass: float = 1.50,
                 n_environment: float = 1.33,
                 angle: float = 0.0,
                 long_pause: float = 0.5,
                 short_pause: float = 0.25,
                 output_digits: int = 6):

        self.filename = filename
        if self.filename is None:
            raise ValueError('Filename is None, set GcodeParameters.filename.')

        self.CWD = os.path.dirname(os.path.abspath(__file__))
        self.samplesize = samplesize
        self.lab = lab
        self.warp_flag = warp_flag
        self.fwarp = self.antiwarp_management(self.warp_flag)
        self.tshutter = self.set_tshutter()
        self.long_pause = long_pause
        self.short_pause = short_pause
        self.output_digits = output_digits

        self._n_glass = n_glass
        self._n_env = n_environment
        self.neff = self._n_glass / self._n_env

        if angle != 0:
            print(' BEWARE ANGLES MUST BE IN DEGREE!! '.center(39, "*"))
            print(f' Given alpha = {angle % 360:.3f} deg. '.center(39, "*"))
        self.angle = radians(angle % 360)

        self.xsample, self.ysample = self.samplesize

    def set_tshutter(self) -> float:
        """
        Function that set the shuttering time given the fabrication laboratory.

        :return: shuttering time
        :rtype: float
        """
        if self.lab.upper() not in ['CAPABLE', 'DIAMOND', 'FIRE']:
            raise ValueError('Lab can be only CAPABLE, DIAMOND or FIRE',
                             f'Given {self.lab}.')
        if self.lab.upper() == 'CAPABLE':
            return 0.000
        else:
            return 0.005

    def antiwarp_management(self, opt: bool, num: int = 16) -> interp2d:
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

        if opt:
            if any(x is None for x in self.samplesize):
                raise ValueError('Wrong sample size dimensions.',
                                 f'Given ({self.samplesize[0]}, {self.samplesize[1]}).')
            function_pickle = os.path.join(self.CWD, "fwarp.pkl")
            if os.path.exists(function_pickle):
                fwarp = pickle.load(open(function_pickle, "rb"))
            else:
                fwarp = self.antiwarp_generation(self.samplesize, num)
                pickle.dump(fwarp, open(function_pickle, "wb"))
        else:
            def fwarp(x, y):
                return 0
        return fwarp

    @staticmethod
    def antiwarp_generation(samplesize: Tuple[float, float], num_tot: int, *, margin: float = 2) -> interp2d:
        """
        Helper for the generation of antiwarp function.
        The minimum number of data points required is (k+1)**2, with k=1 for linear, k=3 for cubic and k=5 for quintic
        interpolation.

        :param samplesize: glass substrate dimensions, (x-dim, y-dim)
        :type samplesize: Tuple(float, float)
        :param num_tot: number of sampling points
        :type num_tot: int
        :param margin: margin [mm] from the borders of the glass samples
        :type margin: float
        :return: warp function, `f(x, y)`
        :rtype: scipy.interpolate.interp2d
        """

        if num_tot < 4 ** 2:
            raise ValueError('I need more values to compute the interpolation.')

        num_side = int(np.ceil(np.sqrt(num_tot)))
        xpos = np.linspace(margin, samplesize[0] - margin, num_side)
        ypos = np.linspace(margin, samplesize[1] - margin, num_side)
        xlist = []
        ylist = []
        zlist = []

        print('Focus height in µm (!!!) at:')
        for pos in list(product(xpos, ypos)):
            xlist.append(pos[0])
            ylist.append(pos[1])
            zlist.append(float(input('X={:.1f} Y={:.1f}: \t'.format(pos[0],
                                                                    pos[1]))) / 1000)
            if zlist[-1] == '':
                raise ValueError('You have missed the last value.')

        # surface interpolation
        func_antiwarp = interp2d(xlist, ylist, zlist, kind='cubic')

        # plot the surface
        xprobe = np.linspace(-3, samplesize[0] + 3)
        yprobe = np.linspace(-3, samplesize[1] + 3)
        zprobe = func_antiwarp(xprobe, yprobe)
        ax = plt.axes(projection='3d')
        ax.contour3D(xprobe, yprobe, zprobe, 200, cmap='viridis')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        plt.show()

        return func_antiwarp
