import os
import pickle
from itertools import product
from math import ceil
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d


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
        self.CWD = os.path.dirname(os.path.abspath(__file__))
        self.samplesize = samplesize
        self.lab = lab
        self.warp_flag = warp_flag
        self.fwarp = self.antiwarp_management(self.warp_flag)
        self.tshutter = self.set_tshutter()
        self.long_pause = long_pause
        self.short_pause = short_pause
        self.output_digits = output_digits

        self.nglass = n_glass
        self.nenv = n_environment
        self.angle = angle

        self.neff = self.nglass / self.nenv
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


class WaveguideParameters:
    """
    Class containing the parameters for the waveguide fabrication.
    """

    def __init__(self,
                 scan: int,
                 speed: float,
                 depth: float = 0.035,
                 radius: float = 15,
                 speedpos: float = 40,
                 dwelltime: float = 0.5,
                 dsafe: float = 0.015,
                 margin: float = 1.0,
                 cmd_rate_max: float = 1200,
                 acc_max: float = 500):
        if not isinstance(scan, int):
            raise ValueError('Number of scan must be integer.')

        # input parameters:
        self.scan = scan
        self.speed = speed
        self.depth = depth
        self.radius = radius
        self.speedpos = speedpos
        self.dwelltime = dwelltime

        self.cmd_rate_max = cmd_rate_max
        self.acc_max = acc_max

        self.dsafe = dsafe
        self.margin = margin

        # Compute parameters:
        self.lvelo = 3 * (0.5 * self.speed ** 2 / self.acc_max)  # length needed to acquire the writing speed [mm]
        self.dl = self.speed / self.cmd_rate_max  # minimum separation between two points [mm]


class TrenchParameters:
    """
    Class containing the parameters for trench fabrication.
    """

    def __init__(self, *,
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
