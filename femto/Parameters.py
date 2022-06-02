import os
from math import ceil
from typing import Tuple


class GcodeParameters:
    def __init__(self,
                 lab: str,
                 samplesize: Tuple[float, float],
                 fwarp_flag: bool,
                 n_glass: float,
                 n_environment: float,
                 angle: float):
        self.lab = lab
        self.samplesize = samplesize
        self.fwarp_flag = fwarp_flag
        self.tshutter = self.set_shutter()

        self.nglass = n_glass
        self.nenv = n_environment
        self.angle = angle

        self.neff = self.nglass / self.nenv
        self.xsample, self.ysample = self.samplesize

    def set_shutter(self) -> float:
        if self.lab.upper() not in ['CAPABLE', 'DIAMOND', 'FIRE']:
            raise ValueError('Lab can be only CAPABLE, DIAMOND or FIRE',
                             f'Given {self.lab}.')
        if self.lab.upper() == 'CAPABLE':
            print('Pay attention to set the ELECTRO-optic shutter (III) on AUTO')
            return 0.000
        else:
            print('Pay attention to set the MECHANIC shutter (I) on AUTO')
            return 0.005


class WaveguideParameters:

    def __init__(self,
                 scan: int,
                 speed: float,
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
        self.speed = speed
        self.scan = scan
        self.radius = radius
        self.speedpos = speedpos
        self.dwelltime = dwelltime

        self.cmd_rate_max = cmd_rate_max
        self.acc_max = acc_max

        self.dsafe = dsafe
        self.margin = margin

        # # Computed parameters:
        # length needed to acquire the writing speed [mm]
        self.lvelo = 3 * (0.5 * self.speed ** 2 / self.acc_max)

        # minimum separation between two points [mm]
        self.dl = self.speed / self.cmd_rate_max

        # distance between points for the warp compensation [mm]
        self.lwarp = 10 * self.dl


class TrenchParameters:

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

        self.adj_bridge = self.bridge / 2 + self.beam_waist + self.round_corner
        self.n_repeat = int(ceil((self.h_box + self.z_off) / self.deltaz))

        # FARCALL directories
        self.base_folder = base_folder
        self.CWD = os.path.dirname(os.path.abspath(__file__))
