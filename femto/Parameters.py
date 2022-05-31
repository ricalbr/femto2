import os
from typing import List
from math import ceil


class GcodeParameters:
    def __init__(self,
                 lab: str,
                 samplesize: List[float],
                 shutter: str,
                 fwarp,
                 n_glass: float,
                 n_environment: float,
                 angle: float):
        self.lab = lab
        self.samplesize = samplesize
        self.fwarp = fwarp
        self.shutter = shutter
        self.tshutter = None
        self.set_shutter()

        self.max_cmd_rate = 1200
        self.accmax = 500
        self.nglass = n_glass
        self.nenv = n_environment
        self.angle = angle

        self.neff = self.nglass / self.nenv
        self.xsample, self.ysample = self.samplesize

    def set_shutter(self):
        if self.shutter.upper() not in ['MECHANIC', 'ELECTRO']:
            raise ValueError('Shutter can be only MECHANIC or ELECTRO',
                             f'Given {self.shutter}.')
        if self.shutter.upper() == 'MECHANIC':
            print('Pay attention to set the MECHANIC shutter (I) on AUTO')
            self.tshutter = 0.005
        else:
            if self.lab.upper() in ['DIAMOND', 'FIRE']:
                raise ValueError('You cannot use the ELECTRO-optic shutter in DIAMOND\n')
            print('Pay attention to set the ELECTRO-optic shutter (III) on AUTO')
            self.tshutter = 0.000


class WaveguideParameters(GcodeParameters):

    def __init__(self, *,
                 lab: str,
                 samplesize: List[float],
                 shutter: str,
                 fwarp,
                 n_glass: float,
                 n_environment: float,
                 angle: float,
                 speed: float,
                 scan: int,
                 radius: float,
                 speedpos: float,
                 dwelltime: float,
                 dsafe: float,
                 margin: float = 1.0):
        super().__init__(lab, samplesize, shutter, fwarp, n_glass, n_environment, angle)

        if not isinstance(scan, int):
            raise ValueError('Number of scan must be integer.')

        # input parameters:
        self.speed = speed
        self.scan = scan
        self.radius = radius
        self.speedpos = speedpos
        self.dwelltime = dwelltime

        self.dsafe = dsafe
        self.margin = margin

        # Computed parameters:
        # length needed to acquire the writing speed [mm]
        self.lvelo = 3 * (0.5 * self.speed ** 2 / self.accmax)

        # minimum separation between two points [mm]
        self.dl = self.speed / self.max_cmd_rate

        # distance between points for the warp compensation [mm]
        self.lwarp = 10 * self.dl

        # equivalent length of the shutter opening/closing delay
        self.lshutter = self.speed * self.tshutter


class TrenchParameters(GcodeParameters):

    def __init__(self, *,
                 lab: str,
                 samplesize: List[float],
                 shutter: str,
                 fwarp,
                 n_glass: float,
                 n_environment: float,
                 angle: float,
                 lenght: float,
                 speed: float,
                 speedpos: float,
                 nboxz: int,
                 deltaz: float,
                 z_off: float,
                 h_box: float,
                 base_folder: str):
        super().__init__(lab, samplesize, shutter, fwarp, n_glass, n_environment, angle)

        self.length = lenght
        self.speed = speed
        self.speedpos = speedpos
        self.nboxz = nboxz
        self.deltaz = deltaz
        self.z_off = z_off
        self.h_box = h_box

        self.n_repeat = int(ceil((self.h_box + self.z_off) / self.deltaz))

        # FARCALL directories
        self.base_folder = base_folder
        self.CWD = os.path.dirname(os.path.abspath(__file__))
