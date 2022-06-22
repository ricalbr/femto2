import warnings
from typing import List

import numpy as np

from femto import Waveguide


class Marker(Waveguide):
    """
    Class representing an ablation marker.
    """

    def __init__(self, param: dict):
        super().__init__(param)

    def cross(self, position: List[float], lx: float = 1, ly: float = 0.05, speedpos: float = None):
        """
        Computes the points of a cross marker of given widht along x- and y-direction.

        :param position: 2D ordered coordinate list that specifies the cross position [mm].
            position[0] -> X
            position[1] -> Y
        :type position: List[float]
        :param lx: Length of the cross marker along x [mm]. The default is 1 mm.
        :type lx: float
        :param ly: Length of the cross marker along y [mm]. The default is 0.05 mm.
        :type ly: float
        :param speedpos: Transition speed with the shutter closes [mm/s]. The default is self.speed.
        :type speedpos: float
        :return: None
        """
        if len(position) == 2:
            position.append(self.depth)
        elif len(position) == 3:
            position[2] = self.depth
            warnings.warn(f'Given 3D coordinate list. Z-coordinate is overwritten to {self.depth} mm.')
        else:
            raise ValueError('Given invalid position.')

        if speedpos is None:
            speedpos = self.speed_closed

        start_pos = np.add(position, [-lx / 2, 0, 0])
        self.start(start_pos, speedpos=5.0) \
            .linear([lx, 0, 0], speed=self.speed) \
            .linear([-lx / 2, -ly / 2, 0], speed=speedpos, shutter=0) \
            .linear([0, 0, 0], speed=speedpos, shutter=1) \
            .linear([0, ly, 0], speed=self.speed)
        self.end()

    def ruler(self, y_ticks: List, lx: float, lx_short: float = None, x_init: float = -2, speedpos: float = None):
        """
        Computes the points of a ruler marker. The y-coordinates of the ticks are specified by the user as well as
        the length of the ticks in the x-direction.

        :param y_ticks: y-coordinates of the ruler's ticks [mm]
        :type y_ticks: List[float]
        :param lx: Long tick length along x [mm]. The default is 1 mm.
        :type lx: float
        :param lx_short: Short tick length along x [mm]. The default is 0.75 mm.
        :type lx_short: float
        :param x_init: Starting x-coordinate of the laser [mm]. The default is -2 mm.
        :type x_init: float
        :param speedpos: Transition speed with the shutter closes [mm/s]. The default is self.speed.
        :type speedpos: float
        :return: None
        """

        if speedpos is None:
            speedpos = self.speed_closed

        if lx_short is None:
            lx_short = 0.75 * lx
        tick_len = lx_short * np.ones_like(y_ticks)
        tick_len[0] = lx

        self.start([x_init, y_ticks[0], self.depth])
        for y, tlen in zip(y_ticks, tick_len):
            self.linear([x_init, y, self.depth], speed=speedpos, mode='ABS', shutter=0)
            self.linear([0, 0, 0], shutter=1)
            self.linear([tlen, 0, 0], speed=self.speed, shutter=1)
            self.linear([0, 0, 0], speed=self.speed, shutter=0)
        self.end(speedpos)


def _example():
    from femto import PGMCompiler
    from femto.helpers import dotdict

    PARAMETERS_MK = dotdict(
            scan=1,
            speed=4,
            speedpos=5,
            depth=0.001
    )

    PARAMETERS_GC = dotdict(
            filename='testMarker.pgm',
            lab='CAPABLE',
            samplesize=(25, 25),
            angle=0.0,
    )

    c = Marker(PARAMETERS_MK)
    c.ruler([0, 1, 2], 5, 3.5)
    print(c.points)

    with PGMCompiler(PARAMETERS_GC) as gc:
        gc.write(c.points)


if __name__ == '__main__':
    _example()
