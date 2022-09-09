import warnings
from typing import List

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
import numpy as np

from femto import Waveguide


class Marker(Waveguide):
    """
    Class representing an ablation marker.
    """

    def __init__(self, param: dict):
        super().__init__(param)

    def start(self, init_pos: List[float], speedpos: float = None) -> Self:
        """
        Starts a waveguide in the initial position given as input.
        The coordinates of the initial position are the first added to the matrix that describes the waveguide.

        :param init_pos: Ordered list of coordinate that specifies the waveguide starting point [mm].
            init_pos[0] -> X
            init_pos[1] -> Y
            init_pos[2] -> Z
        :type init_pos: List[float]
        :param speedpos: Translation speed [mm/s].
        :type speedpos: float
        :return: Self
        :rtype: Waveguide
        """
        if len(init_pos) == 2:
            init_pos.append(self.depth)
        elif len(init_pos) == 3:
            init_pos[2] = self.depth
        else:
            raise ValueError('Given invalid position.')
        if self._x.size != 0:
            raise ValueError('Coordinate matrix is not empty. Cannot start a new waveguide in this point.')
        if speedpos is None:
            speedpos = self.speedpos

        x0, y0, z0 = init_pos
        f0 = np.asarray(speedpos, dtype=np.float32)
        s0 = np.asarray(0.0, dtype=np.float32)
        s1 = np.asarray(1.0, dtype=np.float32)
        self.add_path(x0, y0, z0, f0, s0)
        self.add_path(x0, y0, z0, f0, s1)
        return self

    def end(self):
        """
        Ends a waveguide. The function automatically return to the initial point of the waveguide with a translation
        speed specified by the user.

        :return: Self
        :rtype: Waveguide
        """
        x = np.array([self._x[-1], self._x[-1]]).astype(np.float32)
        y = np.array([self._y[-1], self._y[-1]]).astype(np.float32)
        z = np.array([self._z[-1], self._z[-1]]).astype(np.float32)
        f = np.array([self._f[-1], self.speed_closed]).astype(np.float32)
        s = np.array([0, 0]).astype(np.float32)
        self.add_path(x, y, z, f, s)
        self.fabrication_time()

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
        self.start(start_pos, speedpos=5.0)
        self.linear([lx, 0, 0], speed=self.speed)
        self.linear([-lx / 2, -ly / 2, 0], speed=speedpos, shutter=0)
        self.linear([0, 0, 0], speed=speedpos, shutter=1)
        self.linear([0, ly, 0], speed=self.speed)
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
        self.end()

    def meander(self, init_pos: List[float], final_pos: List[float], width: float = 1, delta: float = 0.001,
                orientation: str = 'x', speedpos: float = None):

        if speedpos is None:
            speedpos = self.speed_closed

        if orientation.lower() not in ['x', 'y']:
            raise ValueError('Orientation must be either "x" (parallel lines along x) or "y" (parallel lines along y).'
                             f'Given {orientation}.')
        elif orientation.lower() == 'x':
            num_passes = int(np.abs(init_pos[1] - final_pos[1]) / delta)
            delta = np.sign(final_pos[1] - init_pos[1]) * delta

            self.start(init_pos, speedpos=5.0)
            for i, _ in enumerate(range(num_passes)):
                self.linear([(-1) ** i * width, 0, 0], mode='INC')
                self.linear([0, delta, 0], mode='INC')
            self.linear([(-1) ** (i + 1) * width, 0, 0], mode='INC')
            self.end()

        else:
            num_passes = int(np.abs(init_pos[0] - final_pos[0]) / delta)
            delta = np.sign(final_pos[0] - init_pos[0]) * delta

            self.start(init_pos, speedpos=5.0)
            for i, _ in enumerate(range(num_passes)):
                self.linear([self.lastx, (-1) ** i * width, self.lastz], mode='ABS')
                self.linear([delta, 0, 0], mode='INC')
            self.linear([(-1) ** i * width, self.lasty, self.lastz], mode='ABS')
            self.end()

    def ablation(self, points: List[List[float]] = None, shift: float = None, speedpos: float = None):
        if points is None: return

        if speedpos is None:
            speedpos = self.speed_closed

        self.start(points.pop(0))
        for p in points:
            if p[0] == None: p[0] = self.lastx
            if p[1] == None: p[1] = self.lasty
            if p[2] == None: p[2] = self.lastz
            self.linear(p, mode='ABS')

        if shift is not None:
            points = np.asarray(points)
            shifted_points = []
            shifted_points.append(np.add(points, [shift, 0, 0]))
            shifted_points.append(np.add(points, [-shift, 0, 0]))
            shifted_points.append(np.add(points, [0, shift, 0]))
            shifted_points.append(np.add(points, [0, -shift, 0]))

            for shift in shifted_points:
                self.linear(shift[0], mode='ABS', shutter=0)
                for p in shift:
                    self.linear(p, mode='ABS')
                self.linear(shift[-1], mode='ABS', shutter=0)
        self.end()


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
