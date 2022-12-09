from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from femto.helpers import sign
from femto.laserpath import LaserPath


@dataclass(repr=False)
class Marker(LaserPath):
    """
    Class representing an ablation marker.
    """

    depth: float = 0.0
    lx: float = 1.0
    ly: float = 0.060

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.z_init is None:
            self.z_init = self.depth

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}'

    def cross(self, position: list[float], lx: float | None = None, ly: float | None = None) -> None:
        """
        The function takes a position (x,y,z) and two lengths (lx,ly) and draws a cross at the position with the given
        lengths

        :param position: The position of the cross
        :type position: list[float]
        :param lx: The length of the cross in the x-direction
        :type lx: float | None
        :param ly: The length of the cross in the y-direction
        :type ly: float | None

        Computes the points of a cross marker of given widht along x- and y-direction.

        :param position: 2D ordered coordinate list that specifies the cross position [mm].
            position[0] -> X
            position[1] -> Y
        :type position: List[float]
        :param lx: Length of the cross marker along x [mm]. The default is self.lx.
        :type lx: float
        :param ly: Length of the cross marker along y [mm]. The default is self.ly.
        :type ly: float
        :return: None
        """

        if len(position) == 2:
            position.append(self.depth)

        if len(position) != 3:
            raise ValueError(
                'The cross position is not valid. The valid number of elements are 2 or 3. The given '
                f'position has {len(position)} elements.'
            )

        if lx is None and self.lx is None:
            raise ValueError(
                "The x-length of the cross is None. Set the Marker's 'lx' attribute or give a valid number as " 'input.'
            )
        lx = self.lx if lx is None else lx
        if ly is None and self.ly is None:
            raise ValueError(
                "The y-length of the cross is None. Set the Marker's 'ly' attribute or give a valid number as " 'input.'
            )
        ly = self.ly if ly is None else ly

        # start_pos = np.add(position, [-lx / 2, 0, 0])
        xi, yi, zi = position
        self.start([xi - lx / 2, yi, zi])
        self.linear([lx, None, None], mode='INC')
        self.linear([None, None, None], mode='INC', shutter=0)
        self.linear([-lx / 2, -ly / 2, None], mode='INC', shutter=0)
        self.linear([None, None, None], mode='INC')
        self.linear([None, ly, None], mode='INC')
        self.linear([None, None, None], mode='INC', shutter=0)
        self.linear(position, mode='ABS', shutter=0)

    def ruler(
        self,
        y_ticks: list[float] | npt.NDArray[np.float32],
        lx: float | None = None,
        lx2: float | None = None,
        x_init: float | None = None,
    ) -> None:
        """
        > This function draws a ruler on the sample

        Computes the points of a ruler marker starting from a given x-coordinate. The y-coordinates of the ticks are
        specified by the user.

        :param y_ticks: list[float] | npt.NDArray[np.float32]
        :type y_ticks: list[float] | npt.NDArray[np.float32]
        :param lx: The length of the ruler's tick marks
        :type lx: float | None
        :param lx2: The length of the tick marks
        :type lx2: float | None
        :param x_init: The x-coordinate of the ruler's origin
        :type x_init: float | None
        :return: None
        """
        if y_ticks is None or len(y_ticks) == 0:
            return None

        # Sort the y-ticks and takes unique points
        y_ticks = np.unique(y_ticks)

        # Ticks x-length
        if lx is None and self.lx is None:
            raise ValueError(
                "The x-length of the ruler is None. Set the Marker's 'lx' attribute or give a valid number as " 'input.'
            )
        lx = self.lx if lx is None else lx
        if lx2 is None:
            lx2 = 0.75 * lx
        x_ticks = np.repeat(lx2, len(y_ticks))
        x_ticks[0] = lx

        # Set x_init
        if x_init is None and self.x_init is None:
            raise ValueError(
                "The x-init value of the ruler is None. Set the Marker's 'x_init' attribute or give a valid number "
                'as input.'
            )
        x_init = self.x_init if x_init is None else x_init

        # Add straight segments
        for xt, yt in zip(x_ticks, y_ticks):
            self.linear([x_init, yt, self.depth], mode='ABS', shutter=0)
            self.linear([None, None, None], mode='ABS')
            self.linear([xt, yt, None], mode='ABS')
            self.linear([None, None, None], mode='ABS', shutter=0)
        self.end()

    def meander(
        self,
        init_pos: list[float],
        final_pos: list[float],
        width: float = 1,
        delta: float = 0.001,
        orientation: str = 'x',
    ) -> None:
        """
        > The function takes in the initial and final positions, the width of the meander, the distance between the
        parallel lines, and the orientation of the meander (either parallel to the x or y axis)

        :param init_pos: Initial position of the meander
        :type init_pos: list[float]
        :param final_pos: The final position of the meander
        :type final_pos: list[float]
        :param width: The width of the meander, defaults to 1
        :type width: float (optional)
        :param delta: The distance between the parallel lines
        :type delta: float
        :param orientation: 'x' or 'y', defaults to x
        :type orientation: str (optional)
        """
        if len(init_pos) not in [2, 3]:
            raise ValueError(
                'Initial position must be a list of 2 (x,y) or 3 (x,y,z) values.'
                f"init_pos has {len(init_pos)} elements. Give a valid 'init_pos' list."
            )
        xi, yi, *_ = init_pos

        if len(final_pos) not in [2, 3]:
            raise ValueError(
                'Final position must be a list of 2 (x,y) or 3 (x,y,z) values.'
                f"final_pos has {len(final_pos)} elements. Give a valid 'final_pos' list."
            )
        xf, yf, *_ = final_pos

        if orientation.lower() not in ['x', 'y']:
            raise ValueError(
                'Orientation must be either "x" (parallel lines along x) or "y" (parallel lines along y).'
                f'Given {orientation}.'
            )

        s = sign()
        if orientation.lower() == 'x':
            num_passes = math.floor(np.abs(yf - yi) / delta)
            delta = np.sign(yf - yi) * delta

            self.start(init_pos)
            for _ in range(num_passes):
                self.linear([next(s) * width, 0, 0], mode='INC')
                self.linear([0, delta, 0], mode='INC')
            self.linear([next(s) * width, 0, 0], mode='INC')
            self.end()

        else:
            num_passes = math.floor(np.abs(xf - xi) / delta)
            delta = np.sign(xf - xi) * delta

            self.start(init_pos)
            for _ in range(num_passes):
                self.linear([0, next(s) * width, 0], mode='INC')
                self.linear([delta, 0, 0], mode='INC')
            self.linear([0, next(s) * width, 0], mode='INC')
            self.end()

    def ablation(
        self,
        points: list[list[float]],
        shift: float | None = None,
    ) -> None:

        """
        It takes a list of points and adds a linear path to the gcode file for each point in the list

        :param points: list[list[float]]
        :type points: list[list[float]]
        :param shift: The amount to shift the path by
        :type shift: float | None
        :return: A list of lists of floats.
        """
        if not points:
            return

        # shift the path's points by shift value
        pts = np.asarray(points)
        if shift is not None:
            path_list = [
                np.add(pts, [0, 0, 0]),
                np.add(pts, [shift, 0, 0]),
                np.add(pts, [-shift, 0, 0]),
                np.add(pts, [0, shift, 0]),
                np.add(pts, [0, -shift, 0]),
            ]
        else:
            path_list = [pts]

        # Add linear segments
        for path in path_list:
            self.linear(path[0], mode='ABS', shutter=0)
            for p in path:
                self.linear(p, mode='ABS')
            self.linear([None, None, None], mode='ABS', shutter=0)
        self.end()


def main():
    import matplotlib.pyplot as plt

    from femto.helpers import dotdict, split_mask

    PARAMETERS_MK = dotdict(scan=1, speed=2, speed_pos=5, speed_closed=5, depth=0.000, lx=1, ly=1)
    PARAMETERS_GC = dotdict(filename='testPGM.pgm', laser='PHAROS', samplesize=(10, 10))

    c = Marker(**PARAMETERS_MK)
    # c.cross([2.5, 1], 5, 2)
    c.ablation([[0, 0, 0], [5, 0, 0]], shift=0.1)
    print(c.points)

    from femto.pgmcompiler import PGMCompiler

    with PGMCompiler(**PARAMETERS_GC) as gc:
        gc.write(c.points)

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlabel('X [mm]'), ax.set_ylabel('Y [mm]')
    x, y, *_, s = c.points
    for x_seg, y_seg in zip(split_mask(x, s.astype(bool)), split_mask(y, s.astype(bool))):
        ax.plot(x_seg, y_seg, '-k', linewidth=2.5)
    for x_seg, y_seg in zip(split_mask(x, ~s.astype(bool)), split_mask(y, ~s.astype(bool))):
        ax.plot(x_seg, y_seg, ':b', linewidth=0.5)
    plt.show()


if __name__ == '__main__':
    main()
