from dataclasses import dataclass
from typing import List, Optional, TypeVar

import numpy as np

from femto.helpers import sign
from femto.LaserPath import LaserPath

MK = TypeVar("MK", bound="Marker")


@dataclass
class Marker(LaserPath):
    """
    Class representing an ablation marker.
    """

    depth: float = 0.0
    lx: float = 1.0
    ly: float = 0.060

    def __post_init__(self: MK):
        super().__post_init__()
        if self.z_init is None:
            self.z_init = self.depth

    def __repr__(self: MK) -> str:
        return f"{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}"

    def cross(self: MK, position: List[float], lx: Optional[float] = None, ly: Optional[float] = None) -> None:
        """
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

        # start_pos = np.add(position, [-lx / 2, 0, 0])
        xi, yi, zi = position
        self.start([xi - lx / 2, yi, zi], speed_pos=5.0)
        self.linear([lx, None, None], mode="INC")
        self.linear([None, None, None], mode="INC", shutter=0)
        self.linear([-lx / 2, -ly / 2, None], mode="INC", shutter=0)
        self.linear([None, None, None], mode="INC")
        self.linear([None, ly, None], mode="INC")
        self.linear(position, mode="ABS", shutter=0)
        self.end()

    def ruler(
        self: MK,
        y_ticks: List[float],
        lx: float,
        lx_short: float = None,
        x_init: Optional[float] = None,
    ) -> None:
        """
        Computes the points of a ruler marker. The y-coordinates of the ticks are specified by the user as well as
        the length of the ticks in the x-direction.

        :param y_ticks: y-coordinates of the ruler's ticks [mm]
        :type y_ticks: List[float]
        :param lx: Long tick length along x [mm]. The default is 1 mm.
        :type lx: float
        :param lx_short: Short tick length along x [mm]. The default is 0.75 mm.
        :type lx_short: float
        :param x_init: Starting x-coordinate of the laser [mm]. The default is self.x_init.
        :type x_init: float
        :return: None
        """

        if lx_short is None:
            lx_short = 0.75 * lx
        tick_len = lx_short * np.ones_like(y_ticks)
        tick_len[0] = lx

        self.start([x_init, y_ticks[0], self.depth])
        for y, tlen in zip(y_ticks, tick_len):
            self.linear([x_init, y, self.depth], mode="ABS", shutter=0)
            self.linear([0, 0, 0], shutter=1)
            self.linear([tlen, 0, 0], speed=self.speed, shutter=1)
            self.linear([0, 0, 0], speed=self.speed, shutter=0)
        self.end()

    def meander(
        self: MK,
        init_pos: List[float],
        final_pos: List[float],
        width: float = 1,
        delta: float = 0.001,
        orientation: str = "x",
    ) -> None:

        if orientation.lower() not in ["x", "y"]:
            raise ValueError(
                'Orientation must be either "x" (parallel lines along x) or "y" (parallel lines along y).'
                f"Given {orientation}."
            )
        s = sign()
        if orientation.lower() == "x":
            num_passes = int(np.abs(init_pos[1] - final_pos[1]) / delta)
            delta = np.sign(final_pos[1] - init_pos[1]) * delta

            self.start(init_pos)
            for _ in range(num_passes):
                self.linear([next(s) * width, 0, 0], mode="INC")
                self.linear([0, delta, 0], mode="INC")
            self.linear([next(s) * width, 0, 0], mode="INC")
            self.end()

        else:
            num_passes = int(np.abs(init_pos[0] - final_pos[0]) / delta)
            delta = np.sign(final_pos[0] - init_pos[0]) * delta

            self.start(init_pos)
            for _ in range(num_passes):
                self.linear([0, next(s) * width, 0], mode="INC")
                self.linear([delta, 0, 0], mode="INC")
            self.linear([0, next(s) * width, 0], mode="INC")
            self.end()

    def ablation(
        self: MK,
        points: List[List[float]] = None,
        shift: float = None,
        speedpos: float = None,
    ) -> None:
        if points is None:
            return

        if speedpos is None:
            speedpos = self.speed_closed

        self.start(points.pop(0), speed_pos=speedpos)
        for p in points:
            if p[0] is None:
                p[0] = self.lastx
            if p[1] is None:
                p[1] = self.lasty
            if p[2] is None:
                p[2] = self.lastz
            self.linear(p, mode="ABS")

        if shift is not None:
            points = np.asarray(points)
            shifted_points = [
                np.add(points, [shift, 0, 0]),
                np.add(points, [-shift, 0, 0]),
                np.add(points, [0, shift, 0]),
                np.add(points, [0, -shift, 0]),
            ]

            for shift in shifted_points:
                self.linear(shift[0], mode="ABS", shutter=0)
                for p in shift:
                    self.linear(p, mode="ABS")
                self.linear(shift[-1], mode="ABS", shutter=0)
        self.end()


def main():
    import matplotlib.pyplot as plt
    from femto.helpers import Dotdict, split_mask

    PARAMETERS_MK = Dotdict(scan=1, speed=1, speed_pos=5, speed_closed=5, depth=0.000, lx=1, ly=1)

    c = Marker(**PARAMETERS_MK)
    c.cross([2.5, 1], 5, 2)
    # print(c.points)

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlabel("X [mm]"), ax.set_ylabel("Y [mm]")
    x, y, *_, s = c.points
    for x_seg, y_seg in zip(split_mask(x, s.astype(bool)), split_mask(y, s.astype(bool))):
        ax.plot(x_seg, y_seg, "-k", linewidth=2.5)
    for x_seg, y_seg in zip(split_mask(x, ~s.astype(bool)), split_mask(y, ~s.astype(bool))):
        ax.plot(x_seg, y_seg, ":b", linewidth=0.5)
    plt.show()


if __name__ == "__main__":
    main()
