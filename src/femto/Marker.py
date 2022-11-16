from dataclasses import dataclass
from typing import List, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt

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

        if len(position) != 3:
            raise ValueError(
                "The cross position is not valid. The valid number of elements are 2 or 3. The given "
                f"position has {len(position)} elements."
            )

        if lx is None and self.lx is None:
            raise ValueError(
                "The x-length of the cross is None. Set the Marker's 'lx' attribute or give a valid number as input."
            )
        lx = self.lx if lx is None else lx
        if ly is None and self.ly is None:
            raise ValueError(
                "The y-length of the cross is None. Set the Marker's 'ly' attribute or give a valid number as input."
            )
        ly = self.ly if ly is None else ly

        # start_pos = np.add(position, [-lx / 2, 0, 0])
        xi, yi, zi = position
        self.start([xi - lx / 2, yi, zi])
        self.linear([lx, None, None], mode="INC")
        self.linear([None, None, None], mode="INC", shutter=0)
        self.linear([-lx / 2, -ly / 2, None], mode="INC", shutter=0)
        self.linear([None, None, None], mode="INC")
        self.linear([None, ly, None], mode="INC")
        self.linear([None, None, None], mode="INC", shutter=0)
        self.linear(position, mode="ABS", shutter=0)

    def ruler(
        self: MK,
        y_ticks: Union[List[float], npt.NDArray[np.float32]],
        lx: Optional[float] = None,
        lx2: Optional[float] = None,
        x_init: Optional[float] = None,
    ) -> None:
        """
        Computes the points of a ruler marker starting from a given x-coordinate. The y-coordinates of the ticks are
        specified by the user.

        :param y_ticks: y-coordinates of the ruler's ticks [mm]
        :type y_ticks: List[float]
        :param lx: Long x-tick coordinate [mm]. The default is self.lx.
        :type lx: float
        :param lx2: Second x-tick coordinate [mm]. The default is 0.75 * self.lx
        :type lx2: float
        :param x_init: Starting x-coordinate of the laser [mm]. The default is self.x_init.
        :type x_init: float
        :return: None
        """
        if y_ticks is None or len(y_ticks) == 0:
            return None

        # Sort the y-ticks and takes unique points
        y_ticks = np.unique(y_ticks)

        # Ticks x-length
        if lx is None and self.lx is None:
            raise ValueError(
                "The x-length of the ruler is None. Set the Marker's 'lx' attribute or give a valid number as input."
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
                "as input."
            )
        x_init = self.x_init if x_init is None else x_init

        # Add straight segments
        for xt, yt in zip(x_ticks, y_ticks):
            self.linear([x_init, yt, self.depth], mode="ABS", shutter=0)
            self.linear([None, None, None], mode="ABS")
            self.linear([xt, yt, None], mode="ABS")
            self.linear([None, None, None], mode="ABS", shutter=0)
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

    PARAMETERS_MK = Dotdict(
        scan=1,
        speed=2,
        speed_pos=5,
        speed_closed=5,
        depth=0.000,
        lx=1,
        ly=1,
    )

    PARAMETERS_GC = Dotdict(
        filename="testPGMcompiler.pgm",
        laser="PHAROS",
        samplesize=(10, 10),
        rotation_angle=0.0,
    )

    c = Marker(**PARAMETERS_MK)
    # c.cross([2.5, 1], 5, 2)
    c.ruler([1, 2, 3, 4], 5, 2, x_init=-2)
    print(c.points)

    from femto.PGMCompiler import PGMCompiler

    with PGMCompiler(**PARAMETERS_GC) as gc:
        gc.write(c.points)

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
