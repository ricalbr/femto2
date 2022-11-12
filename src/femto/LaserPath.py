from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import numpy.typing as npt

from femto.helpers import Dotdict
from femto.helpers import unique_filter

# Create a generic variable that can be 'LaserPath', or any subclass.
LP = TypeVar("LP", bound="LaserPath")


@dataclass
class LaserPath:
    """
    Class of irradiated paths. It manages all the coordinates of the laser path and computes the fabrication writing
    time.
    """

    scan: int = 1
    speed: float = 1.0
    x_init: float = -2.0
    y_init: float = 0.0
    z_init: float | None = None
    lsafe: float = 2.0
    speed_closed: float = 5
    speed_pos: float = 0.5
    cmd_rate_max: float = 1200
    acc_max: float = 500
    samplesize: tuple[float | None, float | None] = (None, None)
    _x: npt.NDArray = np.array([])
    _y: npt.NDArray = np.array([])
    _z: npt.NDArray = np.array([])
    _f: npt.NDArray = np.array([])
    _s: npt.NDArray = np.array([])

    def __post_init__(self):
        if not isinstance(self.scan, int):
            raise ValueError(f"Number of scan must be integer. Given {self.scan}.")

    @classmethod
    def from_dict(cls: type[LP], param: dict | Dotdict) -> LP:
        return cls(**param)

    @property
    def init_point(self: LP) -> list:
        z0 = self.z_init if self.z_init else 0.0
        return [self.x_init, self.y_init, z0]

    @property
    def lvelo(self: LP) -> float:
        # length needed to acquire the writing speed [mm]
        return 3 * (0.5 * self.speed ** 2 / self.acc_max)

    @property
    def dl(self: LP) -> float:
        # minimum separation between two points [mm]
        return self.speed / self.cmd_rate_max

    @property
    def x_end(self: LP) -> float | None:
        # end of laser path (outside the sample)
        if self.samplesize[0] is None:
            return None
        return self.samplesize[0] + self.lsafe

    @property
    def points(self: LP) -> npt.NDArray[np.float32]:
        """
        Getter for the coordinates' matrix as a numpy.ndarray matrix. The dataframe is parsed through a unique functions
        that removes all the subsequent identical points in the set.

        :return: [X, Y, Z, F, S] unique point matrix
        :rtype: numpy.ndarray
        """
        return np.array(self._unique_points()).T

    @property
    def x(self: LP) -> npt.NDArray[np.float32]:
        """
        Getter for the x-coordinate vector as a numpy array. The subsequent identical points in the vector are removed.

        :return: Array of the x-coordinates
        :rtype: numpy.ndarray
        """
        coords = self._unique_points().T
        if coords.ndim == 2:
            return np.array(coords[0])
        return np.array([])

    @property
    def lastx(self: LP) -> float | None:
        arrx = self.x
        if arrx.size:
            return float(arrx[-1])
        return None

    @property
    def y(self: LP) -> npt.NDArray[np.float32]:
        """
        Getter for the y-coordinate vector as a numpy array. The subsequent identical points in the vector are removed.

        :return: Array of the y-coordinates
        :rtype: numpy.ndarray
        """
        coords = self._unique_points().T
        if coords.ndim == 2:
            return np.array(coords[1])
        return np.array([])

    @property
    def lasty(self: LP) -> float | None:
        arry = self.y
        if arry.size:
            return float(arry[-1])
        return None

    @property
    def z(self: LP) -> npt.NDArray[np.float32]:
        """
        Getter for the z-coordinate vector as a numpy array. The subsequent identical points in the vector are removed.

        :return: Array of the z-coordinates
        :rtype: numpy.ndarray
        """
        coords = self._unique_points().T
        if coords.ndim == 2:
            return np.array(coords[2])
        return np.array([])

    @property
    def lastz(self: LP) -> float | None:
        arrz = self.z
        if arrz.size:
            return float(arrz[-1])
        return None

    @property
    def lastpt(self: LP) -> npt.NDArray[np.float32]:
        """
        Getter for the last point of the laser path.

        :return: Final point [x, y, z]
        :rtype: numpy.ndarray
        """
        if self._x.size > 0:
            return np.array([self._x[-1], self._y[-1], self._z[-1]])
        return np.array([])

    @property
    def path(self: LP) -> list[npt.NDArray[np.float32]]:
        x, y, _ = self.path3d
        return [x, y]

    @property
    def path3d(self: LP) -> list:
        if self._x.size:
            # filter 3D points without F
            x, y, z, s = unique_filter([self._x, self._y, self._z, self._s]).T
            # mask and select just those with s = 1
            x = np.delete(x, np.where(np.invert(s.astype(bool))))
            y = np.delete(y, np.where(np.invert(s.astype(bool))))
            z = np.delete(z, np.where(np.invert(s.astype(bool))))
            return [x, y, z]
        return [np.array([]), np.array([]), np.array([])]

    @property
    def length(self: LP) -> float:
        x, y, z = self.path3d
        return float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)))

    @property
    def fabrication_time(self: LP) -> float:
        """
        Getter for the laserpath fabrication time.

        :return: Fabrication time in seconds
        :rtype: float
        """
        x = np.tile(self._x, self.scan)
        y = np.tile(self._y, self.scan)
        z = np.tile(self._z, self.scan)
        f = np.tile(self._f, self.scan)

        dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
        times = dists / f[1:]
        return float(sum(times))

    # Methods
    def start(self: LP, init_pos: list[float] | None = None, speed_pos: float | None = None) -> LP:
        """
        Starts a laser path in the initial position given as input.
        The coordinates of the initial position are the first added to the matrix that describes the laser path.

        :param init_pos: Ordered list of coordinate that specifies the laser path starting point [mm].
            init_pos[0] -> X
            init_pos[1] -> Y
            init_pos[2] -> Z
        :type init_pos: List[float]
        :param speed_pos: Translation speed [mm/s].
        :type speed_pos: float
        :return: Self
        :rtype: LaserPath
        """
        if self._x.size != 0:
            raise ValueError("Coordinate matrix is not empty. Cannot start a new laser path in this point.")

        if init_pos is None:
            xi, yi, zi = self.init_point
        else:
            if np.size(init_pos) != 3:
                raise ValueError(f"Given initial position is not valid. 3 values required. {np.size(init_pos)} given.")
            xi, yi, zi = init_pos

        if speed_pos is None:
            speed_pos = self.speed_pos

        x0 = np.array([xi, xi])
        y0 = np.array([yi, yi])
        z0 = np.array([zi, zi])
        f0 = np.array([speed_pos, speed_pos])
        s0 = np.array([0.0, 1.0])

        self.add_path(x0, y0, z0, f0, s0)
        return self

    def end(self: LP) -> None:
        """
        Ends a laser path. The function automatically return to the initial point of the laser path with a translation
        speed specified by the user.

        :return: None
        :rtype: None
        """

        if not self._x.size:
            raise IndexError("Try to access an empty array. Use the start() method before the end() method.")

        # append the transformed path and add the coordinates to return to the initial point
        x = np.array([self._x[-1], self._x[0]])
        y = np.array([self._y[-1], self._y[0]])
        z = np.array([self._z[-1], self._z[0]])
        f = np.array([self._f[-1], self.speed_closed])
        s = np.array([0, 0])
        self.add_path(x, y, z, f, s)

    def linear(
        self: LP,
        increment: list,
        mode: str = "INC",
        shutter: int = 1,
        speed: float | None = None,
    ) -> LP:
        """
        Adds a linear increment to the last point of the current laser path.

        :param increment: Ordered list of coordinate that specifies the increment if mode is INC or new position if
        mode is ABS.
            increment[0] -> X-coord [mm]
            increment[1] -> Y-coord [mm]
            increment[2] -> Z-coord [mm]
        :type increment: List[float, float, float]
        :param mode: Select incremental or absolute mode. The default is 'INC'.
        :type mode: str
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.param.speed.
        :type speed: float
        :return: Self
        :rtype: LaserPath

        :raise ValueError: Mode is neither INC nor ABS.
        """
        if mode.lower() not in ["abs", "inc"]:
            raise ValueError(f"Mode should be either ABS or INC. {mode.upper()} was given.")

        if len(increment) != 3:
            raise ValueError(f"Increment should be a list of three values. Increment has {len(increment)} entries.")

        if (speed or self.speed) is None:
            raise ValueError('Speed is None. Set LaserPath\'s "speed" attribute or give a speed as input.')

        if mode.lower() == "abs":
            # If increment is None use the last value on the coordinate-array
            x_inc = np.array([increment[0] or self._x[-1]])
            y_inc = np.array([increment[1] or self._y[-1]])
            z_inc = np.array([increment[2] or self._z[-1]])
        else:
            x, y, z = map(lambda k: k or 0, increment)
            x_inc = np.array([self._x[-1] + x])
            y_inc = np.array([self._y[-1] + y])
            z_inc = np.array([self._z[-1] + z])

        f_inc = np.array([speed or self.speed])
        s_inc = np.array([shutter])

        self.add_path(x_inc, y_inc, z_inc, f_inc, s_inc)
        return self

    def subs_num(self: LP, l_curve: float = 0, speed: float | None = None) -> int:
        """
        Utility function that, given the length of a segment and the fabrication speed, computes the number of points
        required to work at the maximum command rate (attribute of LaserPath obj).

        :param l_curve: Length of the laser path segment [mm]. The default is 0 mm.
        :type l_curve: float
        :param speed: Fabrication speed [mm/s]. The default is 0 mm/s.
        :type speed: float
        :return: Number of subdivisions.
        :rtype: int

        :raise ValueError: Speed is set too low.
        """
        f = self.speed if speed is None else speed
        if f < 1e-6:
            raise ValueError("Speed set to 0.0 mm/s. Check speed parameter.")

        dl = f / self.cmd_rate_max
        num = int(np.ceil(l_curve / dl))
        if num <= 1:
            print("I had to add use an higher instruction rate.\n")
            return 3
        return num

    def add_path(
        self: LP,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        z: npt.NDArray[np.float32],
        f: npt.NDArray[np.float32],
        s: npt.NDArray[np.float32],
    ):
        """
        Takes [x, y, z, f, s] numpy.ndarrays and adds it to the class coordinates.

        :param x: array of x-coordinates.
        :type x: numpy.ndarray
        :param y: array of y-coordinates.
        :type y: numpy.ndarray
        :param z: array of z-coordinates.
        :type z: numpy.ndarray
        :param f: array of feed rates (speed) coordinates.
        :type f: numpy.ndarray
        :param s: array of shutter coordinates.
        :type s: numpy.ndarray
        """
        self._x = np.append(self._x, x.astype(np.float32))
        self._y = np.append(self._y, y.astype(np.float32))
        self._z = np.append(self._z, z.astype(np.float32))
        self._f = np.append(self._f, f.astype(np.float32))
        self._s = np.append(self._s, s.astype(np.float32))

    # Private interface
    def _unique_points(self: LP) -> npt.NDArray[np.float32]:
        """
        Remove duplicate subsequent points. At least one coordinate have to change between two consecutive lines of the
        (X,Y,Z,F,S) matrix.

        Duplicates can be selected by creating a boolean index mask as follows:
            - make a row-wise diff (`numpy.diff <https://numpy.org/doc/stable/reference/generated/numpy.diff.html>`_)
            - compute absolute value of all elements in order to work only with positive numbers
            - make a column-wise sum (`numpy.diff <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_)
            - mask is converted to boolean values
        In this way consecutive duplicates correspond to a 0 value in the latter array.
        Converting this array to boolean (all non-zero values are True) the index mask can be retrieved.
        The first element is set to True by default since it is lost by the diff operation.

        :return: Modified coordinate matrix (x, y, z, f, s) without duplicates.
        :rtype: numpy.ndarray
        """
        return unique_filter([self._x, self._y, self._z, self._f, self._s])


def main():
    from femto.helpers import Dotdict
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Data
    PARAMETERS_LP = Dotdict(
        scan=6,
        speed=20,
        lsafe=3,
    )

    lpath = LaserPath(**PARAMETERS_LP)

    path_x = np.array([0, 1, 1, 2, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4])
    path_y = np.array([0, 0, 2, 3, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4])
    path_z = np.array([0, 0, 0, 3, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4])
    path_f = np.array([1, 2, 3, 4, 3, 1, 1, 1, 1, 3, 1, 1, 1, 6, 1])
    path_s = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 0])
    lpath.add_path(path_x, path_y, path_z, path_f, path_s)
    print(lpath.points.T)

    # Plot
    fig = plt.figure()
    fig.clf()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.plot(lpath.x, lpath.y, lpath.z, "-k", linewidth=2.5)
    ax.set_box_aspect(aspect=(3, 1, 0.5))
    plt.show()

    print(f"Expected writing time {lpath.fabrication_time:.3f} seconds")
    print(f"Laser path length {lpath.length:.6f} mm")


if __name__ == "__main__":
    main()
