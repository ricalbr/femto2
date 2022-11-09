from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import numpy.typing as npt

from femto.helpers import unique_filter

# Create a generic variable that can be 'LaserPath', or any subclass.
C = TypeVar('C', bound='LaserPath')


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
    z_init: Optional[float] = None
    lsafe: float = 2.0
    speed_closed: float = 5
    speed_pos: float = 0.5
    cmd_rate_max: float = 1200
    acc_max: float = 500
    samplesize: Tuple[Optional[float], Optional[float]] = (None, None)
    _x: npt.NDArray = np.array([])
    _y: npt.NDArray = np.array([])
    _z: npt.NDArray = np.array([])
    _f: npt.NDArray = np.array([])
    _s: npt.NDArray = np.array([])

    def __post_init__(self):
        if not isinstance(self.scan, int):
            raise ValueError(f'Number of scan must be integer. Given {self.scan}.')

    @classmethod
    def from_dict(cls: Type[C], param: Union[dict, dotdict]) -> C:
        return cls(**param)

    @property
    def init_point(self) -> List:
        z0 = self.z_init if self.z_init else 0.0
        return [self.x_init, self.y_init, z0]

    @property
    def lvelo(self) -> float:
        # length needed to acquire the writing speed [mm]
        return 3 * (0.5 * self.speed ** 2 / self.acc_max)

    @property
    def dl(self) -> float:
        # minimum separation between two points [mm]
        return self.speed / self.cmd_rate_max

    @property
    def x_end(self) -> Optional[float]:
        # end of laser path (outside the sample)
        if self.samplesize[0] is None:
            return None
        return self.samplesize[0] + self.lsafe

    @property
    def points(self) -> npt.NDArray[np.float32]:
        """
        Getter for the coordinates' matrix as a numpy.ndarray matrix. The dataframe is parsed through a unique functions
        that removes all the subsequent identical points in the set.

        :return: [X, Y, Z, F, S] unique point matrix
        :rtype: numpy.ndarray
        """
        return np.array(self._unique_points())

    @property
    def x(self) -> npt.NDArray[np.float32]:
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
    def lastx(self) -> Optional[float]:
        arrx = self.x
        if arrx.size:
            return float(arrx[-1])
        return None

    @property
    def y(self) -> npt.NDArray[np.float32]:
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
    def lasty(self) -> Optional[float]:
        arry = self.y
        if arry.size:
            return float(arry[-1])
        return None

    @property
    def z(self) -> npt.NDArray[np.float32]:
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
    def lastz(self) -> Optional[float]:
        arrz = self.z
        if arrz.size:
            return float(arrz[-1])
        return None

    @property
    def lastpt(self) -> npt.NDArray[np.float32]:
        """
        Getter for the last point of the waveguide.

        :return: Final point [x, y, z]
        :rtype: numpy.ndarray
        """
        if self._x.size > 0:
            return np.array([self._x[-1], self._y[-1], self._z[-1]])
        return np.array([])

    @property
    def path(self) -> List[npt.NDArray[np.float32]]:
        x, y, _ = self.path3d
        return [x, y]

    @property
    def path3d(self) -> List:
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
    def length(self) -> float:
        x, y, z = self.path3d
        return float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)))

    @property
    def fabrication_time(self) -> float:
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

    def subs_num(self, l_curve: float = 0, speed: Optional[float] = None) -> int:
        """
        Utility function that, given the length of a segment and the fabrication speed, computes the number of points
        required to work at the maximum command rate (attribute of Waveguide obj).

        :param l_curve: Length of the waveguide segment [mm]. The default is 0 mm.
        :type l_curve: float
        :param speed: Fabrication speed [mm/s]. The default is 0 mm/s.
        :type speed: float
        :return: Number of subdivisions.
        :rtype: int

        :raise ValueError: Speed is set too low.
        """
        f = self.speed if speed is None else speed
        if f < 1e-6:
            raise ValueError('Speed set to 0.0 mm/s. Check speed parameter.')

        dl = f / self.cmd_rate_max
        num = int(np.ceil(l_curve / dl))
        if num <= 1:
            print('I had to add use an higher instruction rate.\n')
            return 3
        return num

    def add_path(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], z: npt.NDArray[np.float32],
                 f: npt.NDArray[np.float32], s: npt.NDArray[np.float32]):
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
    def _unique_points(self):
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


if __name__ == '__main__':
    from femto.helpers import dotdict
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Data
    PARAMETERS_LP = dotdict(
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
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.plot(lpath.x, lpath.y, lpath.z, '-k', linewidth=2.5)
    ax.set_box_aspect(aspect=(3, 1, 0.5))
    plt.show()

    print("Expected writing time {:.3f} seconds".format(lpath.fabrication_time))
    print("Laser path length {:.6f} mm".format(lpath.length))
