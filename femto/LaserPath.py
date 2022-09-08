from operator import add

import numpy as np

from femto.helpers import dotdict
from femto.Parameters import WaveguideParameters


class LaserPath(WaveguideParameters):
    """
    Class of irradiated paths. It manages all the coordinates of the laser path and computes the fabrication writing
    time.
    """

    def __init__(self, param: dict):
        super().__init__(**param)
        self.wtime = 0

        # Points
        self._x = np.asarray([])
        self._y = np.asarray([])
        self._z = np.asarray([])
        self._f = np.asarray([])
        self._s = np.asarray([])

    @property
    def points(self) -> np.ndarray:
        """
        Getter for the coordinates' matrix as a numpy.ndarray matrix. The dataframe is parsed through a unique functions
        that removes all the subsequent identical points in the set.

        :return: [X, Y, Z, F, S] unique point matrix
        :rtype: numpy.ndarray
        """
        return self._unique_points()

    @property
    def x(self) -> np.ndarray:
        """
        Getter for the x-coordinate vector as a numpy array. The subsequent identical points in the vector are removed.

        :return: Array of the x-coordinates
        :rtype: numpy.ndarray
        """
        coords = self._unique_points().T
        return coords[0]

    @property
    def lastx(self) -> float:
        return self.x[-1]

    @property
    def y(self) -> np.ndarray:
        """
        Getter for the y-coordinate vector as a numpy array. The subsequent identical points in the vector are removed.

        :return: Array of the y-coordinates
        :rtype: numpy.ndarray
        """
        coords = self._unique_points().T
        return coords[1]

    @property
    def lasty(self) -> float:
        return self.y[-1]

    @property
    def z(self) -> np.ndarray:
        """
        Getter for the z-coordinate vector as a numpy array. The subsequent identical points in the vector are removed.

        :return: Array of the z-coordinates
        :rtype: numpy.ndarray
        """
        coords = self._unique_points().T
        return coords[2]

    @property
    def lastz(self) -> float:
        return self.z[-1]

    @property
    def lastpt(self) -> np.ndarray:
        """
        Getter for the last point of the waveguide.

        :return: Final point [x, y, z]
        :rtype: numpy.ndarray
        """
        if self._x.size > 0:
            return np.array([self._x[-1], self._y[-1], self._z[-1]])
        return np.array([])

    def add_path(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, f: np.ndarray, s: np.ndarray):
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
        self._x = np.append(self._x, x)
        self._y = np.append(self._y, y)
        self._z = np.append(self._z, z)
        self._f = np.append(self._f, f)
        self._s = np.append(self._s, s)

    def fabrication_time(self):
        """
        Computes the time needed to travel along the line.
        """
        x, y, z = self._x, self._y, self._z

        dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
        linelength_shutter_on = np.sum(dists[:-1])
        linelength_shutter_off = dists[-1]
        self.wtime = (linelength_shutter_on * 1 / self.speed) * self.scan + \
                     (linelength_shutter_off * 1 / self.speed_closed) * self.scan

    def flip_path(self):
        """
        Flip the laser path along the x-, y- and z-coordinates
        :return: None
        """

        m = np.array([1, 1, 1])
        if self.flip_x:
            m[0] = -1
            xc = np.flip(self._x)
        else:
            xc = self._x
        if self.flip_y:
            m[1] = -1
            yc = np.flip(self._y)
        else:
            yc = self._y
        if self.flip_z:
            m[2] = -1
            zc = np.flip(self._z)
        else:
            zc = self._z

        M = np.diag(m)
        C = np.array([xc, yc, zc])
        d = np.array([np.max(xc) + np.min(xc),
                      np.max(yc) + np.min(yc),
                      np.max(zc) + np.min(zc)])
        S = np.multiply((1 - m) / 2, d)

        flip_x, flip_y, flip_z = map(add, M @ C, S)

        if self.flip_x:
            self._x = np.flip(flip_x)
        else:
            self._x = flip_x
        if self.flip_y:
            self._y = np.flip(flip_y)
        else:
            self._y = flip_y
        if self.flip_z:
            self._z = np.flip(flip_z)
        else:
            self._z = flip_z

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
        data = np.stack((self._x, self._y, self._z, self._f, self._s), axis=-1).astype(np.float32)
        mask = np.diff(data, axis=0)
        mask = np.sum(np.abs(mask), axis=1, dtype=bool)
        mask = np.insert(mask, 0, True)
        return np.delete(data, np.where(mask is False), 0).astype(np.float32)


def _example():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from femto import Waveguide

    # Data
    PARAMETERS_WG = dotdict(
            scan=6,
            speed=20,
            radius=15,
            pitch=0.080,
            int_dist=0.007,
            lsafe=3,
    )

    increment = [PARAMETERS_WG.lsafe, 0, 0]

    # Calculations
    mzi = [Waveguide(PARAMETERS_WG) for _ in range(2)]
    for index, wg in enumerate(mzi):
        [xi, yi, zi] = [-2, -wg.pitch / 2 + index * wg.pitch, 0.035]

        wg.start([xi, yi, zi]) \
            .linear([10, 0, 0]) \
            .sin_mzi((-1) ** index * wg.dy_bend) \
            .spline_bridge((-1) ** index * 0.08, (-1) ** index * 0.015) \
            .sin_mzi((-1) ** (index + 1) * wg.dy_bend) \
            .linear(increment)
        wg.end()

    print(wg.x)

    # Plot
    fig = plt.figure()
    fig.clf()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    for wg in mzi:
        ax.plot(wg.x[:-1], wg.y[:-1], wg.z[:-1], '-k', linewidth=2.5)
        ax.plot(wg.x[-2:], wg.y[-2:], wg.z[-2:], ':b', linewidth=1.0)
    ax.set_box_aspect(aspect=(3, 1, 0.5))
    plt.show()


if __name__ == '__main__':
    _example()
