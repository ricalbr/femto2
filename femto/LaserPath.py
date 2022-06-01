
from copy import deepcopy

import numpy as np

class LaserPath:
    def __init__(self, param):
        self.param = param
        self.twriting = 0

        # Points
        self._x = np.asarray([])
        self._y = np.asarray([])
        self._z = np.asarray([])
        self._f = np.asarray([])
        self._s = np.asarray([])

    @property
    def points(self) -> np.ndarray:
        """
        COORDINATES MATRIX GETTER.

        The getter returns the coordinates' matrix as a numpy.ndarray matrix.
        The dataframe is parsed through a unique functions that removes all the
        subsequent identical points in the set.

        Returns
        -------
        numpy.ndarray
            [X, Y, Z, F, S] unique point matrix.

        """
        return self._unique_points()

    @property
    def x(self) -> np.ndarray:
        """
        X-COORDINATE.

        The getter returns the x-coordinate vector as a numpy array. The
        subsequent identical points in the vector are removed.

        Returns
        -------
        numpy.ndarray
            Array of the x-coordinates.

        """
        coords = self._unique_points().T
        return coords[0]

    @property
    def y(self) -> np.ndarray:
        """
        Y-COORDINATE.

        The getter returns the y-coordinate vector as a numpy array. The
        subsequent identical points in the vector are removed.

        Returns
        -------
        numpy.ndarray
            Array of the y-coordinates.

        """
        coords = self._unique_points().T
        return coords[1]

    @property
    def z(self) -> np.ndarray:
        """
        Z-COORDINATE.

        The getter returns the z-coordinate vector as a numpy array. The
        subsequent identical points in the vector are removed.

        Returns
        -------
        numpy.ndarray
            Array of the z-coordinates.

        """
        coords = self._unique_points().T
        return coords[2]

    @property
    def lastpt(self) -> np.ndarray:
        """
        LAST POINT.

        The function return the last point of the waveguide.

        Returns
        -------
        numpy.ndarray
            Final point [x, y, z].

        """
        if self._x.size > 0:
            return np.array([self._x[-1],
                             self._y[-1],
                             self._z[-1]])
        return np.array([])

    def fabrication_time(self):
        """ It computes the time needed to travel along the line."""
        points = np.stack((self._x, self._y, self._z), axis=-1).astype(np.float32)
        linelength = np.linalg.norm(np.diff(points))
        self.twriting = linelength * (1 / self.param.speed + 1 / self.param.speed) * self.param.scan

    def add_path(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, f: np.ndarray, s: np.ndarray):
        """ It takes a LaserPath object and it adds it to itself """
        self._x = np.append(self._x, x)
        self._y = np.append(self._y, y)
        self._z = np.append(self._z, z)
        self._f = np.append(self._f, f)
        self._s = np.append(self._s, s)

    def compensate(self, pts):
        """
        pts : [X,Y,Z] matrix or just a single point
        It returns the points compensated along Z
        for the refractive index, the offset and the glass warp.
        """
        pts = np.array(pts)
        pts_comp = deepcopy(pts)

        if pts_comp.size > 3:
            zwarp = [float(self.param.fwarp(x, y)) for x, y
                     in zip(pts_comp[:, 0], pts_comp[:, 1])]
            zwarp = np.array(zwarp)
            pts_comp[:, 2] = (pts_comp[:, 2] / self.param.neff
                              + self.param.zoff
                              + zwarp)
        else:
            pts_comp[2] = (pts_comp[2] / self.param.neff
                           + self.param.zoff
                           + self.param.fwarp(pts_comp[0], pts_comp[1]))
        return pts_comp

    # def linear(self, pos_fin, shutter='ON', *, mode='INC'):
        # linelength = np.sqrt(np.sum(np.square(pos_fin-pos_ini)))
        # num = int(linelength/self.param.lwarp)

        # # long line case (a lot of points)
        # if num > 3 and shutter == 'ON':
        #     points = np.vstack((np.linspace(pos_ini[0], pos_fin[0], num),
        #                         np.linspace(pos_ini[1], pos_fin[1], num),
        #                         np.linspace(pos_ini[2], pos_fin[2], num))).T
        #     points_comp = self.compensate(points)
        # # short line case (just 2 points)
        # else:
        #     num = 2
        #     pmid = (pos_fin+pos_ini)/2
        #     points = np.vstack((pmid, pos_fin))
        #     points_comp = self.compensate(points)
