import numpy as np


class LaserPath:
    """
    Class of irradiated paths. It manages all the coordinates of the laser path and computes the fabrication writing
    time.
    """

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
    def y(self) -> np.ndarray:
        """
        Getter for the y-coordinate vector as a numpy array. The subsequent identical points in the vector are removed.

        :return: Array of the y-coordinates
        :rtype: numpy.ndarray
        """
        coords = self._unique_points().T
        return coords[1]

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
        points = np.stack((self._x, self._y, self._z), axis=-1).astype(np.float32)
        linelength = np.linalg.norm(np.diff(points))
        self.twriting = linelength * (1 / self.param.speed + 1 / self.param.speed) * self.param.scan

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
