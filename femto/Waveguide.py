from typing import List

import numpy as np

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from functools import partialmethod


class Waveguide:
    def __init__(self, num_scan: int = None, c_max: int = 1200):

        self.num_scan = num_scan
        self.c_max = c_max
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

    # Methods
    def start(self, init_pos: List[float], speed: float = 5):
        """
        START.

        The function starts a waveguide in the initial position given as
        input.
        The coordinates of the initial position are the first added to the
        matrix that describes the waveguide.

        Parameters
        ----------
        init_pos : List[float]
            Ordered list of coordinate that specifies the waveguide starting
            point [mm].
            init_pos[0] -> X
            init_pos[1] -> Y
            init_pos[2] -> Z
        speed : float, optional
            Translation speed. The default is 5.

        Returns
        -------
        None.

        """
        if np.size(init_pos) != 3:
            raise ValueError('Given initial position is not valid.',
                             f'3 values required. {np.size(init_pos)} given.')
        if self._x.size != 0:
            raise ValueError('Coordinate matrix is not empty. ',
                             'Cannot start a new waveguide in this point.')

        x0, y0, z0 = init_pos
        self._x = np.append(self._x, x0)
        self._y = np.append(self._y, y0)
        self._z = np.append(self._z, z0)
        self._f = np.append(self._f, speed)
        self._s = np.append(self._s, 0)

    def end(self, speed: float = 75):
        """
        END.

        End a waveguide. The function automatically

        Parameters
        ----------
        speed : float, optional
            Traslation speed. The default is 75.

        Returns
        -------
        None.

        """
        self._x = np.append(self._x, [self._x[-1], self._x[0]])
        self._y = np.append(self._y, [self._y[-1], self._y[0]])
        self._z = np.append(self._z, [self._z[-1], self._z[0]])
        self._f = np.append(self._f, [self._f[-1], speed])
        self._s = np.append(self._s, [0, 0])

    def linear(self,
               increment: List[float],
               mode: str = 'INC',
               speed: float = 0.0,
               shutter: int = 1) -> Self:
        """
        LINEAR.

        The function add a linear increment to the last point of the current
        waveguide.


        Parameters
        ----------
        increment : List[float]
            Ordered list of coordinate that specifies the increment if mode
            is INC or new position if mode is ABS. Units are [mm].
            increment[0] -> X-coord
            increment[1] -> Y-coord
            increment[2] -> Z-coord
        mode : str, optional
            Select incremental or absolute mode. The default is 'INC'.
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.

        Raises
        ------
        ValueError
            Mode is neither INC nor ABS.

        Returns
        -------
        None.

        """
        if mode.upper() not in ['ABS', 'INC']:
            raise ValueError('Mode should be either ABS or INC.',
                             f'{mode.upper()} was given.')
        x_inc, y_inc, z_inc = increment
        if mode.upper() == 'ABS':
            self._x = np.append(self._x, x_inc)
            self._y = np.append(self._y, y_inc)
            self._z = np.append(self._z, z_inc)
        else:
            self._x = np.append(self._x, self._x[-1] + x_inc)
            self._y = np.append(self._y, self._y[-1] + y_inc)
            self._z = np.append(self._z, self._z[-1] + z_inc)
        self._f = np.append(self._f, speed)
        self._s = np.append(self._s, shutter)
        return self

    def circ(self,
             initial_angle: float,
             final_angle: float,
             radius: float,
             speed: float = 0.0,
             shutter: int = 1,
             N: int = 25) -> Self:
        """
        CIRC.

        Compute the points in the xy-plane that connects two angles
        (initial_angle and final_angle) with a circular arc of a given radius.
        The user can set the transition speed, the shutter state during
        the movement and the number of points of the arc.

        Parameters
        ----------
        initial_angle : float
            Starting angle of the circular arc [radians].
        final_angle : float
            Ending angle of the circular arc [radians].
        radius : float
            Radius of the circular arc [mm].
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        N : int, optional
            Number of points. The default is 25.

        Raises
        ------
        ValueError
            Speed values has the wrong shape.

        Returns
        -------
        None.

        """
        t = np.linspace(initial_angle, final_angle, N)
        new_x = self._x[-1] - radius * np.cos(initial_angle) + radius * np.cos(t)
        new_y = self._y[-1] - radius * np.sin(initial_angle) + radius * np.sin(t)
        new_z = self._z[-1] * np.ones(new_x.shape)

        # update coordinates
        self._x = np.append(self._x, new_x)
        self._y = np.append(self._y, new_y)
        self._z = np.append(self._z, new_z)

        # update speed array
        if np.size(speed) == 1:
            self._f = np.append(self._f, speed * np.ones(new_x.shape))
        elif np.size(speed) == np.size(new_x):
            self._f = np.append(self._f, speed)
        else:
            raise ValueError('Speed array is neither a single value nor ',
                             'array of appropriate size.')

        self._s = np.append(self._s, shutter * np.ones(new_x.shape))
        return self

    def arc_bend(self,
                 dy: float,
                 radius: float,
                 speed: float = 0.0,
                 shutter: int = 1,
                 N: int = 25) -> Self:
        """
        CIRCULAR BEND.

        The function concatenate two circular arc to make a circular S-bend.
        The user can specify the amplitude of the S-bend (height in the y
        direction) and the curvature radius. Starting and ending angle of the
        two arcs are computed automatically.
        The sign of D encodes the direction of the S-bend:
            - D > 0, upward S-bend
            - D < 0, downward S-bend

        Parameters
        ----------
        dy : float
            Amplitude of the S-bend along the y direction [mm].
        radius : float
            Curvature radius of the S-bend [mm].
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        N : int, optional
            Number of points. The default is 25.

        Returns
        -------
        None.

        """
        (a, _) = self.get_sbend_parameter(dy, radius)

        if dy > 0:
            self.circ(np.pi * (3 / 2),
                      np.pi * (3 / 2) + a,
                      radius,
                      speed=speed,
                      shutter=shutter,
                      N=np.round(N / 2))
            self.circ(np.pi * (1 / 2) + a,
                      np.pi * (1 / 2),
                      radius,
                      speed=speed,
                      shutter=shutter,
                      N=np.round(N / 2))
        else:
            self.circ(np.pi * (1 / 2),
                      np.pi * (1 / 2) - a,
                      radius,
                      speed=speed,
                      shutter=shutter,
                      N=np.round(N / 2))
            self.circ(np.pi * (3 / 2) - a,
                      np.pi * (3 / 2),
                      radius,
                      speed=speed,
                      shutter=shutter,
                      N=np.round(N / 2))
            return self

    def arc_acc(self,
                dy: float,
                radius: float,
                arm_length: float = 0.0,
                speed: float = 0.0,
                shutter: int = 1,
                N: int = 50) -> Self:
        """
        CIRCULAR COUPLER.

        The function concatenate two circular S-bend to make a single mode of
        a circular directional coupler.
        The user can specify the amplitude of the coupler (height in the y
        direction) and the curvature radius.
        The sign of D encodes the direction of the coupler:
            - D > 0, upward S-bend
            - D < 0, downward S-bend

        Parameters
        ----------
        dy : float
            Amplitude of the coupler along the y-direction [mm].
        radius : float
            Curvature radius of the coupler's bends [mm].
        arm_length : float, optional
            Length of the coupler straight arm [mm]. The default is 0.0.
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        N : int, optional
            Number of points. The default is 50.

        Returns
        -------
        None.

        """
        self.arc_bend(dy, radius,
                      speed=speed,
                      shutter=shutter,
                      N=N / 2)
        self.linear([arm_length, 0, 0], speed=speed, shutter=shutter)
        self.arc_bend(-dy, radius,
                      speed=speed,
                      shutter=shutter,
                      N=N / 2)
        return self

    def arc_mzi(self,
                dy: float,
                radius: float,
                int_length: float = 0.0,
                arm_length: float = 0.0,
                speed: float = 0.0,
                shutter: int = 1,
                N: int = 100) -> Self:
        """
        CIRCULAR MACH-ZEHNDER INTERFEROMETER (MZI).

        The function concatenate two circular couplers to make a single mode
        of a circular MZI.
        The user can specify the amplitude of the coupler (height in the y
        direction) and the curvature radius.
        The sign of D encodes the direction of the coupler:
            - D > 0, upward S-bend
            - D < 0, downward S-bend

        Parameters
        ----------
        dy : float
            Amplitude of the MZI along the y-direction [mm].
        radius : float
            Curvature radius of the MZI's bends [mm].
        int_length : float, optional
            Interaction distance of the MZI [mm]. The default is 0.0.
        arm_length : float, optional
            Length of the MZI straight arm [mm]. The default is 0.0.
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        N : int, optional
            Number of points. The default is 100.

        Returns
        -------
        None.

        """
        self.arc_acc(dy, radius,
                     arm_length=arm_length,
                     speed=speed,
                     shutter=shutter,
                     N=N / 2)
        self.linear([int_length, 0, 0], speed=speed, shutter=shutter)
        self.arc_acc(dy, radius,
                     arm_length=arm_length,
                     speed=speed,
                     shutter=shutter,
                     N=N / 2)
        return self

    def sin_bridge(self,
                   dy: float,
                   radius: float,
                   dz: float = None,
                   speed: float = 0.0,
                   shutter: int = 1,
                   N: int = 25) -> Self:
        """
        SINUSOIDAL BRIDGE.

        The function compute the points in the xy-plane of a Sin-bend curve
        and the points in the xz-plane of a Sin-bridge of height Dz.
        The distance between the initial and final point is the same of the
        equivalent (circular) S-bend of given radius.
        The user can specify the amplitude of the Sin-bend (height in the y
        direction) and the curvature radius as well as the transition speed,
        the shutter state during the movement and the number of points of the
        arc.
        The sign of Dy encodes the direction of the Sin-bend:
            - Dy > 0, upward S-bend
            - Dy < 0, downward S-bend
        The sign of Dz encodes the direction of the Sin-bridge:
            - Dz > 0, top bridge
            - Dz < 0, under bridge

        NB: the radius is an effective radius. The radius of curvature of the
            overall curve will be lower (in general) than the specified
            radius.

        Parameters
        ----------
        dy : float
            Amplitude of the Sin-bend along the y-direction [mm].
        radius : float
            Effective curvature radius of the Sin-bend [mm].
        dz : float, optional
            Height of the Sin-bridge along the z-direction [mm].
            The default is None.
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        N : int, optional
            Number of points. The default is 25.

        Raises
        ------
        ValueError
            Speed values has the wrong shape.

        Returns
        -------
        None.

        """
        (_, dx) = self.get_sbend_parameter(dy, radius)

        new_x = np.arange(self._x[-1], self._x[-1] + dx, dx / (N - 1))
        new_y = self._y[-1] + 0.5 * dy * (1 - np.cos(np.pi / dx * (new_x - self._x[-1])))
        if dz:
            new_z = self._z[-1] + 0.5 * dz * (1 - np.cos(2 * np.pi / dx * (new_x - self._x[-1])))
        else:
            new_z = self._z[-1] * np.ones(new_x.shape)

        # update coordinates
        self._x = np.append(self._x, new_x)
        self._y = np.append(self._y, new_y)
        self._z = np.append(self._z, new_z)

        # update speed array
        if np.size(speed) == 1:
            self._f = np.append(self._f, speed * np.ones(new_x.shape))
        elif np.size(speed) == np.size(new_x):
            self._f = np.append(self._f, speed)
        else:
            raise ValueError('Speed array is neither a single value nor ',
                             'array of appropriate size.')

        self._s = np.append(self._s, shutter * np.ones(new_x.shape))
        return self

    sin_bend = partialmethod(sin_bridge, dz=None)

    def sin_acc(self,
                dy: float,
                radius: float,
                int_length: float = 0.0,
                speed: float = 0.0,
                shutter: int = 1,
                N: int = 50) -> Self:
        """
        SINUSOIDAL COUPLER.

        The function concatenate two Sin-bend to make a single mode of a
        sinusoidal directional coupler.
        The user can specify the amplitude of the coupler (height in the y
        direction) and the effective curvature radius.
        The sign of Dy encodes the direction of the coupler:
            - Dy > 0, upward Sin-bend
            - Dy < 0, downward Sin-bend

        Parameters
        ----------
        dy : float
            Amplitude of the coupler along the y-direction [mm].
        radius : float
            Effective curvature radius of the coupler's bends [mm].
        int_length : float, optional
            Length of the coupler straight arm [mm]. The default is 0.0.
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        N : int, optional
            Number of points. The default is 50.

        Returns
        -------
        None.

        """
        self.sin_bend(dy, radius, speed=speed, shutter=shutter, N=N / 2)
        self.linear([int_length, 0, 0], speed=speed, shutter=shutter)
        self.sin_bend(-dy, radius, speed=speed, shutter=shutter, N=N / 2)
        return self

    def sin_mzi(self,
                dy: float,
                radius: float,
                int_length: float = 0.0,
                arm_length: float = 0.0,
                speed: float = 0.0,
                shutter: int = 1,
                N: float = 100) -> Self:
        """
        SINUSOIDAL MACH-ZEHNDER INTERFEROMETER (MZI).

        The function concatenate two sinusoidal couplers to make a single mode
        of a sinusoidal MZI.
        The user can specify the amplitude of the coupler (height in the y
        direction) and the curvature radius.
        The sign of Dy encodes the direction of the coupler:
            - Dy > 0, upward S-bend
            - Dy < 0, downward S-bend

        Parameters
        ----------
        dy : float
            Amplitude of the MZI along the y-direction [mm].
        radius : float
            Effective curvature radius of the coupler's bends [mm].
        int_length : float, optional
            Interaction distance of the MZI [mm]. The default is 0.0.
        arm_length : float, optional
            Length of the MZI straight arm [mm]. The default is 0.0.
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        N : int, optional
            Number of points. The default is 100.

        Returns
        -------
        None.

        """
        self.sin_acc(dy, radius,
                     int_length=int_length,
                     speed=speed,
                     shutter=shutter,
                     N=N / 2)
        self.linear([arm_length, 0, 0], speed=speed, shutter=shutter)
        self.sin_acc(dy, radius,
                     int_length=int_length,
                     speed=speed,
                     shutter=shutter,
                     N=N / 2)
        return self

    def spline(self,
               dy: float,
               dz: float,
               init_pos: np.ndarray = None,
               radius: float = 20,
               disp_x: float = 0,
               speed: float = 0,
               shutter: int = 1,
               bc_y: tuple = ((1, 0.0), (1, 0.0)),
               bc_z: tuple = ((1, 0.0), (1, 0.0))) -> Self:
        """
        SPLINE

        Function wrapper. It computes the x,y,z coordinates of spline curve
        starting from init_pos with Dy and Dz displacements.
        See :func:`~femto.Waveguide._get_spline_points`.
        The points are then appended to the Waveguide coordinate list.

        Returns
        -------
        None.

        """
        x_spl, y_spl, z_spl = self._get_spline_points(locals())

        # update coordinates or return
        self._x = np.append(self._x, x_spl)
        self._y = np.append(self._y, y_spl)
        self._z = np.append(self._z, z_spl)

        # update speed array
        if np.size(speed) == 1:
            self._f = np.append(self._f, speed * np.ones(x_spl.shape))
        elif np.size(speed) == np.size(x_spl):
            self._f = np.append(self._f, speed)
        else:
            raise ValueError('Speed array is neither a single value nor ',
                             'array of appropriate size.')

        self._s = np.append(self._s, shutter * np.ones(x_spl.shape))
        return self

    def spline_bridge(self,
                      dy: float,
                      dz: float,
                      init_pos: list = None,
                      radius: float = 20,
                      disp_x: float = 0,
                      speed: float = 0,
                      shutter: int = 1) -> Self:
        """
        SPLINE BRIDGE.

        Compute a spline bridge as a sequence of two spline segments. Dy is
        the total displacement along the y-direction of the bridge and Dz is
        the height of the bridge along z.
        First, the function computes the dx-displacement of the planar spline
        curve with a Dy displacement. This datum is used to compute the value
        of the first derivative along the y-coordinate for the costruction of
        the spline bridge such that
            df(x, y)/dx = Dy/dx
        in the peak point of the bridge.

        The cubic spline bridge obtained in this way has a first derivatives
        df(x, y)/dx, df(x, z)/dx which are zero (for construction) in the
        initial and final point.
        However, the second derivatives are not null in principle. To cope with
        this the spline points are fitted with a spline of the 5-th order.
        In this way the final curve has second derivatives close to zero
        (~1e-4) while maintaining the first derivative to zero.

        Parameters
        ----------
        dy : float
            Displacement along y-direction [mm].
        dz : float
            Displacement along z-direction [mm].
        init_pos : list, optional
            Initial position, if None the initial position is the last point
            of the waveguide. The default is None.
        radius : float, optional
            Radius for computing the displacement along x-direction [mm].
            The default is 20.
        disp_x : float, optional
            Length of the curve. The default is 0.
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.

        Raises
        ------
        ValueError
            Speed values has the wrong shape.

        Returns
        -------
        None.

        """
        if init_pos is None:
            init_pos = self.lastpt

        dx, *_, l_curve = self.get_spline_parameter(init_pos, dy, 0, radius, disp_x)

        x1, y1, z1 = self._get_spline_points(dy / 2, dz, init_pos, radius,
                                             speed=speed,
                                             bc_y=((1, 0.0), (1, dy / dx)),
                                             bc_z=((1, 0.0), (1, 0.0)))
        init_pos2 = np.array([x1[-1], y1[-1], z1[-1]])
        x2, y2, z2 = self._get_spline_points(dy / 2, -dz, init_pos2, radius,
                                             speed=speed,
                                             bc_y=((1, dy / dx), (1, 0.0)),
                                             bc_z=((1, 0.0), (1, 0.0)))
        x = np.append(x1[:-1], x2)
        y = np.append(y1[:-1], y2)
        z = np.append(z1[:-1], z2)

        # Use CubicSpline as control point for higher order spline.
        us_y = InterpolatedUnivariateSpline(x, y, k=5)
        us_z = InterpolatedUnivariateSpline(x, z, k=5)

        # update coordinates or return
        self._x = np.append(self._x, x)
        self._y = np.append(self._y, us_y(x))
        self._z = np.append(self._z, us_z(x))

        # update speed array
        if np.size(speed) == 1:
            self._f = np.append(self._f, speed * np.ones(x.shape))
        elif np.size(speed) == np.size(x):
            self._f = np.append(self._f, speed)
        else:
            raise ValueError('Speed array is neither a single value nor ',
                             'array of appropriate size.')

        self._s = np.append(self._s, shutter * np.ones(x.shape))
        return self

    def curvature(self) -> np.ndarray:
        """
        CURVATURE.

        Compute the 3D point-to-point curvature radius of the waveguide
        shape.

        Returns
        -------
        curvature : numpy.ndarray
            Array of the curvature radii computed at each point of the curve.

        """

        (x, y, z, _, _) = self._unique_points().T

        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        dz_dt = np.gradient(z)

        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        d2z_dt2 = np.gradient(dz_dt)

        num = (dx_dt ** 2 + dy_dt ** 2 + dz_dt ** 2) ** 1.5
        den = np.sqrt((d2z_dt2 * dy_dt - d2y_dt2 * dz_dt) ** 2 +
                      (d2x_dt2 * dz_dt - d2z_dt2 * dx_dt) ** 2 +
                      (d2y_dt2 * dx_dt - d2x_dt2 * dy_dt) ** 2)
        default_zero = np.ones(np.size(num)) * np.inf
        # only divide nonzeros else Inf
        curvature = np.divide(num, den, out=default_zero, where=(den != 0))
        return curvature

    def cmd_rate(self) -> np.ndarray:
        """
        COMMAND RATE.

        Compute the point-to-point command rate of the waveguide shape.

        Returns
        -------
        cmd_rate : numpy.ndarray
            Array of the command rates computed at each point of the curve.

        """

        # exclude last point, it's there just to close the shutter
        (x, y, z, f, _) = self._unique_points()[:-1, :].T

        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        dz_dt = np.gradient(z)
        dt = np.sqrt(dx_dt ** 2 + dy_dt ** 2 + dz_dt ** 2)

        default_zero = np.zeros(np.size(dt))
        # only divide nonzeros else Inf
        cmd_rate = np.divide(f, dt, out=default_zero, where=(dt != 0))
        return np.mean(cmd_rate)

    @staticmethod
    def get_sbend_parameter(dy: float, radius: float) -> tuple:
        """
        GET S-BEND PARAMETERS.

        The function computes the final angle, and x-displacement for a
        circular S-bend given the y-displacement D and curvature radius.

        Parameters
        ----------
        dy : float
            Displacement along y-direction [mm].
        radius : float
            Curvature radius of the S-bend [mm]..

        Returns
        -------
        tuple
            (final angle, x-displacement), ([radians], [mm]).

        """
        dy = np.abs(dy / 2)
        a = np.arccos(1 - (dy / radius))
        dx = 2 * radius * np.sin(a)
        return a, dx

    @staticmethod
    def get_spline_parameter(init_pos: np.ndarray,
                             dy: float,
                             dz: float,
                             radius: float = 20,
                             disp_x: float = 0) -> tuple:
        """
        GET SPLINE PARAMETERS.

        The function computes the delta displacements along x-, y- and
        z-direction and the total lenght of the curve.

        Parameters
        ----------
        init_pos : list
            Initial position of the curve.
        dy : float
            Displacement along y-direction [mm].
        dz : float
            Displacement along z-direction [mm].
        radius : float, optional
            Curvature radius of the S-bend [mm]. The default is 20.
        disp_x : float, optional
            Displacement along x-direction [mm]. The default is 0.

        Returns
        -------
        tuple
            (deltax [mm], deltay [mm], deltaz [mm], curve length [mm]).

        """
        xl, yl, zl = init_pos
        final_pos = np.array([yl + dy, zl + dz])
        if disp_x != 0:
            final_pos = np.insert(final_pos, 0, xl + disp_x)
            pos_diff = np.subtract(final_pos, init_pos)
            l_curve = np.sqrt(np.sum(pos_diff ** 2))
        else:
            final_pos = np.insert(final_pos, 0, xl)
            pos_diff = np.subtract(final_pos, init_pos)
            ang = np.arccos(1 - np.sqrt(pos_diff[1] ** 2 + pos_diff[2] ** 2) /
                            (2 * radius))
            pos_diff[0] = 2 * radius * np.sin(ang)
            l_curve = 2 * ang * radius
        return pos_diff[0], pos_diff[1], pos_diff[2], l_curve

    # Private interface
    def _unique_points(self):
        """
        REMOVE ALL CONSECUTIVE DUPLICATES.

        At least one coordinate (X,Y,Z,F,S) have to change between two
        consecutive lines.

        Duplicates can be selected by crating a boolean index mask as follows:
            - make a row-wise diff operation (data.diff)
            - compute absolute value of all elements in order to work only
                with positive numbers
            - make a column-wise sum (.sum(axis=1))
        In this way consecutive duplicates correspond to a 0.0 value in the
        latter array.
        Converting this array to boolean (all non-zero values are True) the
        index mask can be retrieved.
        The first element is set to True by default since it is lost by the
        diff operation.

        Returns
        -------
        numpy.ndarray
            Coordinate matrix (x, y, z, f, s).

        """

        data = np.stack((self._x, self._y, self._z,
                         self._f, self._s), axis=-1).astype(np.float32)
        mask = np.diff(data, axis=0)
        mask = np.sum(np.abs(mask), axis=1, dtype=bool)
        mask = np.insert(mask, 0, True)
        return np.delete(data, np.where(mask is False), 0).astype(np.float32)

    def _get_spline_points(self,
                           dy: float,
                           dz: float,
                           init_pos: np.ndarray = None,
                           radius: float = 20,
                           disp_x: float = 0,
                           speed: float = 0,
                           bc_y: tuple = ((1, 0.0), (1, 0.0)),
                           bc_z: tuple = ((1, 0.0), (1, 0.0))) -> tuple:
        """
        GET SPLINE POINTS.

        Function for the generation of a 3D spline curve. Starting from
        init_point the function compute a 3D spline with a displacement dy
        in y-direction and dz in z-direction.
        The user can specify the length of the curve or (alternatively) provide
        a curvature radius that is used to compute the displacement along
        x-direction as the displacement of the equivalent circular S-bend.

        User can provide the boundary conditions for the derivatives in the
        y- and z-directions. Boundary conditions are a tuple of tuples in which
        we have:
            bc = ((initial point), (final point))
        where the (initial point) and (final point) tuples are specified as
        follows:
            (derivative order, value of derivative)
        the derivative order can be either 0, 1, 2.


        Parameters
        ----------
        dy : float
            y-displacement [mm].
        dz : float
            z-displacement [mm].
        init_pos : list, optional
            Initial position, if None the initial position is the last point
            of the waveguide. The default is None.
        radius : float, optional
            Radius for computing the displacement along x-direction [mm].
            The default is 20.
        disp_x : float, optional
            Length of the curve. The default is 0.
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        bc_y : tuple, optional
            Boundary conditions for the y-coordinates.
            The default is ((1, 0.0), (1, 0.0)).
        bc_z : tuple, optional
            Boundary conditions for the z-coordinates.
            The default is ((1, 0.0), (1, 0.0)).

        Returns
        -------
        numpy.ndarry
            x-coordinates of the spline curve.
        numpy.ndarry
            y-coordinates of the spline curve.
        numpy.ndarry
            z-coordinates of the spline curve.

        """
        xd, yd, zd, l_curve = self.get_spline_parameter(init_pos, dy, dz,
                                                        radius, disp_x)
        num = self._get_num(l_curve, speed)

        xcoord = np.linspace(0, xd, num)
        cs_y = CubicSpline((0.0, xd), (0.0, yd), bc_type=bc_y)
        cs_z = CubicSpline((0.0, xd), (0.0, zd), bc_type=bc_z)
        return (xcoord + init_pos[0],
                cs_y(xcoord) + init_pos[1],
                cs_z(xcoord) + init_pos[2])

    def _get_num(self, l_curve: float = 0, speed: float = 0) -> int:
        """
        GET NUM POINTS

        Utility function that, given the length of a segment and the
        fabrication speed, computes the number of points required to work at
        the maximum command rate (attribute of Waveguide object).

        Parameters
        ----------
        l_curve : float, optional
            Length of the waveguide segment. Units in [mm]. The default is 0.
        speed : float, optional
            Fabrication speed. Units in [mm/s]. The default is 0.

        Raises
        ------
        ValueError
            Speed is set too low.

        Returns
        -------
        int
            Number of subdivisions.

        """
        if speed < 1e-6:
            raise ValueError('Speed set to 0.0 mm/s. Check speed parameter.')

        dl = speed / self.c_max
        num = int(np.ceil(l_curve / dl))
        if num <= 1:
            print('I had to add use an higher instruction rate.\n')
            return 3
        return num

    def _compute_number_points(self):
        # TODO: write method that compute the optimal number of points given
        #       the max value of cmd rate
        pass


def _example():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    np.set_printoptions(formatter={'float': "\t{: 0.6f}".format})

    # Data
    pitch = 0.080
    int_dist = 0.007
    d_bend = 0.5 * (pitch - int_dist)
    increment = [4, 0, 0]

    # Calculations
    mzi = [Waveguide() for _ in range(2)]
    for index, wg in enumerate(mzi):
        [xi, yi, zi] = [-2, -pitch / 2 + index * pitch, 0.035]

        wg.start([xi, yi, zi])
        wg.linear(increment, speed=20)
        wg.sin_mzi((-1) ** index * d_bend, radius=15, speed=20)
        wg.spline_bridge((-1) ** index * 0.08, (-1) ** index * 0.015, speed=20)
        wg.sin_mzi((-1) ** (index + 1) * d_bend, radius=15, speed=20)
        wg.linear(increment, speed=20)
        wg.end()

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
