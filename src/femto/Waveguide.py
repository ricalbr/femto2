from dataclasses import dataclass
from functools import partialmethod
from typing import Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline

from femto.helpers import Dotdict
from femto.LaserPath import LaserPath

# Create a generic variable that can be 'Waveguide', or any subclass.
WG = TypeVar("WG", bound="Waveguide")


@dataclass
class Waveguide(LaserPath):
    """
    Class representing an optical waveguide.
    """

    depth: float = 0.035
    radius: float = 15
    pitch: float = 0.080
    pitch_fa: float = 0.127
    int_dist: Optional[float] = None
    int_length: float = 0.0
    arm_length: float = 0.0
    ltrench: float = 1.0
    dz_bridge: float = 0.007
    margin: float = 1.0

    def __post_init__(self: WG):
        super().__post_init__()
        if self.z_init is None:
            self.z_init = self.depth

    def __repr__(self: WG) -> str:
        return f"{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}"

    @property
    def dy_bend(self: WG) -> Optional[float]:
        if self.pitch is None:
            raise ValueError("Waveguide pitch is set to None.")
        if self.int_dist is None:
            raise ValueError("Interaction distance is set to None.")
        return 0.5 * (self.pitch - self.int_dist)

    @property
    def dx_bend(self: WG) -> float:
        if self.radius is None:
            raise ValueError("Curvature radius is set to None.")
        return self.sbend_length(self.dy_bend, self.radius)

    @property
    def dx_acc(self: WG) -> Optional[float]:
        if self.dx_bend is None or self.int_length is None:
            return None
        return 2 * self.dx_bend + self.int_length

    @property
    def dx_mzi(self: WG) -> Optional[float]:
        if self.dx_bend is None or self.int_length is None or self.arm_length is None:
            return None
        return 4 * self.dx_bend + 2 * self.int_length + self.arm_length

    @staticmethod
    def get_sbend_parameter(dy: Optional[float], radius: Optional[float]) -> Tuple[float, float]:
        """
        Computes the final rotation_angle, and x-displacement for a circular S-bend given the y-displacement dy and
        curvature
        radius.

        :param dy: Displacement along y-direction [mm].
        :type dy: float
        :param radius: Curvature radius of the S-bend [mm].
        :type radius: float
        :return: (final rotation_angle [radians], x-displacement [mm])
        :rtype: tuple
        """
        if radius is None or radius <= 0:
            raise ValueError(f"Radius should be a positive value. Given {radius}.")
        if dy is None:
            raise ValueError("dy is None. Give a valid input valid.")

        a = np.arccos(1 - (np.abs(dy / 2) / radius))
        dx = 2 * radius * np.sin(a)
        return a, dx

    def sbend_length(self: WG, dy: float, radius: float) -> float:
        """
        Computes the x-displacement for a circular S-bend given the y-displacement dy and curvature radius.

        :param dy: Displacement along y-direction [mm].
        :type dy: float
        :param radius: Curvature radius of the S-bend [mm].
        :type radius: float
        :return: x-displacement [mm]
        :rtype: float
        """
        return float(self.get_sbend_parameter(dy, radius)[1])

    def get_spline_parameter(
        self: WG,
        disp_x: Optional[float] = None,
        disp_y: Optional[float] = None,
        disp_z: Optional[float] = None,
        radius: float = 20,
    ) -> tuple:
        """
        Computes the displacements along x-, y- and z-direction and the total lenght of the curve.
        The disp_y and disp_z displacements are given by the user. The dx displacement can be known (and thus given as
        input) or unknown and it is computed using the get_sbend_parameter() method for the given radius.

        If disp_x, disp_y, disp_z are given they are returned unchanged and unsed to compute l_curve.
        On the other hand, if disp_x is None, it is computed using the get_sbend_parameters() method using the
        displacement 'disp_yz' along both y- and z-direction and the given radius.
        In this latter case, the l_curve is computed using the formula for the circular arc (radius * angle) which is
        multiply by a factor of 2 in order to retrieve the S-bend shape.

        :param disp_x: Displacement along x-direction [mm]. The default is None.
        :type disp_x: float
        :param disp_y: Displacement along y-direction [mm]. The default is None.
        :type disp_y: float
        :param disp_z: Displacement along z-direction [mm]. The default is None.
        :type disp_z: float
        :param radius: Curvature radius of the spline [mm]. The default is 20 mm.
        :type radius: float
        :return: (deltax [mm], deltay [mm], deltaz [mm], curve length [mm]).
        :rtype: Tuple[float, float, float, float]
        """
        if disp_y is None:
            raise ValueError("y-displacement is None. Give a valid disp_y")
        if disp_z is None:
            raise ValueError("z-displacement is None. Give a valid disp_z")

        if disp_x is None:
            disp_yz = np.sqrt(disp_y ** 2 + disp_z ** 2)
            ang, disp_x = self.get_sbend_parameter(disp_yz, radius)
            l_curve = 2 * ang * radius
        else:
            disp = np.array([disp_x, disp_y, disp_z])
            l_curve = np.sqrt(np.sum(disp ** 2))
        return disp_x, disp_y, disp_z, l_curve

    # Methods
    def circ(
        self: WG,
        initial_angle: float,
        final_angle: float,
        radius: Optional[float] = None,
        shutter: int = 1,
        speed: Optional[float] = None,
    ) -> WG:
        """
        Computes the points in the xy-plane that connects two angles (initial_angle and final_angle) with a circular
        arc of a given radius. The user can set the transition speed, the shutter state during the movement and the
        number of points of the arc.

        :param initial_angle: Starting rotation_angle of the circular arc [radians].
        :type initial_angle: float
        :param final_angle: Ending rotation_angle of the circular arc [radians].
        :type final_angle: float
        :param radius: Radius of the circular arc [mm]. The default is self.radius.
        :type radius: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: float
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: int
        :return: Self
        :rtype: Waveguide
        """
        if radius is None and self.radius is None:
            raise ValueError('Radius is None. Set Waveguide\'s "radius" attribute or give a radius as input.')
        r = radius if radius is not None else self.radius
        if r < 0:
            raise ValueError('Radius is negative. Set Waveguide\'s "radius" attribute or give a radius as input.')

        f = speed if speed is not None else self.speed
        if f is None:
            raise ValueError('Speed is None. Set Waveguide\'s "speed" attribute or give a speed as input.')

        delta_angle = final_angle - initial_angle
        num = self.subs_num(np.fabs(delta_angle * r), f)

        t = np.linspace(initial_angle, final_angle, num)
        x_circ = self._x[-1] + np.fabs(r) * (-np.cos(initial_angle) + np.cos(t))
        y_circ = self._y[-1] + np.fabs(r) * (-np.sin(initial_angle) + np.sin(t))
        z_circ = np.repeat(self._z[-1], num)
        f_circ = np.repeat(f, num)
        s_circ = np.repeat(shutter, num)

        # update coordinates
        self.add_path(x_circ, y_circ, z_circ, f_circ, s_circ)
        return self

    def arc_bend(
        self: WG,
        dy: float,
        radius: Optional[float] = None,
        shutter: int = 1,
        speed: Optional[float] = None,
    ) -> WG:
        """
        Concatenates two circular arc to make a circular S-bend.
        The user can specify the amplitude of the S-bend (height in the y direction) and the curvature radius.
        Starting and ending rotation_angle of the two arcs are computed automatically.

        The sign of dy encodes the direction of the S-bend:
            - dy > 0, upward S-bend
            - dy < 0, downward S-bend

        :param dy: Amplitude of the S-bend along the y direction [mm].
        :type dy: float
        :param radius: Curvature radius of the S-bend [mm]. The default is self.radius.
        :type radius: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :return: Self
        :rtype: Waveguide
        """
        r = radius if radius is not None else self.radius
        a, _ = self.get_sbend_parameter(dy, r)

        if dy > 0:
            self.circ(
                np.pi * (3 / 2),
                np.pi * (3 / 2) + a,
                radius=r,
                speed=speed,
                shutter=shutter,
            )
            self.circ(
                np.pi * (1 / 2) + a,
                np.pi * (1 / 2),
                radius=r,
                speed=speed,
                shutter=shutter,
            )
        else:
            self.circ(
                np.pi * (1 / 2),
                np.pi * (1 / 2) - a,
                radius=radius,
                speed=speed,
                shutter=shutter,
            )
            self.circ(
                np.pi * (3 / 2) - a,
                np.pi * (3 / 2),
                radius=radius,
                speed=speed,
                shutter=shutter,
            )
        return self

    def arc_acc(
        self: WG,
        dy: float,
        radius: Optional[float] = None,
        int_length: Optional[float] = None,
        shutter: int = 1,
        speed: Optional[float] = None,
    ) -> WG:
        """
        Concatenates two circular S-bend to make a single mode of a circular directional coupler.
        The user can specify the amplitude of the coupler (height in the y direction) and the curvature radius.

        The sign of dy encodes the direction of the coupler:
            - dy > 0, upward S-bend
            - dy < 0, downward S-bend

        :param dy: Amplitude of the S-bend along the y direction [mm].
        :type dy: float
        :param radius: Curvature radius of the S-bend [mm]. The default is self.radius.
        :type radius: float
        :param int_length: Length of the coupler straight arm [mm]. The default is self.int_length.
        :type int_length: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :return: Self
        :rtype: Waveguide
        """
        if int_length is None and self.int_length is None:
            raise ValueError(
                'Interaction length is None. Set Waveguide\'s "int_length" attribute or give a valid '
                "interaction length as input."
            )

        int_length = int_length if int_length is not None else self.int_length
        self.arc_bend(dy, radius=radius, speed=speed, shutter=shutter)
        self.linear([np.fabs(int_length), 0, 0], speed=speed, shutter=shutter, mode="INC")
        self.arc_bend(-dy, radius=radius, speed=speed, shutter=shutter)
        return self

    def arc_mzi(
        self: WG,
        dy: float,
        radius: Optional[float] = None,
        int_length: Optional[float] = None,
        arm_length: Optional[float] = None,
        shutter: int = 1,
        speed: Optional[float] = None,
    ) -> WG:
        """
        Concatenates two circular couplers to make a single mode of a circular MZI.
        The user can specify the amplitude of the coupler (height in the y direction) and the curvature radius.

        The sign of dy encodes the direction of the coupler:
            - dy > 0, upward S-bend
            - dy < 0, downward S-bend

        :param dy: Amplitude of the S-bend along the y direction [mm].
        :type dy: float
        :param radius: Curvature radius of the S-bend [mm]. The default is self.radius.
        :type radius: float
        :param int_length: Interaction distance of the MZI [mm]. The default is self.int_length.
        :type int_length: float
        :param arm_length: Length of the coupler straight arm [mm]. The default is self.arm_length.
        :type arm_length: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :return: Self
        :rtype: Waveguide
        """
        if arm_length is None and self.arm_length is None:
            raise ValueError(
                'Arm length is None. Set Waveguide\'s "arm_length" attribute or give a valid ' "arm length as " "input."
            )

        arm_length = arm_length if arm_length is not None else self.arm_length

        self.arc_acc(dy, radius=radius, int_length=int_length, speed=speed, shutter=shutter)
        self.linear([np.fabs(arm_length), 0, 0], speed=speed, shutter=shutter, mode="INC")
        self.arc_acc(dy, radius=radius, int_length=int_length, speed=speed, shutter=shutter)
        return self

    def sin_bridge(
        self: WG,
        dy: float,
        dz: Optional[float] = None,
        omega: Tuple[float, float] = (1, 2),
        radius: Optional[float] = None,
        shutter: int = 1,
        speed: Optional[float] = None,
    ) -> WG:
        """
        Computes the points in the xy-plane of a Sin-bend curve and the points in the xz-plane of a
        Sin-bridge of height Dz. The distance between the initial and final point is the same of the equivalent
        (circular) S-bend of given radius.
        The user can specify the amplitude of the Sin-bend (height in the y direction) and the curvature radius as
        well as the transition speed, the shutter state during the movement and the number of points of the arc.

        The sign of dy encodes the direction of the Sin-bend:
            - dy > 0, upward S-bend
            - dy < 0, downward S-bend

        The sign of dz encodes the direction of the Sin-bridge:
            - dz > 0, top bridge
            - dz < 0, under bridge

        .. notes
            The radius is an effective radius. The radius of curvature of the
            overall curve will be lower (in general) than the specified
            radius.

        :param dy: Amplitude of the Sin-bend along the y direction [mm].
        :type dy: float
        :param dz: Amplitude of the Sin-bend along the y direction [mm]. The default is self.dz_bridge.
        :type dz: float
        :param omega: Frequency of the Sin-bend oscillations for y- and z- coordinates. The deafult are fy = 1, fz = 2.
        :type omega: tuple
        :param radius: Curvature radius of the Sin-bend [mm]. The default is self.radius.
        :type radius: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :return: Self
        :rtype: Waveguide
        """

        if radius is None and self.radius is None:
            raise ValueError('Radius is None. Set Waveguide\'s "radius" attribute or give a radius as input.')
        if speed is None and self.speed is None:
            raise ValueError('Speed is None. Set Waveguide\'s "speed" attribute or give a speed as input.')
        if dz is None and self.dz_bridge is None:
            raise ValueError('dz bridge is None. Set Waveguide\'s "dz_bridge" attribute or give a valid dz as input.')
        if dy is None:
            raise ValueError("dy is None. Give a valid dy as input.")

        omega_y, omega_z = omega
        r = radius if radius is not None else self.radius
        f = speed if speed is not None else self.speed
        dzb = dz if dz is not None else self.dz_bridge

        _, dx = self.get_sbend_parameter(dy, r)
        num = self.subs_num(dx, f)

        x_sin = np.linspace(self._x[-1], self._x[-1] + dx, num)
        y_sin = self._y[-1] + 0.5 * dy * (1 - np.cos(omega_y * np.pi / dx * (x_sin - self._x[-1])))
        z_sin = np.repeat(self._z[-1], num) + 0.5 * dzb * (1 - np.cos(omega_z * np.pi / dx * (x_sin - self._x[-1])))
        f_sin = np.repeat(f, num)
        s_sin = np.repeat(shutter, num)

        # update coordinates
        self.add_path(x_sin, y_sin, z_sin, f_sin, s_sin)
        return self

    sin_bend = partialmethod(sin_bridge, dz=0.0)
    sin_comp = partialmethod(sin_bridge, dz=0.0, omega=(2.0, 2.0))

    def sin_acc(
        self: WG,
        dy: float,
        radius: Optional[float] = None,
        int_length: Optional[float] = 0.0,
        shutter: int = 1,
        speed: Optional[float] = None,
    ) -> WG:
        """
        Concatenates two Sin-bend to make a single mode of a sinusoidal directional coupler.
        The user can specify the amplitude of the coupler (height in the y direction) and the effective curvature
        radius.

        The sign of dy encodes the direction of the coupler:
            - dy > 0, upward Sin-bend
            - dy < 0, downward Sin-bend

        :param dy: Amplitude of the Sin-bend along the y direction [mm].
        :type dy: float
        :param radius: Curvature radius of the Sin-bend [mm]. The default is self.radius.
        :type radius: float
        :param int_length: Interaction distance of the MZI [mm]. The default is self.int_length.
        :type int_length: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :return: Self
        :rtype: Waveguide
        """
        if int_length is None and self.int_length is None:
            raise ValueError(
                "Interaction length is None."
                'Set Waveguide\'s "int_length" attribute or give a valid "int_length" as input.'
            )

        int_length = int_length if int_length is not None else self.int_length

        self.sin_bend(dy, radius=radius, speed=speed, shutter=shutter)
        self.linear([np.fabs(int_length), 0, 0], speed=speed, shutter=shutter)
        self.sin_bend(-dy, radius=radius, speed=speed, shutter=shutter)
        return self

    def sin_mzi(
        self: WG,
        dy: float,
        radius: Optional[float] = None,
        int_length: Optional[float] = None,
        arm_length: Optional[float] = None,
        shutter: int = 1,
        speed: Optional[float] = None,
    ) -> WG:
        """
        Concatenates two sinusoidal couplers to make a single mode of a sinusoidal MZI.
        The user can specify the amplitude of the coupler (height in the y direction) and the curvature radius.

        The sign of dy encodes the direction of the coupler:
            - dy > 0, upward S-bend
            - dy < 0, downward S-bend

        :param dy: Amplitude of the Sin-bend along the y direction [mm].
        :type dy: float
        :param radius: Curvature radius of the Sin-bend [mm]. The default is self.radius.
        :type radius: float
        :param int_length: Interaction distance of the MZI [mm]. The default is self.int_length.
        :type int_length: float
        :param arm_length: Length of the coupler straight arm [mm]. The default is self.arm_length.
        :type arm_length: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :return: Self
        :rtype: Waveguide
        """
        if arm_length is None and self.arm_length is None:
            raise ValueError(
                'Arm length is None. Set Waveguide\'s "arm_length" attribute or give a valid "arm_length" as input.'
            )

        arm_length = arm_length if arm_length is not None else self.arm_length

        self.sin_acc(dy, radius=radius, int_length=int_length, shutter=shutter, speed=speed)
        self.linear([np.fabs(arm_length), 0, 0], shutter=shutter, speed=speed)
        self.sin_acc(dy, radius=radius, int_length=int_length, shutter=shutter, speed=speed)
        return self

    def spline(
        self: WG,
        disp_x: Optional[float] = None,
        disp_y: Optional[float] = None,
        disp_z: Optional[float] = None,
        init_pos: Optional[npt.NDArray[np.float32]] = None,
        radius: Optional[float] = None,
        shutter: int = 1,
        speed: Optional[float] = None,
        bc_y: tuple = ((1, 0.0), (1, 0.0)),
        bc_z: tuple = ((1, 0.0), (1, 0.0)),
    ) -> WG:
        """
        Function wrapper. It computes the x,y,z coordinates of spline curve starting from init_pos with Dy and Dz
        displacements.
        See :func:`~femto.Waveguide._get_spline_points`.
        The points are then appended to the Waveguide coordinate list.

        :param disp_y: Amplitude of the spline along the y direction [mm].
        :type disp_y: float
        :param disp_z: Amplitude of the spline along the y direction [mm].
        :type disp_z: float
        :param init_pos: Initial position of the spline. The default is last point of the waveguide (self.lastpt).
        :type init_pos: np.ndarray
        :param radius: Curvature radius of the spline [mm]. The default is self.radius.
        :type radius: float
        :param disp_x: Displacement of the spline along the x direction [mm].
        :type disp_x: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :param bc_y:
        :type bc_y: tuple
        :param bc_z:
        :type bc_z: tuple
        :return: Self
        :rtype: Waveguide
        """

        f = speed if speed is not None else self.speed
        if f is None:
            raise ValueError('Speed is None. Set Waveguide\'s "speed" attribute or give a speed as input.')

        x_spl, y_spl, z_spl = self._get_spline_points(
            disp_x=disp_x, disp_y=disp_y, disp_z=disp_z, init_pos=init_pos, radius=radius, speed=f, bc_y=bc_y, bc_z=bc_z
        )
        f_spl = np.repeat(f, x_spl.size)
        s_spl = np.repeat(shutter, x_spl.size)

        # update coordinates or return
        self.add_path(x_spl, y_spl, z_spl, f_spl, s_spl)
        return self

    def spline_bridge(
        self: WG,
        disp_x: Optional[float] = None,
        disp_y: Optional[float] = None,
        disp_z: Optional[float] = None,
        init_pos: Optional[npt.NDArray[np.float32]] = None,
        radius: Optional[float] = None,
        shutter: int = 1,
        speed: Optional[float] = None,
    ) -> WG:
        """
        Computes a spline bridge as a sequence of two spline segments. disp_y is the total displacement along the
        y-direction of the bridge and disp_z is the height of the bridge along z.
        First, the function computes the dx-displacement of the planar spline curve with a disp_y displacement. This
        datum is used to compute the value of the first derivative along the y-coordinate for the costruction of the
        spline bridge such that:

            df(x, y)/dx = disp_y/dx

        in the peak point of the bridge.

        The cubic spline bridge obtained in this way has a first derivatives df(x, y)/dx, df(x, z)/dx which are zero
        (for construction) in the initial and final point.
        However, the second derivatives are not null in principle. To cope with this the spline points are fitted
        with a spline of the 5-th order. In this way the final curve has second derivatives close to zero (~1e-4)
        while maintaining the first derivative to zero.

        :param disp_x: Displacement of the spline along the x direction [mm].
        :type disp_x: float
        :param disp_y: Amplitude of the spline along the y direction [mm].
        :type disp_y: float
        :param disp_z: Amplitude of the spline along the y direction [mm].
        :type disp_z: float
        :param init_pos: Initial position of the spline. The new segments attaches to the last point of the waveguide by
        default.
        :type init_pos: np.ndarray
        :param radius: Curvature radius of the spline [mm]. The default is self.radius.
        :type radius: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :return: Self
        :rtype: Waveguide
        """

        f = speed if speed is not None else self.speed

        dx, *_, l_curve = self.get_spline_parameter(disp_x=disp_x, disp_y=disp_y, disp_z=0.0, radius=radius)
        # First half of the spline
        x1, y1, z1 = self._get_spline_points(
            disp_x=disp_x,
            disp_y=disp_y / 2,
            disp_z=disp_z,
            init_pos=init_pos,
            radius=radius,
            speed=f,
            bc_y=((1, 0.0), (1, disp_y / dx)),
            bc_z=((1, 0.0), (1, 0.0)),
        )
        # Second half of the spline
        init_pos2 = np.array([x1[-1], y1[-1], z1[-1]])
        x2, y2, z2 = self._get_spline_points(
            disp_x=disp_x,
            disp_y=disp_y / 2,
            disp_z=-disp_z,
            init_pos=init_pos2,
            radius=radius,
            speed=f,
            bc_y=((1, disp_y / dx), (1, 0.0)),
            bc_z=((1, 0.0), (1, 0.0)),
        )
        # Merge points
        x = np.append(x1[:-1], x2)
        y = np.append(y1[:-1], y2)
        z = np.append(z1[:-1], z2)

        # Construct a 5th-order spline using CubicSpline points as control points for interpolation
        y_uspline = InterpolatedUnivariateSpline(x, y, k=5)(x)
        z_uspline = InterpolatedUnivariateSpline(x, z, k=5)(x)
        f_uspline = np.repeat(f, x.size)
        s_uspline = np.repeat(shutter, x.size)

        self.add_path(x, y_uspline, z_uspline, f_uspline, s_uspline)
        return self

    # Private interface
    def _get_spline_points(
        self: WG,
        disp_x: Optional[float] = None,
        disp_y: Optional[float] = None,
        disp_z: Optional[float] = None,
        init_pos: Optional[npt.NDArray[np.float32]] = None,
        radius: float = 20,
        speed: Optional[float] = None,
        bc_y: tuple = ((1, 0.0), (1, 0.0)),
        bc_z: tuple = ((1, 0.0), (1, 0.0)),
    ) -> tuple:
        """
        Function for the generation of a 3D spline curve. Starting from init_point the function compute a 3D spline
        with a displacement dy in y-direction and dz in z-direction.
        The user can specify the length of the curve or (alternatively) provide a curvature radius that is used to
        compute the displacement along x-direction as the displacement of the equivalent circular S-bend.

        User can provide the boundary conditions for the derivatives in the y- and z-directions. Boundary conditions
        are a tuple of tuples in which we have:
            bc = ((initial point), (final point))
        where the (initial point) and (final point) tuples are specified as follows:
            (derivative order, value of derivative)
        the derivative order can be either 0, 1, 2.

        :param disp_y: Displacement along y-direction [mm].
        :type disp_y: float
        :param disp_z: Displacement along z-direction [mm].
        :type disp_z: float
        :param init_pos: Initial position of the curve.
        :type init_pos: np.ndarray
        :param radius: Curvature radius of the spline [mm]. The default is 20 mm.
        :type radius: float
        :param disp_x: Displacement along x-direction [mm]. The default is 0 mm.
        :type disp_x: float
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :param bc_y: Boundary conditions for the Y-coordinates. The default is ((1, 0.0), (1, 0.0)).
        :type bc_y: tuple
        :param bc_z: Boundary conditions for the z-coordinates. The default is ((1, 0.0), (1, 0.0)).
        :type bc_z: tuple
        :return: (x-coordinates, y-coordinates, z-coordinates) of the spline curve.
        :rtype: Tuple(np.ndarray, np.ndarray, np.ndarray)
        """
        if init_pos is None:
            if self._x.size == 0:
                if not any([self.x_init, self.y_init, self.z_init]):
                    raise ValueError(
                        'Initial position is None or non-valid. Set Waveguide\'s "x_init", "y_init" and "z_init"'
                        'attributes or give a valid "init_pos" as input or the current Waveguide is empty, '
                        "in that case use the start() method before attaching spline segments."
                    )
                else:
                    init_pos = np.array([self.x_init, self.y_init, self.z_init])
            else:
                init_pos = np.array([self._x[-1], self._y[-1], self._z[-1]])

        if (radius is None and self.radius is None) and disp_x is None:
            raise ValueError(
                'Radius is None. Set Waveguide\'s "radius" attribute or give a radius as input.'
                "Alternatively, give a valid x-displacement as input."
            )

        f = speed if speed is not None else self.speed
        if f is None:
            raise ValueError('Speed is None. Set Waveguide\'s "speed" attribute or give a speed as input.')

        r = radius if radius is not None else self.radius

        dx, dy, dz, l_curve = self.get_spline_parameter(disp_x=disp_x, disp_y=disp_y, disp_z=disp_z, radius=np.fabs(r))
        num = self.subs_num(l_curve, f)

        t = np.linspace(0, dx, num)
        x_cspline = init_pos[0] + t
        y_cspline = init_pos[1] + CubicSpline((0.0, dx), (0.0, dy), bc_type=bc_y)(t)
        z_cspline = init_pos[2] + CubicSpline((0.0, dx), (0.0, dz), bc_type=bc_z)(t)

        return x_cspline, y_cspline, z_cspline


def coupler(param: Union[dict, Dotdict]):

    mode1 = Waveguide(**param)
    lx = mode1.samplesize[0] / 2 - mode1.dx_bend

    mode1.start()
    mode1.linear([lx, None, None], mode="ABS")
    mode1.sin_acc(mode1.dy_bend)
    mode1.linear([mode1.x_end, None, None], mode="ABS")
    mode1.end()

    mode2 = Waveguide(**param)
    mode2.y_init = mode1.y_init + mode2.pitch
    mode2.start()
    mode1.linear([lx, None, None], mode="ABS")
    mode2.sin_acc(-mode2.dy_bend)
    mode2.linear([mode2.x_end, None, None], mode="ABS")
    mode2.end()

    return [mode1, mode2]


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Data
    PARAMETERS_WG = Dotdict(
        scan=6,
        speed=20,
        radius=15,
        pitch=0.080,
        int_dist=0.007,
        lsafe=3,
    )

    increment = [PARAMETERS_WG.lsafe, 0, 0]

    # Calculations
    mzi = []
    for index in range(2):
        PARAMETERS_WG.y_init = -PARAMETERS_WG.pitch / 2 + index * PARAMETERS_WG.pitch

        wg = Waveguide(**PARAMETERS_WG)
        wg.start()
        wg.linear(increment)
        wg.sin_mzi((-1) ** index * wg.dy_bend)
        wg.sin_bridge((-1) ** index * 0.08, (-1) ** index * 0.015)
        wg.sin_mzi((-1) ** (index + 1) * wg.dy_bend)
        wg.arc_mzi((-1) ** (index + 1) * wg.dy_bend)
        wg.linear(increment)
        wg.arc_bend((-1) ** (index + 1) * wg.dy_bend)
        wg.linear(increment)
        wg.end()
        mzi.append(wg)

    # Plot
    fig = plt.figure()
    fig.clf()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlabel("X [mm]"), ax.set_ylabel("Y [mm]"), ax.set_zlabel("Z [mm]")
    for wg in mzi:
        ax.plot(wg.x[:-1], wg.y[:-1], wg.z[:-1], "-k", linewidth=2.5)
        ax.plot(wg.x[-2:], wg.y[-2:], wg.z[-2:], ":b", linewidth=1.0)
    ax.set_box_aspect(aspect=(3, 1, 0.5))
    plt.show()

    print(f"Expected writing time {sum(wg.fabrication_time for wg in mzi):.3f} seconds")
    print(f"Laser path length {mzi[0].length:.3f} mm")


if __name__ == "__main__":
    main()
