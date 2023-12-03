from __future__ import annotations

from typing import Any
from typing import Callable

import attrs
import numpy as np
import numpy.typing as npt
from femto import logger
from femto.laserpath import LaserPath

# Define array type
nparray = npt.NDArray[np.float32]


@attrs.define(kw_only=True, repr=False)
class Waveguide(LaserPath):
    """Class that computes and stores the coordinates of an optical waveguide."""

    depth: float = 0.035  #: Distance for sample's bottom facet, `[mm]`.
    pitch: float = 0.080  #: Distance between adjacent modes, `[mm]`.
    pitch_fa: float = 0.127  #: Distance between fiber arrays' adjacent modes (for fan-in, fan-out), `[mm]`.
    int_dist: float | None = None  #: Directional Coupler's interaction distance, `[mm]`.
    int_length: float = 0.0  #: Directional Coupler's interaction length, `[mm]`.
    arm_length: float = 0.0  #: Mach-Zehnder interferometer's length of central arm, `[mm]`.
    dz_bridge: float = 0.007  #: Maximum `z`-height for 3D bridges, `[mm]`.
    ltrench: float = 0.0  #: Length of straight segment to accomodate trenches, `[mm]`.

    _id: str = attrs.field(alias='_id', default='WG')  #: Waveguide ID.

    def __init__(self, **kwargs):
        filtered = {att.name: kwargs[att.name] for att in self.__attrs_attrs__ if att.name in kwargs}
        self.__attrs_init__(**filtered)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.z_init is None:
            self.z_init = self.depth
            logger.debug(f'Set z_init to {self.z_init} mm.')

        # Adjust the pitch for the glass shrinking. Only change the external pitch in case of fan-in/out segments.
        if self.pitch == self.pitch_fa:
            logger.debug(f'Need to correct the pitch of the waveguides by a factor {self.shrink_correction_factor}.')
            self.pitch /= self.shrink_correction_factor
        self.pitch_fa /= self.shrink_correction_factor
        logger.debug(f'Correct the pitch of the fan-in/fan-out sections by a factor {self.shrink_correction_factor}.')

    @property
    def dy_bend(self) -> float:
        """`y`-displacement of an S-bend.

        Returns
        -------
        float
            The difference between the waveguide pitch and the interaction distance.
        """
        if self.pitch is None:
            logger.error('Waveguide pitch is set to None.')
            raise ValueError('Waveguide pitch is set to None.')
        if self.int_dist is None:
            logger.error('Interaction distance is set to None.')
            raise ValueError('Interaction distance is set to None.')
        dy = 0.5 * (self.pitch - self.int_dist)
        logger.debug(f'Return dy for a curve. dy = {dy}')
        return dy

    @property
    # Defining a function that calculates the length of a sbend.
    def dx_bend(self) -> float:
        """`x`-displacement of an S-bend.

        Returns
        -------
        float

        """

        dx = float(self.get_sbend_parameter(self.dy_bend, self.radius)[1])
        logger.debug(f'Return dx for a curve. dx = {dx}')
        return dx

    @property
    def dx_coupler(self) -> float:
        """`x`-displacement of a Directional Coupler.

        Returns
        -------
        float
            Sum of two `x`-displacements S-bend segments and the interaction distance straight segment.
        """
        dx = 2 * self.dx_bend + self.int_length
        logger.debug(f'Return dx for a coupler. dx = {dx}')
        return dx

    @property
    def dx_mzi(self) -> float:
        """`x`-displacement of a Mach-Zehnder Interferometer.

        Returns
        -------
        float
            Sum of two `x`-displacements Directional Coupler segments and the ``arm_length`` distance straight segment.
        """
        dx = 4 * self.dx_bend + 2 * self.int_length + self.arm_length
        logger.debug(f'Return dx for a MZI. dx = {dx}')
        return dx

    def bend(
        self,
        dy: float,
        dz: float,
        fx: Callable[..., tuple[nparray, nparray, nparray]],
        disp_x: float | None = None,
        radius: float | None = None,
        num_points: int | None = None,
        reverse: bool = False,
        speed: float | None = None,
        shutter: int = 1,
        **kwargs,
    ) -> Waveguide:
        """Bend segment.

        Add a bent segment to the current waveguide with a shape that is defined by the ``fx`` function. The ``fx``
        function is a ``Callable`` that can take arbitrary inputs but must return a tuple of three array-like objects
        representing the list of (`x`, `y`, z`)-coordinate of the path.

        Parameters
        ----------
        dy: float
            `y`-displacement of the waveguide of the S-bend [mm].
        dz: float
            `z`-displacement of the waveguide of the S-bend [mm].
        fx: Callable
            Custom function that returns a triple of (`x`, `y`, `z`) coordinates describing the profile of the S-bend.
        disp_x: float, optional
            `x`-displacement  for the sinusoidal bend. If the value is ``None`` (the default value),
            the `x`-displacement is computed with the formula for the circular `S`-bend.
        radius: float, optional
            Curvature radius [mm]. The default value is `self.radius`.
        num_points: int, optional
            Number of points of the S-bend. The default value is computed using `self.speed` and `self.cmd_rate_max`.
        reverse: bool, optional
            Flag to compute the bend's coordinate in reverse order.
        speed: float, optional
            Translation speed [mm/s]. The default value is `self.speed`.
        shutter: int, optional
            State of the shutter during the transition (0: 'OFF', 1: 'ON'). The default value is 1.
        kwargs: optional
            Additional arguments for the `fx` function.

        Returns
        -------
        The object itself.
        """
        if radius is None and self.radius is None:
            logger.error('Radius is None.')
            raise ValueError('Radius is None. Set Waveguide\'s "radius" attribute or give a radius as input.')
        if speed is None and self.speed is None:
            logger.error('Speed is None.')
            raise ValueError('Speed is None. Set Waveguide\'s "speed" attribute or give a speed as input.')
        if dz is None and self.dz_bridge is None:
            logger.error('dz bridge is None.')
            raise ValueError('dz bridge is None. Set Waveguide\'s "dz_bridge" attribute or give a valid dz as input.')
        if dy is None:
            logger.error('dy is None.')
            raise ValueError('dy is None. Give a valid dy as input.')

        r = radius if radius is not None else self.radius
        logger.debug(f'Radius set to r = {r}.')
        f = speed if speed is not None else self.speed
        logger.debug(f'Speed set to f = {f}.')
        dzb = dz if dz is not None else self.dz_bridge
        logger.debug(f'dz for the bridges set to dzb = {dzb}.')

        ang, tmp_x = self.get_sbend_parameter(np.sqrt(dy**2 + dz**2), r)
        dx = disp_x if disp_x is not None else tmp_x
        logger.debug(f'dx for the bend set to dx = {dx}.')

        num = num_points if num_points is not None else self.num_subdivisions(r * ang, f)
        logger.debug(f'Total number of points is num = {num}.')

        x, y, z = fx(dx=dx, dy=dy, dz=dzb, num_points=num, **dict(kwargs, radius=r))
        logger.debug(f'Computed (x, y, z) using {fx} function.')

        if reverse:
            x = np.flip(x) - x[-1]
            y = -(np.flip(y) - y[-1])
            z = np.flip(z) - z[-1]

        # update coordinates
        self.add_path(
            x + self._x[-1],
            y + self._y[-1],
            z + self._z[-1],
            np.repeat(f, len(x)),
            np.repeat(shutter, len(x)),
        )
        return self

    def coupler(
        self,
        dy: float,
        dz: float,
        fx: Callable[..., tuple[nparray, nparray, nparray]],
        int_length: float | None = None,
        disp_x: float | None = None,
        radius: float | None = None,
        num_points: int | None = None,
        reverse: bool = False,
        speed: float | None = None,
        shutter: int = 1,
        **kwargs,
    ) -> Waveguide:
        """Concatenate two bends to make a single mode of a Directional Coupler.

        The ``fx`` function describing the profile of the bend is repeated twice, make sure the joints connect smoothly.

        Parameters
        ----------
        dy: float
            `y`-displacement of the waveguide of the S-bend [mm].
        dz: float
            `z`-displacement of the waveguide of the S-bend [mm].
        fx: Callable
            Custom function that returns a triple of (`x`, `y`, `z`) coordinates describing the profile of the S-bend.
        int_length: float, optional
            Length of the Directional Coupler's straight interaction region [mm]. The default is `self.int_length`.
        disp_x: float, optional
            `x`-displacement  for the sinusoidal bend. If the value is ``None`` (the default value),
            the `x`-displacement is computed with the formula for the circular `S`-bend.
        radius: float, optional
            Curvature radius [mm]. The default value is `self.radius`.
        num_points: int, optional
            Number of points of the S-bend. The default value is computed using `self.speed` and `self.cmd_rate_max`.
        reverse: bool, optional
            Flag to compute the coupler's coordinate in reverse order.
        speed: float, optional
            Translation speed [mm/s]. The default value is `self.speed`.
        shutter: int
            State of the shutter during the transition (0: 'OFF', 1: 'ON'). The default value is 1.
        kwargs: optional
            Additional arguments for the `fx` function.

        Returns
        -------
        The object itself.
        """

        if int_length is None and self.int_length is None:
            logger.error('Interaction length is None.')
            raise ValueError(
                'Interaction length is None.'
                'Set Waveguide\'s "int_length" attribute or give a valid "int_length" as input.'
            )

        int_length = int_length if int_length is not None else self.int_length
        logger.debug(f'Interaction lenght set to int_length = {int_length}.')

        self.bend(
            dy=dy,
            dz=dz,
            fx=fx,
            disp_x=disp_x,
            radius=radius,
            num_points=num_points,
            reverse=reverse,
            speed=speed,
            **kwargs,
        )
        if reverse:
            self.linear([-np.fabs(int_length), 0, 0], speed=speed, shutter=shutter, mode='INC')
        else:
            self.linear([np.fabs(int_length), 0, 0], speed=speed, shutter=shutter)
        self.bend(
            dy=-dy,
            dz=-dz,
            fx=fx,
            disp_x=disp_x,
            radius=radius,
            num_points=num_points,
            reverse=reverse,
            speed=speed,
            **kwargs,
        )
        return self

    def mzi(
        self,
        dy: float,
        dz: float,
        fx: Callable[..., tuple[nparray, nparray, nparray]],
        int_length: float | None = None,
        arm_length: float | None = None,
        disp_x: float | None = None,
        radius: float | None = None,
        num_points: int | None = None,
        reverse: bool = False,
        speed: float | None = None,
        shutter: int = 1,
        **kwargs,
    ) -> Waveguide:
        """Concatenate two Directional Couplers segments to make a single mode of a Mach-Zehnder Interferometer.

        Parameters
        ----------
        dy: float
            `y`-displacement of the waveguide of the S-bend [mm].
        dz: float
            `z`-displacement of the waveguide of the S-bend [mm].
        fx: Callable
            Custom function that returns a triple of (`x`, `y`, `z`) coordinates describing the profile of the S-bend.
        int_length: float, optional
            Length of the Directional Coupler's straight interaction region [mm]. The default is `self.int_length`.
        arm_length: float, optional
            Length of the Mach-Zehnder Interferometer's straight arm [mm]. The default is `self.arm_length`.
        disp_x: float, optional
            `x`-displacement  for the bend. If the value is ``None`` (the default value), the `x`-displacement is
            computed with the formula for the circular `S`-bend.
        radius: float, optional
            Curvature radius [mm]. The default value is `self.radius`.
        num_points: int, optional
            Number of points of the S-bend. The default value is computed using `self.speed` and `self.cmd_rate_max`.
        reverse: bool, optional
            Flag to compute the MZI's coordinate in reverse order.
        speed: float, optional
            Translation speed [mm/s]. The default value is `self.speed`.
        shutter: int
            State of the shutter during the transition (0: 'OFF', 1: 'ON'). The default value is 1.
        kwargs: optional
            Additional arguments for the `fx` function.

        Returns
        -------
        The object itself.
        """

        if arm_length is None and self.arm_length is None:
            logger.error('Arm length is None.')
            raise ValueError(
                'Arm length is None. Set Waveguide\'s "arm_length" attribute or give a valid "arm_length" as input.'
            )

        arm_length = arm_length if arm_length is not None else self.arm_length
        logger.debug(f'Arm length set to arm_length = {arm_length}.')

        self.coupler(
            dy=dy,
            dz=dz,
            fx=fx,
            disp_x=disp_x,
            radius=radius,
            num_points=num_points,
            int_length=int_length,
            reverse=reverse,
            speed=speed,
            **kwargs,
        )
        if reverse:
            self.linear([-np.fabs(arm_length), 0, 0], shutter=shutter, speed=speed, mode='INC')
        else:
            self.linear([np.fabs(arm_length), 0, 0], shutter=shutter, speed=speed)
        self.coupler(
            dy=dy,
            dz=dz,
            fx=fx,
            disp_x=disp_x,
            radius=radius,
            num_points=num_points,
            int_length=int_length,
            reverse=reverse,
            speed=speed,
            **kwargs,
        )
        return self


@attrs.define(kw_only=True, repr=False)
class NasuWaveguide(Waveguide):
    """Class that computes and stores the coordinates of a Nasu optical waveguide [#]_.

    References
    ----------
    .. [#] `Nasu Waveguides <https://opg.optica.org/ol/abstract.cfm?uri=ol-30-7-723>`_ on Optics Letters.
    """

    adj_scan_shift: tuple[float, float, float] = (0, 0.0004, 0)  #: (`x`, `y`, `z`)-shifts between adjacent passes
    adj_scan: int = attrs.field(validator=attrs.validators.instance_of(int), default=5)  #: Number of adjacent scans.

    _id: str = attrs.field(alias='_id', default='NWG')  #: Nasu Waveguide ID.

    def __init__(self, **kwargs):
        filtered = {att.name: kwargs[att.name] for att in self.__attrs_attrs__ if att.name in kwargs}
        self.__attrs_init__(**filtered)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.z_init is None:
            self.z_init = self.depth

    @property
    def adj_scan_order(self) -> list[float]:
        """Scan order.

        Given the number of adjacent scans it computes the list of adjacent scans as:

        * if `self.adj_scan` is odd

            ``[0, 1, -1, 2, -2, ...]``

        * if `self.adj_scan` is even

            ``[0.5, -0.5, 1.5, -1.5, 2.5, -2.5, ...]``

        Returns
        -------
        list(float)
            Ordered list of adjacent scans.
        """

        adj_scan_list = []
        if self.adj_scan % 2:
            adj_scan_list.append(0.0)
            for i in range(1, self.adj_scan // 2 + 1):
                adj_scan_list.extend([i, -i])
        else:
            for i in range(0, self.adj_scan // 2):
                adj_scan_list.extend([i + 0.5, -i - 0.5])
        logger.debug(f'Computed adjacent scans, {adj_scan_list}.')
        return adj_scan_list


def coupler(
    param: dict[str, Any], f_profile: Callable[..., tuple[nparray, nparray, nparray]], nasu: bool = False
) -> list[Waveguide | NasuWaveguide]:
    """
    Directional coupler.

    Creates the two modes of a Directional Coupler. Waveguides can be standard multi-scan waveguides or Nasu
    Waveguides. The interaction region of the coupler is in the center of the sample.

    Parameters
    ----------
    param : dict
        Set of Waveguide parameters.
    f_profile: Callable
        Shape profile for the couplers.
    nasu : bool, optional
        Flag value for selecting Nasu Waveguide over standard multi-scan Waveguides. The default value is False.

    See Also
    --------
    femto.curves: collections of functions for bend segment profiles.

    Returns
    -------
    list(Waveguide) or list(NasuWaveguide)
        List of the two modes of the Directional Coupler.
    """

    mode1: Waveguide | NasuWaveguide
    mode2: Waveguide | NasuWaveguide
    mode1 = NasuWaveguide(**param) if nasu else Waveguide(**param)
    mode2 = NasuWaveguide(**param) if nasu else Waveguide(**param)

    lx = (mode1.samplesize[0] - mode1.dx_coupler) / 2
    logger.debug(f'Linear segment lx = {lx}.')

    logger.debug('Start mode 1.')
    mode1.start()
    mode1.linear([lx, None, None], mode='ABS')
    mode1.coupler(dy=mode1.dy_bend, dz=0.0, fx=f_profile)
    mode1.linear([mode1.x_end, None, None], mode='ABS')
    mode1.end()
    logger.debug('End mode 1.')

    logger.debug('Start mode 2.')
    mode2.y_init = mode1.y_init + mode2.pitch
    mode2.start()
    mode2.linear([lx, None, None], mode='ABS')
    mode2.coupler(dy=-mode2.dy_bend, dz=0.0, fx=f_profile)
    mode2.linear([mode2.x_end, None, None], mode='ABS')
    mode2.end()
    logger.debug('End mode 2.')

    return [mode1, mode2]


def main() -> None:
    """The main function of the script."""
    import matplotlib.pyplot as plt
    from curves import circ, sin
    from mpl_toolkits.mplot3d import Axes3D
    from addict import Dict as ddict

    # Data
    param_wg = ddict(scan=6, speed=20, radius=15, pitch=0.080, int_dist=0.007, lsafe=3, samplesize=(50, 3))

    increment = [5.0, 0, 0]

    # Calculations
    mzi = []
    for index in range(2):
        wg = Waveguide(**param_wg)
        wg.y_init = -wg.pitch / 2 + index * wg.pitch
        wg.start()
        wg.linear(increment)
        wg.coupler(dy=(-1) ** index * wg.dy_bend, dz=0.0, fx=sin, flat_peaks=0)
        wg.bend(dy=(-1) ** index * wg.dy_bend, dz=0.0, fx=circ)
        wg.coupler(dy=(-1) ** index * wg.dy_bend, dz=0.0, fx=circ, flat_peaks=0)
        wg.linear(increment)
        wg.end()
        mzi.append(wg)

    # Plot
    fig = plt.figure()
    fig.clf()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlabel('X [mm]'), ax.set_ylabel('Y [mm]'), ax.set_zlabel('Z [mm]')
    for wg in mzi:
        ax.plot(wg.x[:-1], wg.y[:-1], wg.z[:-1], '-k', linewidth=2.5)
        ax.plot(wg.x[-2:], wg.y[-2:], wg.z[-2:], ':b', linewidth=1.0)
    ax.set_box_aspect(aspect=(3, 1, 0.5))
    plt.show()

    print(f'Expected writing time {sum(wg.fabrication_time for wg in mzi):.3f} seconds')
    print(f'Laser path length {mzi[0].length:.3f} mm')


if __name__ == '__main__':
    main()
