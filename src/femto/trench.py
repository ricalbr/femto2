from __future__ import annotations

import dataclasses
import math
import pathlib
from typing import Generator
from typing import Iterator

import numpy as np
import numpy.typing as npt
from femto.helpers import almost_equal
from femto.helpers import dotdict
from femto.helpers import flatten
from femto.helpers import listcast
from femto.waveguide import Waveguide
from shapely import geometry


class Trench:
    """Class that represents a trench block and provides methods to compute the toolpath of the block."""

    def __init__(self, block: geometry.Polygon, delta_floor: float = 0.001) -> None:
        self.block: geometry.Polygon = block  #: Polygon shape of the trench.
        self.delta_floor: float = delta_floor  #: Offset distance between buffered polygons in the trench toolpath.
        # TODO: create properties for floor_length and wall_length and rename these with underscores
        self.floor_length: float = 0.0  #: Length of the floor path.
        self.wall_length: float = 0.0  #: Length of the wall path.

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trench):
            raise TypeError(f'Trying comparing Trench with {other.__class__.__name__}')
        return almost_equal(self.block, other.block)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Trench):
            raise TypeError(f'Trying comparing Trench with {other.__class__.__name__}')
        return bool(self.yborder[0] < other.yborder[0])

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Trench):
            raise TypeError(f'Trying comparing Trench with {other.__class__.__name__}')
        return bool(self.yborder[0] <= other.yborder[0])

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Trench):
            raise TypeError(f'Trying comparing Trench with {other.__class__.__name__}')
        return bool(self.yborder[0] > other.yborder[0])

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Trench):
            raise TypeError(f'Trying comparing Trench with {other.__class__.__name__}')
        return bool(self.yborder[0] >= other.yborder[0])

    @property
    def border(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Border of the trench.

        It returns the border of the block as a tuple of two numpy arrays, one for the `x` coordinates and one for
        the `y` coordinates.

        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray)
            `x` and `y`-coordinates arrays of the trench border.
        """
        xx, yy = self.block.exterior.coords.xy
        return np.asarray(xx, dtype=np.float32), np.asarray(yy, dtype=np.float32)

    @property
    def xborder(self) -> npt.NDArray[np.float32]:
        """`x`-coordinates of the trench border.

        Returns
        -------
        numpy.ndarray
            `x`-coordinates arrays of the trench border.
        """
        x, _ = self.border
        return x

    @property
    def yborder(self) -> npt.NDArray[np.float32]:
        """`y`-coordinates of the trench border.

        Returns
        -------
        numpy.ndarray
            `y`-coordinates arrays of the trench border.
        """
        _, y = self.border
        return y

    @property
    def xmin(self) -> float:
        """Minimum `x` value of the trench boundary.

        Returns
        -------
        float
            Minimum `x` value of the block border.
        """
        return float(self.block.bounds[0])

    @property
    def ymin(self) -> float:
        """Minimum `y` value of the trench boundary.

        Returns
        -------
        float
            Minimum `y` value of the block border.
        """
        return float(self.block.bounds[1])

    @property
    def xmax(self) -> float:
        """Maximum `x` value of the trench boundary.

        Returns
        -------
        float
            Maximum `x` value of the block border.
        """
        return float(self.block.bounds[2])

    @property
    def ymax(self) -> float:
        """Maximum `y` value of the trench boundary.

        Returns
        -------
        float
            Maximum `y` value of the block border.
        """
        return float(self.block.bounds[3])

    @property
    def center(self) -> tuple[float, float]:
        """Baricenter of the trench block.

        Returns
        -------
        tuple(float, float)
            `x` and `y` coordinates of the centroid of the block.
        """
        return self.block.centroid.x, self.block.centroid.y

    def toolpath(self) -> Generator[npt.NDArray[np.float32], None, None]:
        """Toolpath generator.

        The function takes a polygon.


        First, the outer border is added to the ``polygon_list``. The functions pops polygon objects from this list,
        buffers it, and yields the exterior coordinates of the buffered polygon.
        Before yielding, the new polygon is added to the list as the buffered inset will be computed in the next
        iteration. If the buffering operation returns polygons composed of different non-touching parts (`i.e.`
        ``Multipolygon``), each part is added to the list as a single ``Polygon`` object.
        If no inset can be computed from the starting polygon, no object is added to the list. The generator
        terminates when no more buffered polygons can be computed.

        Yields
        ------
        numpy.ndarray
            (`x`, `y`) coordinates of each of the buffered polygons.

        See Also
        --------
        geometry.Polygon : shapely polygon object.
        geometry.Multipolygon : collections of shapely polygon objects.
        """

        self.wall_length = self.block.length
        polygon_list = [self.block]

        while polygon_list:
            current_poly = polygon_list.pop(0)
            if current_poly:
                polygon_list.extend(self.buffer_polygon(current_poly, offset=-np.fabs(self.delta_floor)))
                self.floor_length += current_poly.length
                yield np.array(current_poly.exterior.coords).T

    @staticmethod
    def buffer_polygon(shape: geometry.Polygon, offset: float) -> list[geometry.Polygon]:
        """Buffer a polygon.

        It takes a polygon and returns a list of polygons that are offset by a given distance.

        Parameters
        ----------
        shape : geometry.Polygon
            Shape of the trench block to buffer.
        offset : float
            The offset to buffer the polygon by [mm].

        Returns
        -------
        list(geometry.Polygon)
            List of buffered polygons. If the buffered polygon is still a ``Polyon`` object the list contains just a
            single polygon. If the buffered polygon is ``MultiPolygon``, the list contais all the single ``Polygon``
            objects that compose the multipolygon. Finally, if the buffered polygon cannot be computed the list
            contains just the empty polygon ``Polygon()``.

        Notes
        -----
        The buffer operation returns a polygonal result. The new polygon is checked for validity using
        ``obj.is_valid`` in the sense of [#]_.

        For a reference, read the buffer operations `here
        <https://shapely.readthedocs.io/en/stable/manual.html#constructive-methods>`_

        .. [#] John R. Herring, Ed., “OpenGIS Implementation Specification for Geographic information - Simple
            feature access - Part 1: Common architecture,” Oct. 2006

        See Also
        --------
        geometry.Polygon.buffer : buffer operations on ``Polygon`` objects.
        geometry.Polygon : shapely polygon object.
        geometry.Multipolygon : collections of shapely polygon objects.
        """

        if shape.is_valid or isinstance(shape, geometry.MultiPolygon):
            buff_polygon = shape.buffer(offset)
            if isinstance(buff_polygon, geometry.MultiPolygon):
                return [geometry.Polygon(subpol) for subpol in buff_polygon.geoms]
            return [geometry.Polygon(buff_polygon)]
        return [geometry.Polygon()]


@dataclasses.dataclass
class TrenchColumn:
    """Class representing a column of isolation trenches."""

    x_center: float  #: Center of the trench blocks [mm].
    y_min: float  #: Minimum `y` coordinates of the trench blocks [mm].
    y_max: float  #: Maximum `y` coordinates of the trench blocks [mm].
    bridge: float = 0.026  #: Separation length between nearby trench blocks [mm].
    length: float = 1  #: Lenght of the trench along the `x` axis [mm].
    h_box: float = 0.075  #: Height of the single trench box [mm].
    nboxz: int = 4  #: Number of stacked box along the `z` axis.
    z_off: float = -0.020  #: Starting offset in `z` with respect to the sample's surface [mm].
    deltaz: float = 0.0015  #: Offset distance between countors paths of the trench wall [mm].
    delta_floor: float = 0.001  #: Offset distance between buffered polygons in the trench toolpath [mm].
    u: list[float] | None = None  #:
    speed: float = 4  #: Translation speed [mm/s].
    speed_closed: float = 5  #: Translation speed with closed shutter [mm/s].
    speed_pos: float = 0.5  #: Positioning speed with closed shutter [mm/s].
    base_folder: str = ''  #: Location where PGM files are stored in lab PC. If empty, load files with relative path.
    beam_waist: float = 0.004  #: Diameter of the laser beam-waist [mm].
    round_corner: float = 0.010  #: Radius of the blocks round corners [mm].

    def __post_init__(self):
        self.CWD: pathlib.Path = pathlib.Path.cwd()  #:
        self.trench_list: list[Trench] = []  #:

    def __iter__(self) -> Iterator[Trench]:
        """Iterator that yields single trench blocks of the column.

        Yields
        ------
        Trench
            Single trench block of the TrenchColumn.
        """
        return iter(self.trench_list)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}'

    @property
    def adj_bridge(self) -> float:
        """Bridge length adjusted for the laser beam waist.

        Returns
        -------
        float
            Adjustted bridge size considering the size of the laser focus [mm].
        """
        return self.bridge / 2 + self.beam_waist + self.round_corner

    @property
    def n_repeat(self) -> int:
        """Number of laser passes required to cover the vertical height of the trench box.

        Returns
        -------
        int
            The number of times the border path is repeated in the `z` direction.
        """
        return int(abs(math.ceil((self.h_box - self.z_off) / self.deltaz)))

    @property
    def fabrication_time(self) -> float:
        """Total fabrication time.

        The fabrication time is the sum of the lengths of all the walls and floors of all the trenches, divided by the
        translation speed.

        Returns
        -------
        float
            Total fabrication time [s].
        """
        l_tot = sum([self.nboxz * (self.n_repeat * t.wall_length + t.floor_length) for t in self.trench_list])
        return l_tot / self.speed

    @property
    def rect(self) -> geometry.Polygon:
        """Area of the trench column.

        The rectangular box is centered in ``x_c`` along the `x` axis, while the `y`-borders are ``y_min`` and
        ``y_max``. ::

            ┌─────┐  ► y_max
            │     │
            │     │
            │     │
            └─────┘  ► y_min
               ▲
               x_c

        Returns
        -------
        geometry.box
            Rectangular box polygon.
        """

        if self.length is None:
            return geometry.Polygon()
        return geometry.box(self.x_center - self.length / 2, self.y_min, self.x_center + self.length / 2, self.y_max)

    def dig_from_waveguide(
        self,
        waveguides: list[Waveguide],
        remove: list[int] | None = None,
    ) -> None:
        """Dig trenches from waveguide input.

        The function uses a list of ``Waveguide`` objects as a mold to define the trench shapes. It populates
        `self.trech_list` with ``Trench`` objects.
        If some of the generated trenches are not needed they can be removed from the list is a ``remove`` list of
        indeces is given as input. Trenches are numbered such that the one with lowest `y` coordinate has index 0,
        the one with second-lowest `y` coordinate has index 1 and so on. If ``remove`` is empty or ``None`` all the
        generated trenches are added to the `self.trench_list`.

        Parameters
        ----------
        waveguides : list(Waveguide)
            List of ``Waveguide`` objects that will be used as a mold to define trench shapes.
        remove : list[int], optional
            List of indides of trench to be removed from the ``TrenchColumn``.

        Returns
        -------
        None
        """

        if not all(isinstance(wg, Waveguide) for wg in waveguides):
            raise ValueError(
                f'All the input objects must be of type Waveguide.\nGiven {[type(wg) for wg in waveguides]}'
            )

        coords = []
        for wg in waveguides:
            x, y = wg.path
            coords.extend([list(zip(x, y))])
        self._dig(coords, remove)

    def dig_from_array(
        self,
        waveguides: list[npt.NDArray[np.float32]],
        remove: list[int] | None = None,
    ) -> None:
        """Dig trenches from array-like input.

        The function uses a list of `array-like` objects as a mold to define the trench shapes. It populates
        `self.trech_list` with ``Trench`` objects.
        If some of the generated trenches are not needed they can be removed from the list is a ``remove`` list of
        indeces is given as input. Trenches are numbered such that the one with lowest `y` coordinate has index 0,
        the one with second-lowest `y` coordinate has index 1 and so on. If ``remove`` is empty or ``None`` all the
        generated trenches are added to the `self.trench_list`.

        Parameters
        ----------
        waveguides : list(numpy.ndarray)
            List of ``numpy.ndarray`` objects that will be used as a mold to define trench shapes.
        remove : list[int], optional
            List of indides of trench to be removed from the ``TrenchColumn``.

        Returns
        -------
        None
        """
        if not all(isinstance(wg, np.ndarray) for wg in waveguides):
            raise ValueError(f'All the input objects must be numpy arrays. Given {[type(wg) for wg in waveguides]}')

        coords = []
        for wg in waveguides:
            x, y = wg.T if wg.shape[1] == 2 else wg
            coords.extend([list(zip(x, y))])
        self._dig(coords, remove)

    def _dig(
        self,
        coords_list: list[list[tuple[float, float]]],
        remove: list[int] | None = None,
    ) -> None:
        """Compute the trench blocks from the waveguide of the optical circuit.

        Trench blocks shapes are defined using a list of paths (``coords_list``) as mold matrix.
        The waveguides are converted to ``LineString`` and buffered to be as large as the adjusted bridge width.

        Using polygon difference operation, the rectangular area of the ``TrenchColumn`` is cut obtaining a
        ``MultiPolygon`` made of all the trench blocks.

        All the blocks are then treated individually. Each block is buffered to obtain an outset polygon with
        rounded corners a ``Trench`` object is created with the new polygon box and appended to the ``trench_list``,
        if their index is not present in the ``remove`` list.

        Parameters
        ----------
        coords_list : list(list(tuple(float, float)))
            List of ``numpy.ndarray`` objects that will be used as a mold to define trench shapes.
        remove : list[int], optional
            List of indides of trench to be removed from the ``TrenchColumn``.

        Returns
        -------
        None
        """
        if remove is None:
            remove = []

        trench_blocks = self.rect
        for coords in coords_list:
            dilated = geometry.LineString(coords).buffer(self.adj_bridge, cap_style=1)
            trench_blocks = trench_blocks.difference(dilated)

        # if coordinates are empty or coordinates do not intersect the trench column rectangle box
        if almost_equal(trench_blocks, self.rect, tol=1e-8):
            print('No trench found intersecting waveguides with trench area.\n')
            return None

        for block in listcast(sorted(trench_blocks.geoms, key=Trench)):
            # buffer to round corners
            block = block.buffer(self.round_corner, resolution=256, cap_style=1)
            # simplify the shape to avoid path too much dense of points
            block = block.simplify(tolerance=5e-7, preserve_topology=True)
            self.trench_list.append(Trench(self.normalize(block), self.delta_floor))

        for index in sorted(listcast(remove), reverse=True):
            del self.trench_list[index]

    @staticmethod
    def normalize(poly: geometry.Polygon) -> geometry.Polygon:
        """Normalize polygon.

        The function standardize the input polygon. It set a given orientation and set a definite starting point for
        the inner and outer rings of the polygon.

        Parameters
        ----------
        poly: geometry.Polygon
            Input ``Polygon`` object.

        Returns
        -------
        geometry.Polygon
            New ``Polygon`` object constructed with the new ordered sequence of points.

        See Also
        --------
        `This <https://stackoverflow.com/a/63402916>`_ stackoverflow answer.
        """

        def normalize_ring(ring: geometry.polygon.LinearRing):
            """Normalize ring
            It takes the exterior ring (a list of coordinates) of a ``Polygon`` object and returns the same ring,
            but with the sorted coordinates.


            Parameters
            ----------
            ring : geometry.LinearRing
                List of coordinates of a ``Polygon`` object.

            Returns
            -------
                The coordinates of the ring, sorted from the minimum value to the maximum.

            See Also
            --------
            shapely.geometry.LinearRing : ordered sequence of (x, y[, z]) point tuples.
            """
            coords = ring.coords[:-1]
            start_index = min(range(len(coords)), key=coords.__getitem__)
            return coords[start_index:] + coords[:start_index]

        poly = geometry.polygon.orient(poly)
        normalized_exterior = normalize_ring(poly.exterior)
        normalized_interiors = list(map(normalize_ring, poly.interiors))
        return geometry.Polygon(normalized_exterior, normalized_interiors)


def main() -> None:
    # Data
    PARAM_WG = dotdict(speed=20, radius=25, pitch=0.080, int_dist=0.007, samplesize=(25, 3))
    PARAM_TC = dotdict(length=1.0, base_folder='', y_min=-0.1, y_max=19 * PARAM_WG['pitch'] + 0.1, u=[30.339, 32.825])

    # Calculations
    x_c = 0
    coup = [Waveguide(**PARAM_WG) for _ in range(20)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch, 0.035])
        wg.sin_coupler((-1) ** i * wg.dy_bend)
        x_c = wg.x[-1]
        wg.sin_coupler((-1) ** i * wg.dy_bend)
        wg.end()

    # Trench
    T = TrenchColumn(x_center=x_c, **PARAM_TC)
    T.dig_from_waveguide(flatten([coup]))


if __name__ == '__main__':
    main()
