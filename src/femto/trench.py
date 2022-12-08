from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
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
    """
    Class representing a single trench block.
    """

    def __init__(self, block: geometry.Polygon, delta_floor: float = 0.001) -> None:

        self.block: geometry.Polygon = block
        self.delta_floor: float = delta_floor
        self.floor_length: float = 0.0
        self.wall_length: float = 0.0

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
        xx, yy = self.block.exterior.coords.xy
        return np.asarray(xx, dtype=np.float32), np.asarray(yy, dtype=np.float32)

    @property
    def xborder(self) -> npt.NDArray[np.float32]:
        x, _ = self.border
        return x

    @property
    def yborder(self) -> npt.NDArray[np.float32]:
        _, y = self.border
        return y

    @property
    def xmin(self) -> float:
        return float(self.block.bounds[0])

    @property
    def ymin(self) -> float:
        return float(self.block.bounds[1])

    @property
    def xmax(self) -> float:
        return float(self.block.bounds[2])

    @property
    def ymax(self) -> float:
        return float(self.block.bounds[3])

    @property
    def center(self) -> tuple[float, float]:
        """
        Baricenter of the trench block.
        Returns the (x, y) coordinates of the center ponit.

        :return: (x, y) coordinates of the block's center point
        :rtype: tuple(float, float)
        """
        return self.block.centroid.x, self.block.centroid.y

    def toolpath(self) -> Generator[npt.NDArray[np.float32], None, None]:
        """
        Generator of the inset paths of the trench block.

        First, the outer trench polygon obj is insert in the trench ``polygon_list``. While the list is not empty
        we can extract the outer polygon from the list and compute the ``inset_polygon`` and insert it back to the list.
        ``inset_polygon`` can be:
        ``Polygon`` obj
            The obj is appended to the ``inset_polygon`` list and the exterior (x, y) coordinates are yielded.
        ``MultiPolygon`` obj
            All the single ``Polygon`` objects composing the ``MultiPolygon`` are appended to the ``inset_polygon``
            list as ``Polygon`` objects and the exterior (x, y) coordinates are yielded.
        ``None``
            In this case, we cannot extract a ``inset_polygon`` from the ``Polygon`` obj extracted from the
            ``inset_polygon``. Nothing is appended to the ``polygon_list`` and its size is reduced.

        :return: (x, y) coordinates of the inset path.
        :rtype: Generator[numpy.ndarray]
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
        """
        Compute a buffer operation of shapely ``Polygon`` obj.

        :param shape: ``Polygon`` of the trench block
        :type shape: shapely.geometry.Polygon
        :param offset: offset of the buffered polygon
        :type offset: float
        :return: Buffered polygon
        :rtype: List[shapely.geometry.Polygon]

        .. note::
        The buffer operation returns a polygonal result. The new polygon is checked for validity using
        ``obj.is_valid`` in the sense of [#]_.

        For a reference, read the buffer operations `here
        <https://shapely.readthedocs.io/en/stable/manual.html#constructive-methods>`_
        .. [#] John R. Herring, Ed., “OpenGIS Implementation Specification for Geographic information - Simple feature
        access - Part 1: Common architecture,” Oct. 2006
        """
        if shape.is_valid or isinstance(shape, geometry.MultiPolygon):
            buff_polygon = shape.buffer(offset)
            if isinstance(buff_polygon, geometry.MultiPolygon):
                return [geometry.Polygon(subpol) for subpol in buff_polygon.geoms]
            return [geometry.Polygon(buff_polygon)]
        return [geometry.Polygon()]


@dataclass
class TrenchColumn:
    """
    Class representing a column of trenches.
    """

    x_center: float
    y_min: float
    y_max: float
    bridge: float = 0.026
    length: float = 1
    nboxz: int = 4
    z_off: float = 0.020
    h_box: float = 0.075
    base_folder: str = ''
    deltaz: float = 0.0015
    delta_floor: float = 0.001
    beam_waist: float = 0.004
    round_corner: float = 0.005
    u: list[float] | None = None
    speed: float = 4
    speed_closed: float = 5
    speed_pos: float = 0.5

    def __post_init__(self):
        self.CWD: Path = Path.cwd()
        self.trench_list: list[Trench] = []

    def __iter__(self) -> Iterator[Trench]:
        """
        Iterator that yields the single trench blocks of the column.

        :return: Iterator over the trench objects in trench column
        :rtype: Iterator
        """
        return iter(self.trench_list)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}'

    @property
    def adj_bridge(self) -> float:
        # adjust bridge size considering the size of the laser focus [mm]
        return self.bridge / 2 + self.beam_waist + self.round_corner

    @property
    def n_repeat(self) -> int:
        return int(math.ceil((self.h_box + self.z_off) / self.deltaz))

    @property
    def fabrication_time(self) -> float:
        l_tot = 0.0
        for trench in self.trench_list:
            l_tot += self.nboxz * (self.n_repeat * trench.wall_length + trench.floor_length)
        return l_tot / self.speed

    @property
    def rect(self) -> geometry.Polygon:
        """
        Getter for the rectangular box for the whole trench column. If the ``x_c``, ``y_min`` and ``y_max`` are set we
        create a rectangular polygon that will be used to create the single trench blocks.

        ::
            +-------+  -> y_max
            |       |
            |       |
            |       |
            +-------+  -> y_min
               x_c

        :return: Rectangular box centered in ``x_c`` and y-borders at ``y_min`` and ``y_max``.
        :rtype: shapely.geometry.box
        """
        if self.length is None:
            return geometry.Polygon()
        return geometry.box(self.x_center - self.length / 2, self.y_min, self.x_center + self.length / 2, self.y_max)

    def dig_from_waveguide(
        self,
        waveguides: list[Waveguide],
        remove: list[int] | None = None,
    ) -> None:
        """
        Compute the trench blocks from the waveguide of the optical circuit.
        To get the trench blocks, the waveguides are used as mold matrix for the trench_list. The waveguides are
        converted to ``LineString`` and buffered to be as large as the adjusted bridge width.

        Using polygon difference, the rectangle (minx, miny, maxx, maxy) = (x_c - l, y_min, x_c + l, y_max) is cut
        obtaining a ``MultiPolygon`` with all the trench blocks.

        All the blocks are treated individually. Each block is then buffered to obtain an outset polygon with rounded
        corners a Trench obj is created with the new polygon box and the trench_list are appended to the
        ``trench_list``.

        :param waveguides: List of the waveguides composing the optical circuit.
        :type waveguides: List[Waveguide]
        :param remove: List of trench to remove.
        :type remove: List[int]
        """

        if not all(isinstance(wg, Waveguide) for wg in waveguides):
            raise ValueError(
                f'All the input objects must be instances of Waveguide. Given ' f'{[type(wg) for wg in waveguides]}'
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
        """
        Compute the trench blocks from the waveguide of the optical circuit.
        To get the trench blocks, the waveguides are used as mold matrix for the trench_list. The waveguides are
        converted to ``LineString`` and buffered to be as large as the adjusted bridge width.

        Using polygon difference, the rectangle (minx, miny, maxx, maxy) = (x_c - l, y_min, x_c + l, y_max) is cut
        obtaining a ``MultiPolygon`` with all the trench blocks.

        All the blocks are treated individually. Each block is then buffered to obtain an outset polygon with rounded
        corners a Trench obj is created with the new polygon box and the trench_list are appended to the
        ``trench_list``.

        :param waveguides: List of the waveguides composing the optical circuit.
        :type waveguides: List[Waveguide]
        :param remove: List of trench to remove.
        :type remove: List[int]
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

        if remove is None:
            remove = []

        trench_block = self.rect
        for coords in coords_list:
            dilated = geometry.LineString(coords).buffer(self.adj_bridge, cap_style=1)
            trench_block = trench_block.difference(dilated)

        # if coordinates are empty or coordinates do not intersect the trench column rectangle box
        if almost_equal(trench_block, self.rect, tol=1e-8):
            print('No trench found intersecting waveguides with trench area.\n')
            return None

        for block in listcast(sorted(trench_block.geoms, key=Trench)):
            block = block.buffer(self.round_corner, resolution=256, cap_style=1)
            self.trench_list.append(Trench(self.normalize(block), self.delta_floor))

        for index in sorted(listcast(remove), reverse=True):
            del self.trench_list[index]

    @staticmethod
    def normalize(poly: geometry.Polygon) -> geometry.Polygon:
        """
        Normalize polygon

        The function standardize the input polygon. It set a given orientation and set a definite starting point for
        the inner and outer rings of the polygon.
        Finally, it returns a new Polygon object constructed with the new ordered sequence of points.

        Function taken from https://stackoverflow.com/a/63402916
        """

        def normalize_ring(ring):
            coords = ring.coords[:-1]
            start_index = min(range(len(coords)), key=coords.__getitem__)
            return coords[start_index:] + coords[:start_index]

        poly = geometry.polygon.orient(poly)
        normalized_exterior = normalize_ring(poly.exterior)
        normalized_interiors = list(map(normalize_ring, poly.interiors))
        return geometry.Polygon(normalized_exterior, normalized_interiors)


def main():
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
