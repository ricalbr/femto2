import warnings
from typing import Generator, Iterator, List

import numpy as np
import shapely.geometry
from descartes import PolygonPatch

from femto.Parameters import TrenchParameters
from femto.Waveguide import Waveguide

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from shapely.geometry import LineString, Polygon, box, polygon


class Trench:
    """
    Class representing a single trench block.
    """

    def __init__(self,
                 block: Polygon,
                 delta_floor: float = 0.001):

        self.block = block
        self.delta_floor = delta_floor
        self.floor_length = 0.0
        self.wall_length = 0.0

    @property
    def center(self) -> np.ndarray:
        """
        Returns the (x, y) coordinates of the trench block baricenter point.

        :return: (x, y) coordinates of the block's center point
        :rtype: np.ndarray
        """
        return np.asarray([self.block.centroid.x, self.block.centroid.y])

    @property
    def patch(self, fc: str = 'k', ec: str = 'k', alpha: float = 1, zorder: int = 1) -> PolygonPatch:
        """
        Return a Patch obj representing the trench block for plotting.

        :param fc:  Patch face colour. Can be specified with HEX code, e.g. '#9a9a9a'
        :type fc: str
        :param ec: Patch edge colour. Can be specified with HEX code, e.g. '#9a9a9a'
        :type ec: str
        :param alpha: Patch transparency coefficient. Value should be between 0 and 1.
        :type alpha: float
        :param zorder: Overlapping order in 2D plot. Higher values are on top.
        :type zorder: int
        :return: Patch obj for plotting the trench block
        :rtype: descartes.PolygonPatch
        """
        return PolygonPatch(self.block, fc=fc, ec=ec, alpha=alpha, zorder=zorder)

    def trench_paths(self) -> Generator[np.ndarray, None, None]:
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
        polygon_list = [self.block]
        self.wall_length = self.block.length
        while polygon_list:
            current_poly = polygon_list.pop(0)
            inset_polygon = self._buffer_polygon(current_poly)
            if inset_polygon and inset_polygon.type == 'MultiPolygon':
                polygon_list.extend(list(inset_polygon))
                for poly in list(inset_polygon):
                    self.floor_length += poly.length
                    yield np.array(poly.exterior.coords).T
            elif inset_polygon and inset_polygon.type == 'Polygon':
                self.floor_length += inset_polygon.length
                polygon_list.append(inset_polygon)
                yield np.array(inset_polygon.exterior.coords).T
            elif inset_polygon:
                raise ValueError(f'Trench block should be either Polygon or Multipolygon. Given {inset_polygon.type}')

    # Private interface
    def _buffer_polygon(self, shape: Polygon, inset: bool = True) -> shapely.geometry.shape:
        """
        Compute a buffer operation of shapely ``Polygon`` obj.

        :param shape: ``Polygon`` of the trench block
        :type shape: shapely.geometry.Polygon
        :param inset: If ``True`` the eroded polygon is computed (inset), if ``False`` the dilated polygon is
        computed (outset).
        :type inset: bool
        :return: Buffered polygon
        :rtype: shapely.geometry.shape

        .. note::
        The buffer operation returns a polygonal result. The new polygon is checked for validity using
        ``obj.is_valid`` in the sense of [#]_.

        For a reference, read the buffer operations `here
        <https://shapely.readthedocs.io/en/stable/manual.html#constructive-methods>`_
        .. [#] John R. Herring, Ed., “OpenGIS Implementation Specification for Geographic information - Simple feature
        access - Part 1: Common architecture,” Oct. 2006
        """
        if inset:
            new_polygon = shape.buffer(-self.delta_floor)
        else:
            new_polygon = shape.buffer(self.delta_floor)
        return new_polygon if new_polygon.is_valid else None


class TrenchColumn:
    """
    Class representing a column of trenches.
    """

    def __init__(self, param: TrenchParameters):
        self.param = param
        self._trench_list = []
        self._x_c = self.param.x_center
        self._y_min = self.param.y_min
        self._y_max = self.param.y_max

        self._rect = self._make_box()

    def __iter__(self) -> Iterator[Trench]:
        """
        Iterator that yields the single trench blocks of the column.

        :return: Trench obj of the trench column
        :rtype: Trench
        """
        return iter(self._trench_list)

    @property
    def x_c(self) -> float:
        """
        Getter for the x-coordinate of the trench column center.

        :return: center x-coordinate of the trench block
        :rtype: float
        """
        return self._x_c

    @x_c.setter
    def x_c(self, x_center: float):
        """
        Setter for the x-coordinate of the center of the trench column.

        :param x_center: center x-coordinate of the trench block
        :type x_center: float
        """
        self._x_c = x_center
        if x_center is not None:
            self._rect = self._make_box()

    @property
    def y_min(self) -> float:
        """
        Getter for the lower y-coordinate of the trench column.

        :return: y-coordinate of the lower border of trench column rectangle.
        :rtype: float
        """
        return self._y_min

    @y_min.setter
    def y_min(self, y_min: float):
        """
        Setter for the y-coordinate of the lower border of the trench column.

        :param y_min: y-coordinate of the trench block lower border
        :type y_min: float
        """
        self._y_min = y_min
        if y_min is not None:
            self._rect = self._make_box()

    @property
    def y_max(self) -> float:
        """
        Getter for the upper y-coordinate of the trench column.

        :return: y-coordinate of the upper border of trench column rectangle.
        :rtype: float
        """
        return self._y_max

    @y_max.setter
    def y_max(self, y_max):
        """
        Setter for the y-coordinate of the upper border of the trench column.

        :param y_max: y-coordinate of the trench block upper border
        :type y_max: float
        """
        self._y_max = y_max
        if y_max is not None:
            self._rect = self._make_box()

    @property
    def twriting(self):
        l_tot = 0.0
        for trench in self._trench_list:
            l_tot += self.param.nboxz * (self.param.n_repeat * trench.wall_length + trench.floor_length)
        return l_tot / self.param.speed

    def get_trench(self, waveguides: List[Waveguide]):
        """
        Compute the trench blocks from the waveguide of the optical circuit.
        To get the trench blocks, the waveguides are used as mold matrix for the trenches. The waveguides are
        converted to ``LineString`` and buffered to be as large as the adjusted bridge width.

        Using polygon difference, the rectangle (minx, miny, maxx, maxy) = (x_c - l, y_min, x_c + l, y_max) is cut
        obtaining a ``MultiPolygon`` with all the trench blocks.

        All the blocks are treated individually. Each block is then buffered to obtain an outset polygon with rounded
        corners a Trench obj is created with the new polygon box and the trenches are appended to the
        ``trench_list``.

        :param waveguides: List of the waveguides composing the optical circuit.
        :type waveguides: List[Waveguide]
        """
        if not all([isinstance(wg, Waveguide) for wg in waveguides]):
            raise TypeError('Elements circuit list must be of type Waveguide.')

        for wg in waveguides:
            x, y = wg.x[:-2], wg.y[:-2]
            dilated = (LineString(list(zip(x, y))).buffer(self.param.adj_bridge, cap_style=1))
            self._rect = self._rect.difference(dilated)

        for block in list(self._rect):
            block = (polygon.orient(block).buffer(self.param.round_corner, resolution=250, cap_style=1))
            trench = Trench(block, self.param.delta_floor)
            self._trench_list.append(trench)

    # Private interface
    def _make_box(self) -> shapely.geometry.box:
        """
        Create the rectangular box for the whole trench column. If the ``x_c``, ``y_min`` and ``y_max`` are set we
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
        if self._x_c is not None and self._y_min is not None and self._y_max is not None:
            return box(self._x_c - self.param.length / 2, self._y_min,
                       self._x_c + self.param.length / 2, self._y_max)
        else:
            return None


def _example():
    import matplotlib.pyplot as plt

    # Data
    x_mid = None

    PARAMETERS_WG = dict(
        scan=6,
        speed=20,
        radius=15,
        pitch=0.080,
        int_dist=0.007,
    )

    PARAMETERS_TC = TrenchParameters(
        lenght=1.0,
        nboxz=4,
        deltaz=0.0015,
        h_box=0.075,
        base_folder=r'C:\Users\Capable\Desktop\RiccardoA',
        y_min=-0.1,
        y_max=19 * PARAMETERS_WG['pitch'] + 0.1
    )

    # Calculations
    coup = [Waveguide(PARAMETERS_WG) for _ in range(20)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch, 0.035]).sin_acc((-1) ** i * wg.dy_bend)
        x_mid = wg.x[-1]
        wg.sin_acc((-1) ** i * wg.dy_bend).end()

    PARAMETERS_TC.x_center = x_mid
    trench_col = TrenchColumn(PARAMETERS_TC)
    trench_col.get_trench(coup)

    fig, ax = plt.subplots()
    for wg in coup:
        ax.plot(wg.x[:-1], wg.y[:-1], 'b')
    for t in trench_col:
        ax.add_patch(t.patch)
    ax.set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    _example()
