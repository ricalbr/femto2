from __future__ import annotations

import copy
import math
import pathlib
from functools import cached_property
from typing import Any
from typing import Generator
from typing import Iterator
from typing import TypeVar

import attrs
import dill
import largestinteriorrectangle as lir
import numpy as np
import numpy.typing as npt
from femto import logger
from femto.curves import sin
from femto.helpers import almost_equal
from femto.helpers import dotdict
from femto.helpers import flatten
from femto.helpers import listcast
from femto.helpers import normalize_polygon
from femto.waveguide import Waveguide
from shapely import geometry
from shapely.ops import unary_union

TR = TypeVar('TR', bound='Trench')
TC = TypeVar('TC', bound='TrenchColumn')
nparray = npt.NDArray[np.float32]


class Trench:
    """Class that represents a trench block and provides methods to compute the toolpath of the block."""

    def __init__(
        self, block: geometry.Polygon, delta_floor: float = 0.001, height: float = 0.300, safe_inner_turns: int = 5
    ) -> None:
        self.block: geometry.Polygon = block  #: Polygon shape of the trench.
        self.delta_floor: float = delta_floor  #: Offset distance between buffered polygons in the trench toolpath.
        self.height: float = height  #: Depth of the trench box.
        self.safe_inner_turns: int = safe_inner_turns  #: Number of spiral turns befor zig-zag filling

        self._floor_length: float = 0.0  #: Length of the floor path.
        self._wall_length: float = 0.0  #: Length of the wall path.

        self._id: str = 'TR'  #: Trench identifier.

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trench):
            logger.error(f'Trying comparing Trench with {other.__class__.__name__}')
            raise TypeError(f'Trying comparing Trench with {other.__class__.__name__}')
        return almost_equal(self.block, other.block)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Trench):
            logger.error(f'Trying comparing Trench with {other.__class__.__name__}')
            raise TypeError(f'Trying comparing Trench with {other.__class__.__name__}')
        return bool(self.yborder[0] < other.yborder[0])

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Trench):
            logger.error(f'Trying comparing Trench with {other.__class__.__name__}')
            raise TypeError(f'Trying comparing Trench with {other.__class__.__name__}')
        return bool(self.yborder[0] <= other.yborder[0])

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Trench):
            logger.error(f'Trying comparing Trench with {other.__class__.__name__}')
            raise TypeError(f'Trying comparing Trench with {other.__class__.__name__}')
        return bool(self.yborder[0] > other.yborder[0])

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Trench):
            logger.error(f'Trying comparing Trench with {other.__class__.__name__}')
            raise TypeError(f'Trying comparing Trench with {other.__class__.__name__}')
        return bool(self.yborder[0] >= other.yborder[0])

    @property
    def id(self) -> str:
        """Object ID.

        The property returns the ID of a given object.

        Returns
        -------
        str
            The ID of the object.
        """
        return self._id

    @property
    def border(self) -> tuple[nparray, nparray]:
        """Border of the trench.

        It returns the border of the block as a tuple of two numpy arrays, one for the `x` coordinates and one for
        the `y` coordinates.

        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray)
            `x` and `y`-coordinates arrays of the trench border.
        """
        xx, yy = self.block.exterior.coords.xy
        logger.debug('Extracting (x,y) from trench block.')
        return np.asarray(xx, dtype=np.float32), np.asarray(yy, dtype=np.float32)

    @property
    def xborder(self) -> nparray:
        """`x`-coordinates of the trench border.

        Returns
        -------
        numpy.ndarray
            `x`-coordinates arrays of the trench border.
        """
        x, _ = self.border
        logger.debug('Return x-coordinate of the border.')
        return x

    @property
    def yborder(self) -> nparray:
        """`y`-coordinates of the trench border.

        Returns
        -------
        numpy.ndarray
            `y`-coordinates arrays of the trench border.
        """
        _, y = self.border
        logger.debug('Return y-coordinate of the border.')
        return y

    @property
    def xmin(self) -> float:
        """Minimum `x` value of the trench boundary.

        Returns
        -------
        float
            Minimum `x` value of the block border.
        """
        logger.debug('Return minimum x-value for trench block.')
        return float(self.block.bounds[0])

    @property
    def ymin(self) -> float:
        """Minimum `y` value of the trench boundary.

        Returns
        -------
        float
            Minimum `y` value of the block border.
        """
        logger.debug('Return minimum y-value for trench block.')
        return float(self.block.bounds[1])

    @property
    def xmax(self) -> float:
        """Maximum `x` value of the trench boundary.

        Returns
        -------
        float
            Maximum `x` value of the block border.
        """
        logger.debug('Return maximun x-value for trench block.')
        return float(self.block.bounds[2])

    @property
    def ymax(self) -> float:
        """Maximum `y` value of the trench boundary.

        Returns
        -------
        float
            Maximum `y` value of the block border.
        """
        logger.debug('Return maximun y-value for trench block.')
        return float(self.block.bounds[3])

    @property
    def center(self) -> tuple[float, float]:
        """Baricenter of the trench block.

        Returns
        -------
        tuple(float, float)
            `x` and `y` coordinates of the centroid of the block.
        """
        logger.debug('Return block center point.')
        return self.block.centroid.x, self.block.centroid.y

    @property
    def floor_length(self) -> float:
        """Total length of the floor path."""
        logger.debug(f'Total length of the floor path {self._floor_length} mm.')
        return self._floor_length

    @property
    def wall_length(self) -> float:
        """Length of a single layer of the wall path."""
        logger.debug(f'Total length of the wall path {self._wall_length} mm.')
        return self._wall_length

    @cached_property
    def orientation(self) -> str:
        """Orientation of the trench block."""
        (xmin, ymin, xmax, ymax) = self.block.bounds
        if (xmax - xmin) <= (ymax - ymin):
            logger.debug('The block orientation is vertical.')
            return 'v'
        else:
            logger.debug('The block orientation is horizontal.')
            return 'h'

    @cached_property
    def num_insets(self) -> int:
        """Number of spiral turns."""
        if self.block.contains(self.block.convex_hull.buffer(-0.01 * self.delta_floor)):
            logger.debug(f'The number of spiral turns is {self.safe_inner_turns}.')
            return self.safe_inner_turns
        else:
            # External rectangle
            (xmin_ext, ymin_ext, xmax_ext, ymax_ext) = self.block.bounds

            # Internal rectangle
            buffer_length = 2 * self.delta_floor
            p = np.array([[[x, y] for (x, y) in self.block.buffer(-buffer_length).exterior.coords]], np.float32) * 1e3
            xmin_int, ymin_int, dx_int, dy_int = lir.lir(p.astype(np.int32), np.int32) / 1e3
            xmax_int, ymax_int = xmin_int + dx_int, ymin_int + dy_int

            if self.orientation == 'h':
                d_upper = np.abs(ymax_ext - ymax_int)
                d_lower = np.abs(ymin_ext - ymin_int)
            else:
                d_upper = np.abs(xmax_ext - xmax_int)
                d_lower = np.abs(xmin_ext - xmin_int)

            # Distinguish concave / biconcave
            if d_upper <= buffer_length or d_lower <= buffer_length:
                n_turns = int((d_upper + d_lower) / self.delta_floor) + self.safe_inner_turns
            else:
                n_turns = int((d_upper + d_lower) / (2 * self.delta_floor)) + self.safe_inner_turns
            logger.debug(f'The number of spiral turns is {n_turns}.')
            return n_turns

    def zigzag_mask(self) -> geometry.MultiLineString:
        """Zig-zag mask.
        The function returns a Shapely geometry (MultiLineString, or more rarely, GeometryCollection) for a simple
        hatched rectangle. The spacing between the lines is given by ``self.delta_floor`` while the rectangle is the
        minimum rotated rectangle containing the trench block.

        The lines of the zig-zag pattern are along the longest dimension of the rectangular envelope of the trench
        block.

        Returns
        -------
        shapely.MultiLineString | shapely.GeometryCollection
            Collection of hatch lines (in case a hatch line intersects with the corner of the clipping
            rectangle, which produces a point along with the usual lines).
        """

        (xmin, ymin, xmax, ymax) = self.block.bounds
        number_of_lines = 2 + int((xmax - xmin) / self.delta_floor)

        coords = []
        logger.debug('Create rectangular zig-zag MultiString line pattern.')
        if self.orientation == 'v':
            # Vertical hatching
            for i in range(0, number_of_lines, 2):
                coords.extend([((xmin + i * self.delta_floor, ymin), (xmin + i * self.delta_floor, ymax))])
                coords.extend([((xmin + (i + 1) * self.delta_floor, ymax), (xmin + (i + 1) * self.delta_floor, ymin))])
        else:
            # Horizontal hatching
            for i in range(0, number_of_lines, 2):
                coords.extend([((xmin, ymin + i * self.delta_floor), (xmax, ymin + i * self.delta_floor))])
                coords.extend([((xmax, ymin + (i + 1) * self.delta_floor), (xmin, ymin + (i + 1) * self.delta_floor))])
        return geometry.MultiLineString(coords)

    def zigzag(self, poly: geometry.Polygon) -> nparray:
        """Zig-zag filling pattern.
        The function `zigzag` takes a polygon as input, applies a zig-zag filling pattern to it, and returns the
        coordinates of the resulting zigzag pattern.

        Parameters
        ----------
        poly: geometry.Polygon
            The parameter `poly` is of type `geometry.Polygon`. It represents a polygon object that you want to
            apply the zig-zag filling pattern to.

        Returns
        -------
        numpy.ndarray
            Coordinates array of the zig-zag filling pattern.
        """
        mask = self.zigzag_mask()
        path_collection = poly.intersection(mask)
        logger.debug('Intersect rectangular zig-zag line pattern with trench block.')

        coords = []
        for line in path_collection.geoms:
            self._floor_length += line.length + self.delta_floor
            coords.extend(line.coords)
        return np.array(coords).T

    def toolpath(self) -> Generator[nparray, None, None]:
        """Toolpath generator.

        The function takes a polygon and computes the filling toolpath.
        Such path is obtained with two strategy:
        -   First, the outer border is added to the ``polygon_list``. The functions pops polygon objects from this list,
            buffers it, and yields the exterior coordinates of the buffered polygon.
            Before yielding, the new polygon is added to the list as the buffered inset will be computed in the next
            iteration. If the buffering operation returns polygons composed of different non-touching parts (`i.e.`
            ``MultiPolygon``), each part is added to the list as a single ``Polygon`` object.
            If no inset can be computed from the starting polygon, no object is added to the list. The generator
            terminates when no more buffered polygons can be computed.

        -   Second, after a number of insets the center of the trench floor is filled with a zig-zag pattern to avoid
            harsh acceleration and small displacements.
            The zig-zag hatching pattern is computed for each polygon in the ``polygon_list``.

        Yields
        ------
        numpy.ndarray
            (`x`, `y`) coordinates of each of the buffered polygons.

        See Also
        --------
        geometry.Polygon : shapely polygon object.
        geometry.Multipolygon : collections of shapely polygon objects.
        """

        self._wall_length = self.block.length
        polygon_list = [self.block]

        for _ in range(self.num_insets):
            current_poly = polygon_list.pop(0)
            if not current_poly.is_empty:
                polygon_list.extend(self.buffer_polygon(current_poly, offset=-np.fabs(self.delta_floor)))
                self._floor_length += current_poly.length
                logger.debug('Yield buffered contour path.')
                yield np.array(current_poly.exterior.coords).T

        for poly in polygon_list:
            logger.debug('Yield inner zig-zag path.')
            yield self.zigzag(poly.buffer(1.05 * self.delta_floor))

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
            buff_polygon = shape.buffer(offset).simplify(tolerance=1e-5, preserve_topology=True)
            if isinstance(buff_polygon, geometry.MultiPolygon):
                return [geometry.Polygon(subpol) for subpol in buff_polygon.geoms]
            return [geometry.Polygon(buff_polygon)]
        return [geometry.Polygon()]


@attrs.define(repr=False)
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
    safe_inner_turns: int = 5  #: Number of spiral turns befor zig-zag filling
    u: list[float] | None = None  #: List of U coordinate to change irradiation power automatically [deg].
    speed_wall: float = 4.0  #: Translation speed of the wall section [mm/s].
    speed_floor: float | None = None  #: Translation speed of the floor section [mm/s].
    speed_closed: float = 5.0  #: Translation speed with closed shutter [mm/s].
    speed_pos: float = 2.0  #: Positioning speed with closed shutter [mm/s].
    base_folder: str = ''  #: Location where PGM files are stored in lab PC. If empty, load files with relative path.
    beam_waist: float = 0.004  #: Diameter of the laser beam-waist [mm].
    round_corner: float = 0.010  #: Radius of the blocks round corners [mm].

    _id: str = attrs.field(alias='_id', default='TC')  #: TrenchColumn ID.
    _trench_list: list[TR] = attrs.field(alias='_trench_list', factory=list)  #: List of trench objects.

    CWD: pathlib.Path = attrs.field(default=pathlib.Path.cwd())  #: Current working directory

    def __init__(self, **kwargs):
        filtered = {att.name: kwargs[att.name] for att in self.__attrs_attrs__ if att.name in kwargs}
        self.__attrs_init__(**filtered)

    def __attrs_post_init__(self) -> None:
        if self.speed_floor is None:
            self.speed_floor = self.speed_wall
            logger.debug(f'Floor speed is set to {self.speed_floor} mm/s.')

    def __iter__(self) -> Iterator[TR]:
        """Iterator that yields single trench blocks of the column.

        Yields
        ------
        Trench
            Single trench block of the TrenchColumn.
        """
        return iter(self._trench_list)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}'

    @classmethod
    def from_dict(cls: type(TC), param: dict[str, Any], **kwargs) -> TC:
        """Create an instance of the class from a dictionary.

        It takes a class and a dictionary, and returns an instance of the class with the dictionary's keys as the
        instance's attributes.

        Parameters
        ----------
        param: dict()
            Dictionary mapping values to class attributes.
        kwargs: optional
            Series of keyword arguments that will be used to update the param file before the instantiation of the
            class.

        Returns
        -------
        Instance of class
        """
        # Update parameters with kwargs
        p = copy.deepcopy(param)
        p.update(kwargs)

        logger.debug(f'Create {cls.__name__} object from dictionary.')
        return cls(**p)

    @classmethod
    def load(cls: type(TC), pickle_file: str) -> TC:
        """Create an instance of the class from a pickle file.

        The load function takes a class and a pickle file name, and returns an instance of the class with the
        dictionary's keys as the instance's attributes.

        Parameters
        ----------
        pickle_file: str
            Filename of the pickle_file.

        Returns
        -------
        Instance of class
        """

        logger.info(f'Load {cls.__name__} object from pickle file.')
        with open(pickle_file, 'rb') as f:
            tmp = dill.load(f)
            logger.debug(f'{f} file loaded.')
        return cls.from_dict(tmp)

    @property
    def id(self) -> str:
        """Object ID.

        The property returns the ID of a given object.

        Returns
        -------
        str
            The ID of the object.
        """
        return self._id

    @property
    def trench_list(self) -> list[TR]:
        """List of Trench objects.

        Returns
        -------
        list[Trench]
            A list of trenches.
        """
        return self._trench_list

    @property
    def adj_bridge(self) -> float:
        """Bridge length adjusted for the laser beam waist.

        Returns
        -------
        float
            Adjustted bridge size considering the size of the laser focus [mm].
        """
        adj_b = self.bridge / 2 + self.beam_waist + self.round_corner
        logger.debug(f'The beidge size adjusted for the beam size is {adj_b} mm.')
        return adj_b

    @property
    def n_repeat(self) -> int:
        """Number of laser passes required to cover the vertical height of the trench box.

        Returns
        -------
        int
            The number of times the border path is repeated in the `z` direction.
        """
        n_repeat = int(abs(math.ceil((self.h_box - self.z_off) / self.deltaz)))
        logger.debug(f'The number of laser vertical trench layers is {n_repeat}.')
        return n_repeat

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
        fab_time = sum(
            [
                self.nboxz * (self.n_repeat * t.wall_length / self.speed_wall + t.floor_length / self.speed_floor)
                for t in self._trench_list
            ]
        )
        logger.debug(f'The total fabrication time for the trench column is {fab_time} s.')
        return fab_time

    @property
    def total_height(self) -> float:
        """Total trench height.

        Returns
        -------
        float
            Total trench height [um].
        """
        h_tot = float(self.nboxz * self.h_box)
        logger.debug(f'The total height of the trench block is {h_tot} mm.')
        return h_tot

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
            logger.debug('The length is None. Area is empty.')
            return geometry.Polygon()
        logger.debug(f'Return rectangle of sides {self.y_max - self.y_min} mm and {self.length} mm.')
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
            logger.debug(f'All the input objects must be of type Waveguide.\nGiven {[type(wg) for wg in waveguides]}')
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
        waveguides: list[nparray],
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
            logger.debug(f'All the input objects must be numpy arrays. Given {[type(wg) for wg in waveguides]}')
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
            logger.critical('No trench found intersecting waveguides with trench area.\n')
            return None

        logger.debug('Trenches found.')
        for block in listcast(sorted(trench_blocks.geoms, key=Trench)):
            # buffer to round corners
            block = block.buffer(self.round_corner, resolution=256, cap_style=1)
            # simplify the shape to avoid path too much dense of points
            block = block.simplify(tolerance=5e-7, preserve_topology=True)
            self._trench_list.append(
                Trench(normalize_polygon(block), delta_floor=self.delta_floor, safe_inner_turns=self.safe_inner_turns)
            )
        logger.debug('Finished append trenches.')

        for index in sorted(listcast(remove), reverse=True):
            del self._trench_list[index]


@attrs.define(repr=False)
class UTrenchColumn(TrenchColumn):
    """Class representing a column of isolation U-trenches."""

    n_pillars: int = 0  #: number of sustaining pillars.
    pillar_width: float = 0.040  #: width of the pillars.

    _id: str = attrs.field(alias='_id', default='UTC')  #: UTrenchColumn ID.
    _trenchbed: list[TR] = attrs.field(alias='_trenchbed', factory=list)  #: List of beds blocks.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        filtered = {att.name: kwargs[att.name] for att in self.__attrs_attrs__ if att.name in kwargs}
        self.__attrs_init__(**filtered)

    @property
    def trench_bed(self) -> list[TR]:
        return self._trenchbed

    @property
    def adj_pillar_width(self) -> float:
        """Pillar size adjusted for the laser beam waist.

        Returns
        -------
        float
            Adjustted pillar size considering the size of the laser focus [mm].
        """
        adj_p = self.pillar_width / 2 + self.beam_waist
        logger.debug(f'The adjusted pillar size is {adj_p} mm.')
        return adj_p

    @property
    def fabrication_time(self) -> float:
        """Total fabrication time.

        The fabrication time is the sum of the lengths of all the walls, floors and bedfloors of all the trenches,
        divided by the translation speed.

        Returns
        -------
        float
            Total fabrication time [s].
        """
        t_box = sum(
            [
                self.nboxz * (self.n_repeat * t.wall_length / self.speed_wall + t.floor_length / self.speed_floor)
                for t in self._trench_list
            ]
        )
        t_bed = sum([b.floor_length / self.speed_floor for b in self._trenchbed])
        fab_time = t_box + t_bed
        logger.debug(f'The total fabrication time for the trench column is {fab_time} s.')
        return fab_time

    def define_trench_bed(self) -> None:
        """Trenchbed shape.
        This method is used to calculate the shape of the plane beneath the column of trenches based on a list of
        trenches.

        The trench bed is defined as a rectangular-ish shape with the top and bottom part with the same shape of the
        top and bottom trench (respectively) of the column of trenches.
        This trench bed can be divided into several `beds` divided by structural pillars of a given `x`-width. The
        width and the number of the pillars can be defined by the user when the ``UTrenchColumn`` object is created.

        This method populated the ``self._trenchbed`` attribute of the ``UTrenchColumn`` object.


        Returns
        -------
        None
        """
        if not self._trench_list:
            logger.critical('No trench is present. Trenchbed cannot be defined.')
            return None

        # Define the trench bed shape as union of a rectangle and the first and last trench of the column.
        # Automatically convert the bed polygon to a MultiPolygon to keep the code flexible to word with the
        # no-pillar case.
        logger.debug(f'Divide bed in {self.n_pillars +1} parts.')
        t1, t2 = self._trench_list[0], self._trench_list[-1]
        tmp_rect = geometry.box(t1.xmin, t1.center[1], t2.xmax, t2.center[1])
        tmp_bed = geometry.MultiPolygon([unary_union([t1.block, t2.block, tmp_rect])])

        # Add pillars and define the bed layer as a Trench object
        xmin, ymin, xmax, ymax = tmp_bed.bounds
        x_pillars = np.linspace(xmin, xmax, self.n_pillars + 2)[1:-1]
        for x in x_pillars:
            tmp_pillar = geometry.LineString([[x, ymin], [x, ymax]]).buffer(self.adj_pillar_width)
            tmp_bed = tmp_bed.difference(tmp_pillar)

        # Add bed blocks
        logger.debug('Add trench beds as trench objects with height = 0.015 mm.')
        self._trenchbed = [
            Trench(
                block=normalize_polygon(p.buffer(-self.round_corner).buffer(self.round_corner)),
                height=0.015,
                delta_floor=self.delta_floor,
                safe_inner_turns=self.safe_inner_turns,
            )
            for p in tmp_bed.geoms
        ]
        return None

    def _dig(
        self,
        coords_list: list[list[tuple[float, float]]],
        remove: list[int] | None = None,
    ) -> None:
        super()._dig(coords_list, remove)
        self.define_trench_bed()


def main() -> None:
    """The main function of the script."""
    # Data
    param_wg = dotdict(speed=20, radius=25, pitch=0.080, int_dist=0.007, samplesize=(25, 3))
    param_tc = dotdict(length=1.0, base_folder='', y_min=-0.1, y_max=19 * param_wg['pitch'] + 0.1, u=[30.339, 32.825])

    # Calculations
    x_c = 0
    coup = [Waveguide(**param_wg) for _ in range(20)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch, 0.035])
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        x_c = wg.x[-1]
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        wg.end()

    # Trench
    utc = UTrenchColumn(x_center=x_c, n_pillars=3, **param_tc)
    utc.dig_from_waveguide(flatten([coup]))

    import matplotlib.pyplot as plt

    # b = T._trench_list[0]
    # b = T._trenchbed[0]
    for tr in utc.trench_list:
        for (x, y) in tr.toolpath():
            plt.plot(x, y)

    plt.axis('equal')


if __name__ == '__main__':
    main()
