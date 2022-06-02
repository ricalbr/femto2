import warnings
from typing import List

import numpy as np
from descartes import PolygonPatch

from femto.Parameters import TrenchParameters, WaveguideParameters
from femto.Waveguide import Waveguide

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from shapely.geometry import LineString, Polygon, MultiPolygon, box, polygon


class Trench:
    def __init__(self,
                 block: Polygon,
                 delta_floor: float = 0.001):

        self.block = block
        self.delta_floor = delta_floor

    @property
    def center(self):
        return np.asarray([self.block.centroid.x, self.block.centroid.y])

    @property
    def patch(self, fc='k', ec='k', alpha=1, zorder=1):
        return PolygonPatch(self.block,
                            fc=fc,
                            ec=ec,
                            alpha=alpha,
                            zorder=zorder)

    def trench_paths(self):
        polygon_list = [self.block]
        while polygon_list:
            current_poly = polygon_list.pop(0)
            inset_polygon = self._buffer_polygon(current_poly)
            if inset_polygon and inset_polygon.type == 'MultiPolygon':
                polygon_list.extend(list(inset_polygon))
                for poly in list(inset_polygon):
                    yield np.array(poly.exterior.coords).T
            elif inset_polygon and inset_polygon.type == 'Polygon':
                polygon_list.append(inset_polygon)
                yield np.array(inset_polygon.exterior.coords).T
            elif inset_polygon:
                raise ValueError('Trench block should be either Polygon or',
                                 f'Multipolygon. Given {inset_polygon.type}')

    # Private interface
    def _buffer_polygon(self, shape: Polygon, inset=True):
        if inset:
            new_polygon = shape.buffer(-self.delta_floor)
        else:
            new_polygon = shape.buffer(self.delta_floor)
        return new_polygon if new_polygon.is_valid else None


class TrenchColumn:
    def __init__(self, param: TrenchParameters):

        self.param = param
        self._trench_list = []
        self._x_c = self.param.x_center
        self._y_min = self.param.y_min
        self._y_max = self.param.y_max

        self._rect = self._make_box()

    def __iter__(self):
        return iter(self._trench_list)

    @property
    def x_c(self):
        return self._x_c

    @x_c.setter
    def x_c(self, x_center):
        self._x_c = x_center
        if x_center is not None:
            self._rect = self._make_box()

    @property
    def y_min(self):
        return self._y_min

    @y_min.setter
    def y_min(self, y_min):
        self._y_min = y_min
        if y_min is not None:
            self._rect = self._make_box()

    @property
    def y_max(self):
        return self._y_max

    @y_max.setter
    def y_max(self, y_max):
        self._y_max = y_max
        if y_max is not None:
            self._rect = self._make_box()

    def patch(self, fc='k', ec=None, alpha=1, zorder=1):
        if isinstance(self._rect, MultiPolygon):
            return PolygonPatch(self._rect,
                                fc=fc, ec=ec,
                                alpha=alpha,
                                zorder=zorder)

    def get_trench(self, circuit: List):
        if not all([isinstance(wg, Waveguide) for wg in circuit]):
            raise TypeError('Elements circuit list must be of type Waveguide.')

        for wg in circuit:
            x, y = wg.x[:-2], wg.y[:-2]
            dilated = (LineString(list(zip(x, y)))
                       .buffer(self.param.adj_bridge, cap_style=1))
            self._rect = self._rect.difference(dilated)

        for block in list(self._rect):
            block = (polygon.orient(block)
                     .buffer(self.param.round_corner,
                             resolution=150, cap_style=1))
            trench = Trench(block, self.param.delta_floor)
            self._trench_list.append(trench)

    # Private interface
    def _make_box(self):
        if (self._x_c is not None and
                self._y_min is not None and
                self._y_max is not None):
            return box(self._x_c - self.param.length / 2, self._y_min,
                       self._x_c + self.param.length / 2, self._y_max)
        else:
            return None


def _example():
    import matplotlib.pyplot as plt

    # Data
    pitch = 0.080
    int_dist = 0.007
    d_bend = 0.5 * (pitch - int_dist)
    x_mid = None

    PARAMETERS_TC = TrenchParameters(
        lenght=1.0,
        nboxz=4,
        deltaz=0.0015,
        h_box=0.075,
        base_folder=r'C:\Users\Capable\Desktop\RiccardoA'
    )

    PARAMETERS_WG = WaveguideParameters(
        scan=6,
        speed=20,
        radius=15
    )

    # Calculations
    coup = [Waveguide(PARAMETERS_WG) for _ in range(20)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * pitch, 0.035]).sin_acc((-1) ** i * d_bend)
        x_mid = wg.x[-1]
        wg.sin_acc((-1) ** i * d_bend).end()

    PARAMETERS_TC.x_center = x_mid
    PARAMETERS_TC.y_min = -0.1
    PARAMETERS_TC.y_max = 19 * pitch + 0.1
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
