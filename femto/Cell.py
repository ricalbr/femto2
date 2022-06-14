import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from femto import Marker, Trench, TrenchColumn, Waveguide


class Cell:
    def __init__(self, dim=(None, None)):
        self.dim = dim
        self.waveguides = []
        self.markers = []
        self.trenches = []

    def add(self, obj):
        if isinstance(obj, Waveguide):
            self.waveguides.append(obj)
        elif isinstance(obj, Marker):
            self.markers.append(obj)
        elif isinstance(obj, Trench):
            self.trenches.append(obj)
        elif isinstance(obj, TrenchColumn):
            for trc in obj:
                self.add(trc)
        else:
            raise TypeError(f'The object must be a Waveguide, Marker or Trench object. {type(obj)} was given.')

    def plot2d(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        for wg in self.waveguides:
            xo, yo, _ = self._shutter_mask(wg.points, shutter=1)
            ax.plot(xo, yo, '-b', linewidth=2.5)
            xc, yc, _ = self._shutter_mask(wg.points, shutter=0)
            ax.plot(xc, yc, ':b', linewidth=1.0)
        for mk in self.markers:
            xo, yo, _ = self._shutter_mask(mk.points, shutter=1)
            ax.plot(xo, yo, '-k', linewidth=2.5)
        for tr in self.trenches:
            ax.add_patch(tr.patch)

    def plot3d(self):
        fig = plt.figure()
        fig.clf()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        for wg in self.waveguides:
            xo, yo, zo = self._shutter_mask(wg.points, shutter=1)
            ax.plot(xo, yo, zo, '-b', linewidth=2.5)
            xc, yc, zc = self._shutter_mask(wg.points, shutter=0)
            ax.plot(xc, yc, zc, ':b', linewidth=1.0)
        for mk in self.markers:
            xo, yo, zo = self._shutter_mask(mk.points, shutter=1)
            ax.plot(xo, yo, '-k', linewidth=2.5)
        for tr in self.trenches:
            pass
            # ax.add_patch(patch_2d_to_3d(tr.patch))
        ax.set_box_aspect(aspect=(2, 1, 0.25))

    # Private interface
    @staticmethod
    def _shutter_mask(points, shutter: int = 1):
        if shutter not in [0, 1]:
            raise ValueError(f'Shutter must be either OPEN (1) or CLOSE (0). Given {shutter}.')
        x, y, z, _, s = points.T
        if shutter == 1:
            mask = s.astype(bool)
        else:
            mask = np.invert(s.astype(bool))
        return x[mask], y[mask], z[mask]
