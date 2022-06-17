import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from femto import Marker, PGMCompiler, Trench, TrenchColumn, Waveguide


class Cell(PGMCompiler):
    def __init__(self, param, dim=(None, None)):
        super(Cell, self).__init__(param)
        self.dim = dim
        self.waveguides = []
        self.markers = []
        self.trench_cols = []
        self.trenches = []
        self.fig = None
        self.ax = None

    def add(self, obj):
        if isinstance(obj, Marker):
            self.markers.append(obj)
        elif isinstance(obj, Waveguide):
            self.waveguides.append(obj)
        elif isinstance(obj, Trench):
            self.trenches.append(obj)
        elif isinstance(obj, TrenchColumn):
            for trc in obj:
                self.add(trc)
            self.trench_cols.append(obj)
        else:
            raise TypeError(f'The object must be a Waveguide, Marker or Trench object. {type(obj)} was given.')

    def plot2d(self, shutter_close=True, aspect='auto', wg_style=None, sc_style=None, mk_style=None, tc_style=None):
        if tc_style is None:
            tc_style = {}
        if mk_style is None:
            mk_style = {}
        if sc_style is None:
            sc_style = {}
        if wg_style is None:
            wg_style = {}
        default_wgargs = {'linestyle': '-', 'color': 'b', 'linewidth': 2.0}
        wgargs = {**default_wgargs, **wg_style}
        default_scargs = {'linestyle': ':', 'color': 'b', 'linewidth': 0.5}
        scargs = {**default_scargs, **sc_style}
        default_mkargs = {'linestyle': '-', 'color': 'k', 'linewidth': 2.0}
        mkargs = {**default_mkargs, **mk_style}
        default_tcargs = {}
        tcargs = {**default_tcargs, **tc_style}

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('X [mm]')
        self.ax.set_ylabel('Y [mm]')
        for wg in self.waveguides:
            p = np.array(self.transform_points(wg.points)).T
            xo, yo, _ = self._shutter_mask(p, shutter=1)
            self.ax.plot(xo, yo, **wgargs)
            if shutter_close:
                xc, yc, _ = self._shutter_mask(p, shutter=0)
                self.ax.plot(xc, yc, **scargs)
        for mk in self.markers:
            p = np.array(self.transform_points(mk.points)).T
            xo, yo, _ = self._shutter_mask(p, shutter=1)
            self.ax.plot(xo, yo, **mkargs)
        for tr in self.trenches:
            self.ax.add_patch(tr.patch)

        # Glass
        if self.xsample is not None:
            self.ax.axvline(x=0.0 - self.new_origin[0])
            self.ax.axvline(x=self.xsample - self.new_origin[0])

        # Origin
        self.ax.plot(0.0, 0.0, 'or')
        self.ax.annotate('(0,0)', (0.0, 0.0), textcoords="offset points", xytext=(0, 10), ha='left', color='r')
        if isinstance(aspect, str) and aspect.lower() not in ['auto', 'equal']:
            raise ValueError(f'aspect must be either `auto` or `equal`. Given {aspect.lower()}.')
        self.ax.set_aspect(aspect)

    def plot3d(self, shutter_close=True, wg_style=None, sc_style=None, mk_style=None, tc_style=None):
        if tc_style is None:
            tc_style = {}
        if mk_style is None:
            mk_style = {}
        if sc_style is None:
            sc_style = {}
        if wg_style is None:
            wg_style = {}
        default_wgargs = {'linestyle': '-', 'color': 'b', 'linewidth': 2.0}
        wgargs = {**default_wgargs, **wg_style}
        default_scargs = {'linestyle': ':', 'color': 'b', 'linewidth': 0.5}
        scargs = {**default_scargs, **sc_style}
        default_mkargs = {'linestyle': '-', 'color': 'k', 'linewidth': 2.0}
        mkargs = {**default_mkargs, **mk_style}
        default_tcargs = {}
        tcargs = {**default_tcargs, **tc_style}

        self.fig = plt.figure()
        self.fig.clf()
        self.ax = Axes3D(self.fig, auto_add_to_figure=False)
        self.fig.add_axes(self.ax)
        self.ax.set_xlabel('X [mm]')
        self.ax.set_ylabel('Y [mm]')
        self.ax.set_zlabel('Z [mm]')
        for wg in self.waveguides:
            xo, yo, zo = self._shutter_mask(wg.points, shutter=1)
            self.ax.plot(xo, yo, zo, **wgargs)
            if shutter_close:
                xc, yc, zc = self._shutter_mask(wg.points, shutter=0)
                self.ax.plot(xc, yc, zc, **scargs)
        for mk in self.markers:
            xo, yo, zo = self._shutter_mask(mk.points, shutter=1)
            self.ax.plot(xo, yo, zo, **mkargs)
        for tr in self.trenches:
            pass
            # ax.add_patch(patch_2d_to_3d(tr.patch))
        self.ax.set_box_aspect(aspect=(2, 1, 0.25))
        self.ax.plot(0.0, 0.0, 0.0, 'or')

    def save(self, filename='device_scheme.pdf', bbox_inches='tight'):
        self.fig.savefig(filename, bbox_inches=bbox_inches)

    # Private interface
    @staticmethod
    def _shutter_mask(points, shutter: int = 1):
        if shutter not in [0, 1]:
            raise ValueError(f'Shutter must be either OPEN (1) or CLOSE (0). Given {shutter}.')
        x, y, z, _, s = points.T
        ym = np.where(s == shutter, y, np.nan)
        zm = np.where(s == shutter, z, np.nan)
        return x, ym, zm


def _example():
    from femto.helpers import dotdict

    PARAMETERS_GC = dotdict(
        filename='testMarker.pgm',
        lab='CAPABLE',
        new_origin=(1.0, -0.0),
        samplesize=(25, 25),
        angle=0.0,
    )

    PARAMETERS_WG = dotdict(
        scan=6,
        speed=20,
        radius=15,
        pitch=0.080,
        int_dist=0.007,
        lsafe=5,
    )

    increment = [PARAMETERS_WG.lsafe, 0, 0]
    c = Cell(PARAMETERS_GC)

    # Calculations
    mzi = [Waveguide(PARAMETERS_WG) for _ in range(2)]
    for index, wg in enumerate(mzi):
        [xi, yi, zi] = [-2, -wg.pitch / 2 + index * wg.pitch, 0.035]

        wg.start([xi, yi, zi]) \
            .linear(increment) \
            .sin_mzi((-1) ** index * wg.dy_bend) \
            .linear([5, 0, 0]) \
            .sin_mzi((-1) ** index * wg.dy_bend) \
            .linear([27, yi, zi], mode='ABS')
        wg.end()
        c.add(wg)

    c.plot2d()
    # c.save()
    # plt.show()


if __name__ == '__main__':
    _example()
