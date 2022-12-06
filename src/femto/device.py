from __future__ import annotations

import collections
import copy
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from femto.helpers import flatten
from femto.marker import Marker
from femto.trench import TrenchColumn
from femto.waveguide import NasuWaveguide
from femto.waveguide import Waveguide
from femto.writer import MarkerWriter
from femto.writer import NasuWriter
from femto.writer import TrenchWriter
from femto.writer import WaveguideWriter


class Device:
    def __init__(self, **param) -> None:
        self._param: dict[str, Any] = dict(**param)
        self.unparsed_objects: list[Any] = []
        self.fig: go.Figure | None = None
        self.writers = {
            Waveguide: WaveguideWriter(wg_list=[], **param),
            NasuWaveguide: NasuWriter(nw_list=[], **param),
            TrenchColumn: TrenchWriter(tc_list=[], **param),
            Marker: MarkerWriter(mk_list=[], **param),
        }

    def append(self, obj: Any) -> None:
        self.parse_objects(unparsed_objects=copy.copy(flatten([obj])))

    def extend(self, obj: list[Any]) -> None:
        if not isinstance(obj, list):
            raise TypeError(f'The object must be a list. {type(obj)} was given.')
        self.parse_objects(unparsed_objects=copy.copy(obj))

    def parse_objects(self, unparsed_objects: Any | list[Any]) -> None:
        # split the unparsed_object list based on the type of each element
        d = collections.defaultdict(list)
        while unparsed_objects:
            obj = unparsed_objects.pop(0)
            if isinstance(obj, list):
                d[type(obj[0])].append(obj)
            else:
                d[type(obj)].append(obj)

        # add each element to the type-matching writer
        for k, e in d.items():
            try:
                self.writers[k].extend(e)
            except KeyError as err:
                raise TypeError(f'Found unexpected type {err.args}.')

    def plot2d(self, show: bool = True, save: bool = False) -> None:
        self.fig = go.Figure()
        for writer in self.writers.values():
            # TODO: fix standard fig update
            self.fig = writer.plot2d(self.fig)
            self.fig = writer.standard_2d_figure_update(self.fig)
        if show:
            self.fig.show()
        if save:
            self.save()

    def plot3d(self, show: bool = True, save: bool = False) -> None:
        self.fig = go.Figure()
        for key, writer in self.writers.items():
            try:
                self.fig = writer.plot3d(self.fig)
                self.fig = writer.standard_3d_figure_update(self.fig)
            except NotImplementedError:
                print(f'3D {key} plot not yet implemented.\n')
        if show:
            self.fig.show()
        if save:
            self.save()

    def pgm(self, verbose: bool = True) -> None:
        for key, writer in self.writers.items():
            if verbose:
                print(f'Exporting {key.__name__} objects...')
            writer.pgm(verbose=verbose)
        if verbose:
            print('Export .pgm files complete.\n')

    def save(self, filename='scheme.html', opt: dict[str, Any] | None = None) -> None:
        if opt is None:
            opt = dict()
        default_opt = {'width': 1980, 'height': 1080, 'scale': 2, 'engine': 'kaleido'}
        opt = {**default_opt, **opt}

        if self.fig is None:
            return None

        fn = Path(filename)
        if fn.suffix.lower() in ['.html', '']:
            self.fig.write_html(str(fn.with_suffix('.html')))
        else:
            self.fig.write_image(str(fn), **opt)


def main() -> None:
    from femto.trench import TrenchColumn
    from femto.waveguide import Waveguide

    # Parameters
    PARAM_WG: dict[str, Any] = dict(speed=20, radius=25, pitch=0.080, int_dist=0.007, samplesize=(25, 3))
    PARAM_TC: dict[str, Any] = dict(length=1.0, base_folder='', y_min=-0.1, y_max=4 * 0.080 + 0.1, u=[30.0, 32.0])
    PARAM_GC: dict[str, Any] = dict(filename='testCell.pgm', laser='PHAROS', new_origin=(0.5, 0.5), samplesize=(25, 1))

    dev = Device(**PARAM_GC)

    # Waveguides
    x_center = 0
    coup = [Waveguide(**PARAM_WG) for _ in range(5)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch, 0.035])
        wg.sin_acc((-1) ** i * wg.dy_bend)
        x_center = wg.x[-1]
        wg.sin_acc((-1) ** i * wg.dy_bend)
        wg.end()
        dev.append(wg)

    # Trench
    T = TrenchColumn(x_center=x_center, **PARAM_TC)
    T.dig_from_waveguide(coup, remove=[0, 1])
    dev.append(T)

    # Export
    dev.plot2d()
    dev.save('circuit_scheme.pdf')
    dev.pgm()


if __name__ == '__main__':
    main()
