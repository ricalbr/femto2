import time

from femto import Cell, Marker, TrenchColumn, Waveguide
from param import *

t0 = time.perf_counter()

# MxN circuit
circ = Cell(PARAMETERS_GC)

x_trench = []
for i in range(MM):
    [xi, yi, zi] = [x0, y0 + (i - 0.5 * (MM - 1)) * PARAMETERS_WG.pitch, PARAMETERS_WG.depth]

    wg = Waveguide(PARAMETERS_WG)
    wg.start([xi, yi, zi])
    wg.linear(increment)
    wg.sin_bend((-1) ** (i % 2) * d1)
    for j in range(NN - 1):
        wg.sin_bend((-1) ** (j + i % 2 + 1) * d1)
        if i == 0:
            xl, yl, _ = wg.lastpt
            mk = Marker(PARAMETERS_MK)
            mk.cross([xl, yl - 0.2], lx, ly)
            circ.append(mk)
            x_trench.append(xl)
        wg.sin_bend((-1) ** (j + i % 2) * d1)
        wg.sin_bend((-1) ** (j + i % 2 + 1) * d2)
    wg.sin_bend((-1) ** (j + i % 2) * d1)
    if i == 0:
        xl, yl, _ = wg.lastpt
        mk = Marker(PARAMETERS_MK)
        mk.cross([xl, yl - 0.2], lx, ly)
        circ.append(mk)
        x_trench.append(xl)
    wg.sin_acc((-1) ** (j + i % 2 + 1) * d1)
    wg.linear(increment)
    wg.end()
    circ.append(wg)

# Trench
for xt in x_trench:
    col = TrenchColumn(PARAMETERS_TC)
    col.x_center = xt
    col.y_min = y0 - 0.5 * MM * PARAMETERS_WG.pitch
    col.y_max = y0 + 0.5 * MM * PARAMETERS_WG.pitch
    col.get_trench(circ.waveguides)
    circ.append(col)

# # Plot and compilation
circ.plot2d()
circ.pgm()
