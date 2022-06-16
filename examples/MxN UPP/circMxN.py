import os
import time

from femto import Cell, Marker, PGMCompiler, TrenchColumn, Waveguide
from param import *

t0 = time.perf_counter()

# MxN circuit
circ = Cell()

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
            circ.add(mk)
            x_trench.append(xl)
        wg.sin_bend((-1) ** (j + i % 2) * d1)
        wg.sin_bend((-1) ** (j + i % 2 + 1) * d2)
    wg.sin_bend((-1) ** (j + i % 2) * d1)
    if i == 0:
        xl, yl, _ = wg.lastpt
        mk = Marker(PARAMETERS_MK)
        mk.cross([xl, yl - 0.2], lx, ly)
        circ.add(mk)
        x_trench.append(xl)
    wg.sin_acc((-1) ** (j + i % 2 + 1) * d1)
    wg.linear(increment)
    wg.end()
    circ.add(wg)

# Trench
for xt in x_trench:
    col = TrenchColumn(PARAMETERS_TC)
    col.x_center = xt
    col.y_min = y0 - 0.5 * MM * PARAMETERS_WG.pitch
    col.y_max = y0 + 0.5 * MM * PARAMETERS_WG.pitch
    col.get_trench(circ.waveguides)
    circ.add(col)

# # Plot
circ.plot2d()
# plt.show()

# Compilation
# # OPTICAL CIRCUIT
PARAMETERS_GC.filename = f'{MM}x{NN}CIRC.pgm'
with PGMCompiler(PARAMETERS_GC) as gc:
    with gc.repeat(PARAMETERS_WG.scan):
        for i, wg in enumerate(circ.waveguides):
            gc.comment(f' +--- Modo: {i + 1} ---+')
            gc.write(wg.points)

# # MARKERS
PARAMETERS_GC.filename = f'{MM}x{NN}MARKERS.pgm'
with PGMCompiler(PARAMETERS_GC) as gc:
    for i, c in enumerate(circ.markers):
        gc.comment(f' +--- Croce: {i + 1} ---+')
        gc.write(c.points)

# # TRENCH
for col_index, col in enumerate(circ.trench_cols):
    col_filename = os.path.join(os.getcwd(), 's-trench', f'FARCALL_COLONNA{col_index + 1:03}.pgm')
    PARAMETERS_GC.filename = col_filename
    with PGMCompiler(PARAMETERS_GC) as gc:
        gc.trench(col, col_index, base_folder=PARAMETERS_TC.base_folder, u=[31.7, 38.4])
        gc.homing()

print(f'Elapsed time: {time.perf_counter() - t0:.2f} s.')

ttime = 0
[ttime := ttime + wg.wtime for wg in circ.waveguides]
for col in circ.trench_cols:
    ttime += col.wtime
print('Estimated fabrication time: ', time.strftime('%H:%M:%S', time.gmtime(ttime)))
