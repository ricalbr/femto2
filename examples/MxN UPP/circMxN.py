import os
import time

import matplotlib.pyplot as plt

import param as p
from femto import Marker, PGMCompiler, TrenchColumn, Waveguide

t0 = time.perf_counter()

# MxN circuit
circ = {
    'waveguide': [Waveguide(p.PARAMETERS_WG) for _ in range(p.MM)],
    'marker': [Marker() for _ in range(p.NN)],
    'trench': [TrenchColumn(p.PARAMETERS_TC) for _ in range(p.NN)]
}

x_trench = []
for i, wg in enumerate(circ['waveguide']):
    [xi, yi, zi] = [p.x0, p.y0 + (i - 0.5 * (p.MM - 1)) * p.pitch, p.z0]

    wg.start([xi, yi, zi])
    wg.linear(p.increment)
    wg.sin_bend((-1) ** (i % 2) * p.d1)
    for j in range(p.NN - 1):
        wg.sin_bend((-1) ** (j + i % 2 + 1) * p.d1)
        if i == 0:
            xl, yl, _ = wg.lastpt
            circ['marker'][j].cross([xl, yl - 0.2], p.lx, p.ly)
            x_trench.append(xl)
        wg.sin_bend((-1) ** (j + i % 2) * p.d1)
        wg.sin_bend((-1) ** (j + i % 2 + 1) * p.d2)
    wg.sin_bend((-1) ** (j + i % 2) * p.d1)
    if i == 0:
        xl, yl, _ = wg.lastpt
        circ['marker'][j + 1].cross([xl, yl - 0.2], p.lx, p.ly)
        x_trench.append(xl)
    wg.sin_acc((-1) ** (j + i % 2 + 1) * p.d1)
    wg.linear(p.increment)
    wg.end()

# Trench
for xt, col in zip(x_trench, circ['trench']):
    col.x_c = xt
    col.y_min = p.y0 - 0.5 * p.MM * p.pitch
    col.y_max = p.y0 + 0.5 * p.MM * p.pitch
    col.get_trench(circ['waveguide'])

# # Plot
fig, ax = plt.subplots()
for wg in circ['waveguide']:
    ax.plot(wg.x[:-1], wg.y[:-1], '-b', linewidth=0.5)  # shutter on
    ax.plot(wg.x[-2:], wg.y[-2:], ':k', linewidth=0.5)  # shutter off

for c in circ['marker']:
    ax.plot(c.x[:-1], c.y[:-1], '-k', linewidth=1.25)

for col in circ['trench']:
    for t in col:
        ax.add_patch(t.patch)
plt.tight_layout(pad=0)
# plt.show()

# Compilation
# # OPTICAL CIRCUIT
p.PARAMETERS_GC.filename = f'{p.MM}x{p.NN}CIRC.pgm'
with PGMCompiler(p.PARAMETERS_GC) as gc:
    with gc.repeat(p.PARAMETERS_WG.scan):
        for i, wg in enumerate(circ['waveguide']):
            gc.comment(f' +--- Modo: {i + 1} ---+')
            gc.write(wg.points)

# # MARKERS
p.PARAMETERS_GC.filename = f'{p.MM}x{p.NN}MARKERS.pgm'
with PGMCompiler(p.PARAMETERS_GC) as gc:
    for i, c in enumerate(circ['marker']):
        gc.comment(f' +--- Croce: {i + 1} ---+')
        gc.write(c.points)

# # TRENCH
for col_index, col in enumerate(circ['trench']):
    col_filename = os.path.join(os.getcwd(), 's-trench', f'FARCALL_COLONNA{col_index + 1:03}.pgm')
    p.PARAMETERS_GC.filename = col_filename
    with PGMCompiler(p.PARAMETERS_GC) as gc:
        gc.trench(col, col_index,
                  base_folder=p.PARAMETERS_TC.base_folder,
                  u=[31.7, 38.4])
        gc.homing()

print(f'Elapsed time: {time.perf_counter() - t0:.2f} s.')
