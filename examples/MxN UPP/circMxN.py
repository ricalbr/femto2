import time

import matplotlib.pyplot as plt
from femto import PGMCompiler, Waveguide
from param import *

t0 = time.perf_counter()

# MxN circuit
circ = {
    'waveguide': [Waveguide(PARAMETERS_WG) for _ in range(MM)],
    # 'marker': [Marker(PARAMETERS_MK) for _ in range(NN)],
    # 'trench': [TrenchColumn(PARAMETERS_TC) for _ in range(NN)]
}

x_trench = []
for i, wg in enumerate(circ['waveguide']):
    [xi, yi, zi] = [x0, y0 + (i - 0.5 * (MM - 1)) * PARAMETERS_WG.pitch, z0]

    wg.start([xi, yi, zi])
    wg.linear(increment)
    wg.sin_bend((-1) ** (i % 2) * d1)
    for j in range(NN - 1):
        wg.sin_bend((-1) ** (j + i % 2 + 1) * d1)
        if i == 0:
            xl, yl, _ = wg.lastpt
            # circ['marker'][j].cross([xl, yl - 0.2], lx, ly)
            x_trench.append(xl)
        wg.sin_bend((-1) ** (j + i % 2) * d1)
        wg.sin_bend((-1) ** (j + i % 2 + 1) * d2)
    wg.sin_bend((-1) ** (j + i % 2) * d1)
    if i == 0:
        xl, yl, _ = wg.lastpt
        # circ['marker'][j + 1].cross([xl, yl - 0.2], lx, ly)
        x_trench.append(xl)
    wg.sin_acc((-1) ** (j + i % 2 + 1) * d1)
    wg.linear(increment)
    wg.end()

# # Trench
# for xt, col in zip(x_trench, circ['trench']):
#     col.x_c = xt
#     col.y_min = y0 - 0.5 * MM * PARAMETERS_WG.pitch
#     col.y_max = y0 + 0.5 * MM * PARAMETERS_WG.pitch
#     col.get_trench(circ['waveguide'])

# # Plot
fig, ax = plt.subplots()
for wg in circ['waveguide']:
    ax.plot(wg.x[:-1], wg.y[:-1], '-b', linewidth=0.5)  # shutter on
    ax.plot(wg.x[-2:], wg.y[-2:], ':k', linewidth=0.5)  # shutter off

# for c in circ['marker']:
#     ax.plot(c.x[:-1], c.y[:-1], '-k', linewidth=1.25)

# for col in circ['trench']:
#     for t in col:
#         ax.add_patch(t.patch)
plt.tight_layout(pad=0)
# plt.show()

# Compilation
# # OPTICAL CIRCUIT
PARAMETERS_GC.filename = f'{MM}x{NN}CIRC.pgm'
with PGMCompiler(PARAMETERS_GC) as gc:
    with gc.repeat(PARAMETERS_WG.scan):
        for i, wg in enumerate(circ['waveguide']):
            gc.comment(f' +--- Modo: {i + 1} ---+')
            gc.write(wg.points)

# # # MARKERS
# PARAMETERS_GC.filename = f'{MM}x{NN}MARKERS.pgm'
# with PGMCompiler(PARAMETERS_GC) as gc:
#     for i, c in enumerate(circ['marker']):
#         gc.comment(f' +--- Croce: {i + 1} ---+')
#         gc.write(c.points)
#
# # # TRENCH
# for col_index, col in enumerate(circ['trench']):
#     col_filename = os.path.join(os.getcwd(), 's-trench', f'FARCALL_COLONNA{col_index + 1:03}.pgm')
#     PARAMETERS_GC.filename = col_filename
#     with PGMCompiler(PARAMETERS_GC) as gc:
#         gc.trench(col, col_index, base_folder=PARAMETERS_TC.base_folder, u=[31.7, 38.4])
#         gc.homing()

print(f'Elapsed time: {time.perf_counter() - t0:.2f} s.')

ttime = 0
[ttime := ttime + wg.wtime for wg in circ['waveguide']]
# [ttime := ttime + wg.wtime for wg in circ['waveguide']]
# for col in circ['trench']:
#     ttime += col.wtime
print('Estimated fabrication time: ', time.strftime('%H:%M:%S', time.gmtime(ttime)))
