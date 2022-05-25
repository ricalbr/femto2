from femto import Waveguide, TrenchColumn, Marker, PGMCompiler
import matplotlib.pyplot as plt
import param as p
import time
import os

t0 = time.perf_counter()

# MxN circuit
circ = {
    'waveguide': [Waveguide() for _ in range(p.MM)],
    'marker': [Marker() for _ in range(p.NN)],
    'trench': [TrenchColumn() for _ in range(p.NN)]
    }

x_trench = []
for i, wg in enumerate(circ['waveguide']):
    [xi, yi, zi] = [p.x0, p.y0+(i-0.5*(p.MM-1))*p.pitch, p.depth]

    wg.start([xi, yi, zi])
    wg.linear(p.increment, speed=p.speed)
    wg.sin_bend((-1)**(i % 2)*p.d1, p.radius, speed=p.speed, N=p.N)
    for j in range(p.NN-1):
        wg.sin_bend((-1)**(j+i % 2+1)*p.d1, p.radius, speed=p.speed, N=p.N)
        if i == 0:
            xl, yl, _ = wg.lastpt
            circ['marker'][j].cross([xl, yl-0.2], p.lx, p.ly)
            x_trench.append(xl)
        wg.sin_bend((-1)**(j+i % 2)*p.d1, p.radius, speed=p.speed, N=p.N)
        wg.sin_bend((-1)**(j+i % 2+1)*p.d2, p.radius, speed=p.speed, N=p.N)
    wg.sin_bend((-1)**(j+i % 2)*p.d1, p.radius, speed=p.speed, N=p.N)
    if i == 0:
        xl, yl, _ = wg.lastpt
        circ['marker'][j+1].cross([xl, yl-0.2], p.lx, p.ly)
        x_trench.append(xl)
    wg.sin_acc((-1)**(j+i % 2+1)*p.d1, p.radius, speed=p.speed, N=p.N)
    wg.linear(p.increment, speed=p.speed)
    wg.end()

# Trench
for xt, col in zip(x_trench, circ['trench']):
    col.x_c = xt
    col.y_min = p.y0 - 0.5*p.MM*p.pitch
    col.y_max = p.y0 + 0.5*p.MM*p.pitch
    col.get_trench(circ['waveguide'])

# # Plot
# fig, ax = plt.subplots()
# for wg in circ['waveguide']:
#     ax.plot(wg.x[:-1], wg.y[:-1], '-b', linewidth=0.5) # shutter on
#     ax.plot(wg.x[-2:], wg.y[-2:], ':k', linewidth=0.5) # shutter off

# for c in circ['marker']:
#     ax.plot(c.x[:-1], c.y[:-1], '-k', linewidth=1.25)

# for col in circ['trench']:
#     for t in col.trench_list:
#         ax.add_patch(t.patch)
# plt.tight_layout(pad=0)
# plt.show()

# Compilation
# # OPTICAL CIRCUIT
with PGMCompiler(f'{p.MM}x{p.NN}CIRC.pgm',
                 ind_rif=p.ind_env,
                 angle=p.angle) as gc:
    with gc.repeat(circ['waveguide'][0].num_scan):
        for i, wg in enumerate(circ['waveguide']):
            gc.comment(f' +--- Modo: {i+1} ---+')
            gc.write(wg.points)

# # MARKERS
with PGMCompiler(f'{p.MM}x{p.NN}MARKERS.pgm',
                 ind_rif=p.ind_env,
                 angle=p.angle) as gc:
    for i, c in enumerate(circ['marker']):
        gc.comment(f' +--- Croce: {i+1} ---+')
        gc.write(c.points)

# # TRENCH
for col_index, col in enumerate(circ['trench']):
    col_filename = os.path.join(os.getcwd(), 's-trench',
                                f'FARCALL_COLONNA{col_index+1:03}')
    with PGMCompiler(col_filename, ind_rif=p.ind_env, angle=p.angle) as gc:
        gc.trench(col, col_index, base_folder=p.base_folder, u=[31.7, 38.4])
        gc.homing()

print(f'Elapsed time: {time.perf_counter() - t0:.2f} s.')
