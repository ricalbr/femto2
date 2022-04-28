# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:41:06 2022

"""

from femto.objects.Waveguide import Waveguide
from femto.objects.Marker import Marker
from femto.compiler.PGMCompiler import PGMCompiler
import param as p
import matplotlib.pyplot as plt
import time

t = time.perf_counter()

# Mix of guides to test

mix = {
    'straight': [Waveguide() for _ in range(5)],
    'sine': [Waveguide() for _ in range(5)],
    'coupler': [Waveguide() for _ in range(4)]
}

for i, wg in enumerate(mix['straight']):
    [xi, yi, zi] = [p.x0, p.y0-(i-0.5*(5+1))*p.pitch, p.depth]
    wg.start([xi, yi, zi])
    wg.linear([3, 0, 0.005*i], p.speed)
    wg.end()

for w, wg in enumerate(mix['sine']):
    [xi, yi, zi] = [p.x0, p.y0-(w-0.5*(5+1))*p.pitch, p.depth]
    wg.start([xi, yi, zi])


for k, wg in enumerate(mix['coupler']):
    [xi, yi, zi] = [p.x0, p.y0-(k-0.5*(4+1))*p.pitch, p.depth]
    wg.start([xi, yi, zi])
    wg.linear(p.increment, p.speed)
    wg.sin_bend((-1)**(k % 2+1)*p.d1, p.radius, p.speed)
    for j in range(p.NN-1):
        wg.sin_bend((-1)**(j+k % 2)*p.d1, p.radius, speed=p.speed)
        if k == p.NN-1:
            mix['marker'][j].cross([wg.x[-1], wg.y[-1]-.2])
        wg.sin_bend((-1)**(j+k % 2+1)*p.d1, p.radius, speed=p.speed)
        wg.sin_bend((-1)**(j+k % 2)*p.d2, p.radius, p.speed)
    wg.sin_bend((-1)**(j+k % 2+1)*p.d1, p.radius, p.speed)
    if k == p.NN-1:
        mix['marker'][j+1].cross([wg.x[-1], wg.y[-1]-.2])
    wg.sin_acc((-1)**(j+k % 2)*p.d1, p.radius, speed=p.speed)
    wg.linear(p.increment, p.speed)
    wg.end()

# Plot
fig, ax = plt.subplots()
for wg in mix['straight']:
    ax.plot(wg.x[:-1], wg.y[:-1], '-b', linewidth=2.5)
    ax.plot(wg.x[-2:], wg.y[-2:], ':k', linewidth=1)

for wg in mix['sine']:
    ax.plot(wg.x[:-1], wg.y[:-1], '-b', linewidth=2.5)
    ax.plot(wg.x[-2:], wg.y[-2:], ':k', linewidth=1)


# Compilation
# # OPTICAL CIRCUIT
gc = PGMCompiler('MistoMare.pgm', ind_rif=p.ind_env, angle=p.angle)
gc.header()
gc.rpt(wg.num_scan)
for i, wg in enumerate(mix['straight']):
    gc.comment(f' +--- Modo: {i+1} ---+')
    gc.point_to_instruction(wg.M)
gc.endrpt()
gc.compile_pgm()

# # # MARKERS
# gc = PGMCompiler('20x20MARKERS.pgm', ind_rif=p.ind_env, angle=p.angle)
# gc.header()
# for i, c in enumerate(circ['marker']):
#     gc.comment(f' +--- Croce: {i+1} ---+')
#     gc.rpt(c.num_scan)
#     gc.point_to_instruction(c.M)
#     gc.endrpt()
# gc.compile_pgm()

print(f'Elapsed time: {time.perf_counter() - t:.2f} s.')
plt.show()
