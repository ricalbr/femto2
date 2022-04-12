from femto.objects.Waveguide import Waveguide
from femto.objects.Marker import Marker
from femto.compiler.PGMCompiler import PGMCompiler
import param as p
import matplotlib.pyplot as plt
from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt5')
import time

t = time.time()

# 20x20 circuit
circ20x20 = [Waveguide() for _ in range(p.MM)]
marker = [Marker(1, 0.06) for _ in range(p.MM)]

increment = [p.swg_length, 0.0, 0.0]

for i, wg in enumerate(circ20x20):
    [xi, yi, zi] = [p.x0, p.y0-(i-0.5*(p.MM+1))*p.pitch, p.depth]

    wg.start([xi, yi, zi])
    wg.linear(increment, p.speed)
    wg.sin_bend((-1)**(i%2+1)*p.d1, p.radius, p.speed)
    for j in range(p.NN-1):
        wg.sin_bend((-1)**(j+i%2)*p.d1, p.radius, speed=p.speed)
        if i == p.NN-1: marker[j].cross([wg.M['x'].iloc[-1],wg.M['y'].iloc[-1]-.2,0.001])
        wg.sin_bend((-1)**(j+i%2+1)*p.d1, p.radius, speed=p.speed)
        wg.sin_bend((-1)**(j+i%2)*p.d2, p.radius, p.speed)
    wg.sin_bend((-1)**(j+i%2+1)*p.d1, p.radius, p.speed)
    if i == p.NN-1: marker[j+1].cross([wg.M['x'].iloc[-1],wg.M['y'].iloc[-1]-.2,0.001])
    wg.sin_acc((-1)**(j+i%2)*p.d1, p.radius, speed=p.speed)
    wg.linear(increment, p.speed)
    wg.end()

# Plot
fig, ax = plt.subplots()
for wg in circ20x20:
    ax.plot(wg.M['x'][:-1], wg.M['y'][:-1], '-b', linewidth=2.5)
    ax.plot(wg.M['x'][-2:], wg.M['y'][-2:], ':k', linewidth=1)
    
for c in marker:
    ax.plot(c.M['x'][:-1], c.M['y'][:-1], '-k', linewidth=1)
plt.show()

# Compilation
# # OPTICAL CIRCUIT
gc = PGMCompiler('20x20CIRC.pgm', ind_rif=p.ind_env, angle=p.angle)
gc.header()
gc.rpt(wg.num_scan)
for i, wg in enumerate(circ20x20):
    gc.comment(f' +--- Modo: {i+1} ---+')
    gc.point_to_instruction(wg.M)
gc.endrpt()
gc.compile_pgm()

# # MARKERS
gc = PGMCompiler('20x20MARKERS.pgm', ind_rif=p.ind_env, angle=p.angle)
gc.header()
for i, c in enumerate(marker):
    gc.comment(f' +--- Croce: {i+1} ---+')
    gc.rpt(c.num_scan)
    gc.point_to_instruction(c.M)
    gc.endrpt()
gc.compile_pgm()



print(f'Elapsed time: {time.time() - t:.2f} s.')
