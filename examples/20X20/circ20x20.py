from flww.objects.Waveguide import Waveguide
from flww.compiler.PGMCompiler import PGMCompiler
import param as p
import matplotlib.pyplot as plt
import time 

t = time.time()

# 20x20 circuit
circ20x20 = [Waveguide() for _ in range(p.MM)]
increment = [p.swg_length, 0.0, 0.0]

for i, wg in enumerate(circ20x20):
    [xi, yi, zi] = [p.x0, p.y0-(i-0.5*(p.MM+1))*p.pitch, p.depth]
    
    wg.start([xi, yi, zi])
    wg.linear(increment, p.speed)
    wg.sin_bend((-1)**(i%2+1)*p.d1, p.radius, p.speed)
    for j in range(p.NN-1):
        wg.sin_acc((-1)**(j+i%2)*p.d1, p.radius, speed=p.speed)
        wg.sin_bend((-1)**(j+i%2)*p.d2, p.radius, p.speed)
    wg.sin_bend((-1)**(j+i%2+1)*p.d1, p.radius, p.speed)    
    wg.sin_acc((-1)**(j+i%2)*p.d1, p.radius, speed=p.speed)
    wg.linear(increment, p.speed)
    wg.end()

# Plot
fig, ax = plt.subplots()
for wg in circ20x20:
    ax.plot(wg.M['x'][:-1], wg.M['y'][:-1], '-b', linewidth=2.5)
    ax.plot(wg.M['x'][-2:], wg.M['y'][-2:], ':k', linewidth=1)
plt.show()

# Compilation 
gc = PGMCompiler('20x20cirq.pgm', ind_rif=p.ind_env, angle=p.angle)
gc.header()
gc.rpt(wg.num_scan)
for i, wg in enumerate(circ20x20):    
    gc.comment(f'Modo: {i}')
    gc.point_to_instruction(wg.M)
gc.endrpt()
gc.compile_pgm()

print(f'Elapsed time: {time.time() - t:.2f} s.')