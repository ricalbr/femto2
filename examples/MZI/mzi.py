from flww.objects.Waveguide import Waveguide
from flww.compiler.PGMCompiler import PGMCompiler
import param as p
import matplotlib.pyplot as plt

# Simple script for fabricate a MZI interferometer

mzi = [Waveguide() for _ in range(2)]
increment = [p.swg_length, 0.0, 0.0]

for i, wg in enumerate(mzi):
    [xi, yi, zi] = [p.x0, p.y0-(0.5-i)*p.pitch, p.depth]
    
    wg.start([xi, yi, zi])
    wg.linear(increment, p.speed)
    wg.sin_mzi((-1)**i*p.d, p.radius, p.length_arm, p.speed)
    wg.linear(increment, p.speed)
    wg.end()

print(wg.M)

# Plot
fig, ax = plt.subplots()
for wg in mzi:
    x = wg.M['x']; y = wg.M['y']    
    ax.plot(x[:-1], y[:-1], color='k', linewidth=2.5)
plt.show()

# Compilation
gc = PGMCompiler('MZImultiscan.pgm', ind_rif=p.ind_rif)
gc.header()
gc.rpt(wg.num_scan)
for i, wg in enumerate(mzi):
    gc.comment(f'Modo: {i+1}')
    gc.point_to_instruction(wg.M)
gc.endrpt()
gc.compile_pgm()