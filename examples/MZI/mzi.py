from femto import Waveguide, PGMCompiler
import matplotlib.pyplot as plt
import param as p
import numpy as np
np.set_printoptions(formatter={'float': "\t{: 0.6f}".format})


# Simple script for fabricate a MZI interferometer
mzi = [Waveguide(p.PARAMETERS_WG) for _ in range(2)]
increment = [p.swg_length, 0.0, 0.0]

for i, wg in enumerate(mzi):
    [xi, yi, zi] = [p.x0, p.y0-(0.5-i)*p.pitch, p.z0]

    wg.start([xi, yi, zi])\
        .linear(increment)\
        .sin_mzi((-1)**i*p.d, arm_length=p.length_arm)\
        .linear(increment)
    wg.end()

print(wg.points)

# Plot
fig, ax = plt.subplots()
for wg in mzi:
    ax.plot(wg.x[:-1], wg.y[:-1], '-k', linewidth=2.5)  # shutter on
    ax.plot(wg.x[-2:], wg.y[-2:], ':b', linewidth=1.5)  # shutter off
plt.show()

# Compilation
p.PARAMETERS_GC.filename = 'MZImultiscan.pgm'
with PGMCompiler(p.PARAMETERS_GC) as gc:
    with gc.repeat(p.PARAMETERS_WG.scan):
        for i, wg in enumerate(mzi):
            gc.comment(f'Modo: {i+1}')
            gc.write(wg.points)
