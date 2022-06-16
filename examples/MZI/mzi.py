import matplotlib.pyplot as plt
import numpy as np

from femto import Cell, Waveguide
from param import *

np.set_printoptions(formatter={'float': "\t{: 0.6f}".format})

# Simple script for fabricate a MZI interferometer
mzi = Cell()

for i in range(2):
    [xi, yi, zi] = [x0, y0 - (0.5 - i) * PARAMETERS_WG.pitch, z0]

    wg = Waveguide(PARAMETERS_WG)
    wg.start([xi, yi, zi]) \
        .linear(increment) \
        .sin_mzi((-1) ** i * d, arm_length=wg.arm_length) \
        .linear(increment)
    wg.end()
    mzi.add(wg)

# Plot
mzi.plot2d()
plt.show()

# Compilation
# PARAMETERS_GC.filename = 'MZImultiscan.pgm'
# with PGMCompiler(PARAMETERS_GC) as gc:
#     with gc.repeat(PARAMETERS_WG.scan):
#         for i, wg in enumerate(mzi.waveguides):
#             gc.comment(f'Modo: {i + 1}')
#             gc.write(wg.points)
