import os

import matplotlib.pyplot as plt

from femto import Marker, PGMCompiler, TrenchColumn, Waveguide
from femto.Parameters import GcodeParameters, TrenchParameters, WaveguideParameters

# GEOMETRICAL DATA

# Circuit
PARAMETERS_WG = WaveguideParameters(
    scan=6,
    speed=20,
    depth=0.035,
    radius=15,
)

x0 = -2.0
y0 = 0.0
z0 = PARAMETERS_WG.depth
swg_length = 3
increment = [swg_length, 0.0, 0.0]

pitch = 0.080
pitch_fa = 0.127
depth = 0.035
int_distance = 0.007
int_length = 0.0
length_arm = 0.0

d1 = 0.5 * (pitch - int_distance)

# Markers
lx = 1
ly = 0.05

# Trench
PARAMETERS_TC = TrenchParameters(
    lenght=1.0,
    nboxz=4,
    deltaz=0.0015,
    h_box=0.075,
    base_folder=r'C:\Users\Capable\Desktop\RiccardoA',
    y_min=0.08,
    y_max=0.08 + 6 * pitch - 0.02
)

# G-CODE DATA
PARAMETERS_GC = GcodeParameters(
    lab='CAPABLE',
    samplesize=(25, 25),
    angle=0.0
)

# 20x20 circuit
circ = {
    'waveguide': [],
    'marker': [],
    'trench': [],
}

# Guida dritta
wg = Waveguide(PARAMETERS_WG)
wg.start([x0, y0, z0]) \
    .linear([PARAMETERS_GC.xsample + 4, 0.0, 0.0])
wg.end()
circ['waveguide'].append(wg)

# MZI
_, delta_x = wg.get_sbend_parameter(d1, PARAMETERS_WG.radius)
l_x = (PARAMETERS_GC.xsample + 4 - delta_x * 4) / 2
for i in range(6):
    wg = Waveguide(PARAMETERS_WG) \
        .start([x0, (1 + i) * pitch, z0]) \
        .linear([l_x, 0, 0]) \
        .arc_mzi((-1) ** i * d1) \
        .linear([l_x, 0, 0])
    wg.end()
    circ['waveguide'].append(wg)

# # Marker
pos = [PARAMETERS_GC.xsample / 2, y0 - pitch]
c = Marker()
c.cross(pos, ly=0.1)
circ['marker'].append(c)

# # Trench
PARAMETERS_TC.x_center = PARAMETERS_GC.xsample / 2
col = TrenchColumn(PARAMETERS_TC)
col.get_trench(circ['waveguide'])
circ['trench'].append(col)

# Plot
fig, ax = plt.subplots()
for wg in circ['waveguide']:
    ax.plot(wg.x[:-1], wg.y[:-1], '-b', alpha=0.5, linewidth=0.5)

for c in circ['marker']:
    ax.plot(c.x[:-1], c.y[:-1], '-k', linewidth=1.25)

for col_trench in circ['trench']:
    for t in col_trench:
        ax.add_patch(t.patch)
plt.tight_layout(pad=0)
ax.set_aspect(10)
# plt.show()

# Waveguide G-Code
PARAMETERS_GC.filename = 'MZIs.pgm'
with PGMCompiler(PARAMETERS_GC) as gc:
    with gc.repeat(PARAMETERS_WG.scan):
        for i, wg in enumerate(circ['waveguide']):
            gc.comment(f' +--- Modo: {i + 1} ---+')
            gc.write(wg.points)

# Marker G-Code
PARAMETERS_GC.filename = 'Markers.pgm'
with PGMCompiler(PARAMETERS_GC) as gc:
    for mk in circ['marker']:
        gc.write(mk.points)

# Trench G-Code
for col_index, col_trench in enumerate(circ['trench']):
    # Generate G-Code for the column
    col_filename = os.path.join(os.getcwd(),
                                's-trench',
                                f'FARCALL{col_index + 1:03}')
    PARAMETERS_GC.filename = col_filename
    with PGMCompiler(PARAMETERS_GC) as gc:
        gc.trench(col_trench, col_index, base_folder=PARAMETERS_TC.base_folder)
