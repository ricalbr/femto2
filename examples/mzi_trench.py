from __future__ import annotations

from femto.curves import sin
from femto.device import Layer
from femto.marker import Marker
from femto.trench import UTrenchColumn
from femto.waveguide import Waveguide

# WAVEGUIDE PARAMETERS
PARAM_WG = dict(
    scan=6,
    speed=20,
    depth=0.035,
    radius=15,
    pitch=0.080,
    pitch_fa=0.127,
    int_dist=0.007,
    int_length=0.0,
    arm_length=0.0,
    lsafe=3,
    x_init=-2.0,
    y_init=5.0,
    z_init=0.035,
    samplesize=(25, 10),
)

# MARKER PARAMETERS
PARAM_MK = dict(
    scan=1,
    speed=4,
    depth=0.001,
    speed_pos=5,
    lx=1,
    ly=0.05,
)

# TRENCH PARAMETERS
PARAM_TC = dict(
    n_pillars=3,
    length=1.0,
    nboxz=4,
    deltaz=0.0015,
    h_box=0.075,
    base_folder='',
)

# G-CODE PARAMETERS
PARAM_GC = dict(
    filename='MZI.pgm',
    laser='pharos',
    samplesize=(25, 10),
    rotation_angle=1.0,
)

# Create Device object that will contain waveguides, markers and trenches
circ = Layer(**PARAM_GC)

# WAVEGUIDES
# Collect all the waveguides in a list
wgs = []

# SWG
wg = Waveguide(**PARAM_WG)
wg.start().linear([PARAM_WG['samplesize'][0] + 4, 0.0, 0.0], mode='INC').end()
wgs.append(wg)

# MZIs
delta_x = wg.dx_bend
l_x = (PARAM_WG['samplesize'][0] + 4 - delta_x * 4) / 2
for i in range(6):
    wg = Waveguide(**PARAM_WG)
    wg.start([wg.x_init, wg.y_init + (i + 1) * wg.pitch, wg.z_init])
    wg.linear([l_x, 0, 0])
    wg.mzi(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
    wg.linear([wg.x_end, wg.lasty, wg.lastz], mode='ABS')
    wg.end()
    wgs.append(wg)

# After the for loop add all the waveguides to the circuit
circ.extend(wgs)

# MARKER
c = Marker(**PARAM_MK)
c.cross([PARAM_GC['samplesize'][0] / 2, 4.0])
circ.append(c)

# TRENCH
xc = PARAM_GC['samplesize'][0] / 2
ymin = wg.y_init
ymax = wg.y_init + 6 * wg.pitch
col = UTrenchColumn(x_center=xc, y_min=ymin, y_max=ymax, **PARAM_TC)
col.dig_from_waveguide(wgs)
circ.append(col)

# PLOT AND G-CODE EXPORT
# Use Device's built-in functions to generate G-code scripts for the waveguides, markers and trench it contains
circ.plot3d()
circ.pgm()
