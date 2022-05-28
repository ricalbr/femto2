from femto import Waveguide, TrenchColumn, Marker, PGMCompiler
import matplotlib.pyplot as plt
import numpy as np
import os

# %% GEOMETRICAL DATA

# Circuit
glass_height = 25
glass_length = 25
x_init = -2
R = 15
pitch = 0.080
pitch_fa = 0.127
depth = 0.035
int_distance = 0.007
int_length = 0.0
length_arm = 0.0
speed = 20
swg_length = 3
increment = [swg_length, 0.0, 0.0]
N = 150

x0 = -2.0
y0 = 0.0
z0 = depth

ymzi = 0.08

d1 = 0.5*(pitch-int_distance)
d2 = pitch-int_distance

# Markers
lx = 1
ly = 0.05

# Trench
lt = 1
tspeed = 4
nboxz = 4
deltaz = 0.0015
zoff = 0.02
hbox = 0.075
nrpt = np.ceil((hbox+zoff)/deltaz)
speed_pos = 5
base_folder = r'C:\Users\Capable\Desktop'
CWD = os.path.dirname(os.path.abspath(__file__))

# %% G-CODE DATA
n_scan = 6
ind_glass = 1.5
ind_water = 1.33
ind_env = ind_glass/ind_water
angle = 0.0

# 20x20 circuit
circ = {
    'waveguide': [],
    'marker': [],
    'trench': [],
    }

# Guida dritta
wg = Waveguide(num_scan=n_scan)
wg.start([x0, y0, z0])
wg.linear([glass_length+4, 0.0, 0.0], speed=speed)
wg.end()
circ['waveguide'].append(wg)

# MZI
_, delta_x = wg.get_sbend_parameter(d1, R)
l_x = (glass_length + 4 - delta_x*4)/2
for i in range(6):
    wg = Waveguide(num_scan=n_scan)
    wg.start([x0, ymzi+i*pitch, z0])
    wg.linear([l_x, 0, 0], speed=speed)
    wg.arc_mzi((-1)**(i)*d1, R, speed=speed)
    wg.linear([l_x, 0, 0], speed=speed)
    wg.end()
    circ['waveguide'].append(wg)

# # Marker
pos = [glass_length/2, y0-pitch]
c = Marker()
c.cross(pos, ly=0.1)
circ['marker'].append(c)

# # Trench
col = TrenchColumn(x_c=glass_length/2,
                   y_min=ymzi,
                   y_max=ymzi+6*pitch-0.02)
col.get_trench(circ['waveguide'])
circ['trench'].append(col)

# Plot
fig, ax = plt.subplots()
for wg in circ['waveguide']:
    ax.plot(wg.x[:-1], wg.y[:-1], '-b', alpha=0.5, linewidth=0.5)

for c in circ['marker']:
    ax.plot(c.x[:-1], c.y[:-1], '-k', linewidth=1.25)

for col in circ['trench']:
    for t in col:
        ax.add_patch(t.patch)
plt.tight_layout(pad=0)
ax.set_aspect(10)
plt.show()

# Waveguide G-Code
with PGMCompiler('MZIs.pgm', ind_rif=ind_env, angle=angle) as gc:
    with gc.repeat(circ['waveguide'][0].num_scan):
        for wg in circ['waveguide']:
            gc.comment(f' +--- Modo: {i+1} ---+')
            gc.write(wg.points)

# Marker G-Code
with PGMCompiler('Markers.pgm', ind_rif=ind_env, angle=angle) as gc:
    for mk in circ['marker']:
        gc.write(mk.points)

# Trench G-Code
for col_index, col in enumerate(circ['trench']):
    # Generate G-Code for the column
    col_filename = os.path.join(os.getcwd(),
                                's-trench',
                                f'FARCALL{col_index:03}')
    with PGMCompiler(col_filename, ind_rif=ind_env, angle=angle) as gc:
        gc.trench(col, col_index, base_folder=base_folder)
