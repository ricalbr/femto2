import numpy as np
import os

# %% UTILITIES FUNCTIONS


def get_sbend_par(D, R):
    dy = np.abs(D/2)
    a = np.arccos(1-(dy/R))
    return 2*R*np.sin(a)


# %% GEOMETRICAL DATA
MM = 20
NN = 20

# Circuit
radius = 15
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
base_folder = r'C:\Users\Capable\Desktop\RiccardoA'
CWD = os.path.dirname(os.path.abspath(__file__))
# %% G-CODE DATA
n_scan = 6
ind_glass = 1.5
ind_water = 1.33
ind_env = ind_glass/ind_water
angle = 0.0