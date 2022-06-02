import numpy as np

from femto.Parameters import GcodeParameters, TrenchParameters, WaveguideParameters


# UTILITIES FUNCTIONS


def get_sbend_par(D, R):
    dy = np.abs(D / 2)
    a = np.arccos(1 - (dy / R))
    return 2 * R * np.sin(a)


# GEOMETRICAL DATA
MM = 20
NN = 20

PARAMETERS_WG = WaveguideParameters(
    scan=6,
    speed=20,
    depth=0.035,
    radius=15
)

x0 = -2.0
y0 = 0.0
z0 = PARAMETERS_WG.depth
swg_length = 3
increment = [swg_length, 0.0, 0.0]

pitch = 0.080
pitch_fa = 0.127
int_distance = 0.007
int_length = 0.0
length_arm = 0.0

d1 = 0.5 * (pitch - int_distance)
d2 = pitch - int_distance

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
    y_min=-0.1,
    y_max=19 * 0.08 + 0.1
)

# G-CODE DATA
PARAMETERS_GC = GcodeParameters(
    lab='CAPABLE',
    samplesize=(25, 25),
    angle=0.0
)
