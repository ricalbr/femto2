from femto.helpers import dotdict

# GEOMETRICAL DATA
MM = 20
NN = 20

PARAMETERS_WG = dotdict(
    scan=6,
    speed=20,
    radius=15,
    pitch=0.080,
    int_dist=0.007,
    lsafe=3
)

x0 = -2.0
y0 = 0.0
increment = [PARAMETERS_WG.lsafe, 0.0, 0.0]

d1 = 0.5 * (PARAMETERS_WG.pitch - PARAMETERS_WG.int_dist)
d2 = PARAMETERS_WG.pitch - PARAMETERS_WG.int_dist

# Markers
PARAMETERS_MK = dotdict(
    scan=1,
    speed=4,
    depth=0.001,
    speedpos=5,
)
lx = 1
ly = 0.05

# Trench
PARAMETERS_TC = dotdict(
    lenght=1.0,
    nboxz=4,
    deltaz=0.0015,
    h_box=0.075,
    base_folder=r'C:\Users\Capable\Desktop\RiccardoA',
)

# G-CODE DATA
PARAMETERS_GC = dotdict(
    lab='CAPABLE',
    samplesize=(25, 25),
    angle=0.0
)
