from femto.Parameters import GcodeParameters, WaveguideParameters

# GEOMETRICAL DATA

PARAMETERS_WG = WaveguideParameters(
    scan=6,
    speed=20,
    radius=15,
    pitch=0.080,
    int_dist=0.007,
    lsafe=3
)

x0 = -2.0
y0 = 0.0
z0 = PARAMETERS_WG.depth
increment = [PARAMETERS_WG.lsafe, 0.0, 0.0]

d = 0.5 * (PARAMETERS_WG.pitch - PARAMETERS_WG.int_dist)

# G-CODE DATA
PARAMETERS_GC = GcodeParameters(
    lab='CAPABLE',
    samplesize=(25, 25),
    angle=0.0
)
