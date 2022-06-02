from femto.Parameters import GcodeParameters, WaveguideParameters

# GEOMETRICAL DATA

PARAMETERS_WG = WaveguideParameters(
    scan=6,
    speed=20,
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

d = 0.5 * (pitch - int_distance)

# G-CODE DATA
PARAMETERS_GC = GcodeParameters(
    lab='CAPABLE',
    samplesize=(25, 25),
    angle=0.0
)
