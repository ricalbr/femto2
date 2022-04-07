from flww.objects.Waveguide import Waveguide
import matplotlib.pyplot as plt

# Simple script for fabricate a MZI interferometer

# %% GEOMETRICAL DATA
radius = 15
pitch = 0.080
depth = 0.035
int_distance = 0.007
int_length = 0.0
tilt_angle = 0.1
tot_length = 25
length_arm = 1.5
speed = 20

#%% CALCULATIONS
d = 0.5*(pitch-int_distance)
Dx = 4; Dy = 0.0; Dz = 0.0
increment = [Dx, Dy, Dz]

coup = [Waveguide() for _ in range(2)]
for i, wg in enumerate(coup):
    [xi, yi, zi] = [-2, -pitch/2 + i*pitch, depth]
    
    wg.start([xi, yi, zi])
    wg.linear(increment, speed)
    wg.mzi_sin((-1)**i*d, radius, length_arm, speed)
    wg.linear(increment, speed)
    wg.end()

c = wg.M
print(wg.M)

fig, ax = plt.subplots()
for wg in coup:
    ax.plot(wg.M['x'], wg.M['y'], color='k', linewidth=2.5)
    
plt.show()
