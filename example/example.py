from flww.objects.Waveguide import Waveguide
import param as p
import matplotlib.pyplot as plt

# Simple script for fabricate a MZI interferometer

mzi = [Waveguide() for _ in range(2)]
increment = [p.swg_length, 0.0, 0.0]

for i, wg in enumerate(mzi):
    [xi, yi, zi] = [p.x0, p.y0-(0.5-i)*p.pitch, p.depth]
    
    wg.start([xi, yi, zi])
    wg.linear(increment, p.speed)
    wg.mzi_sin((-1)**i*p.d, p.radius, p.length_arm, p.speed)
    wg.linear(increment, p.speed)
    wg.end()

print(wg.M)

fig, ax = plt.subplots()
for wg in mzi:
    ax.plot(wg.M['x'], wg.M['y'], color='k', linewidth=2.5)
plt.show()