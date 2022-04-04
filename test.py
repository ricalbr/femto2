import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

filename = "test.pgm"
nw = 1.33
ng = 1.50
neff = nw/ng
speed = 20

radius = 15
pitch = 0.080
depth = 0.035
int_dist = 0.007
int_length = 0.0
tilt_angle = 0.1
tot_length = 90

#%% INITIAL CONDITIONS
xi = -2
yi = pitch/2
zi = depth

#%% CALCULATIONS
angle = np.arccos(1 - (pitch - int_dist)/(4 * radius))
l_straight = (tot_length - int_dist - 4*radius*np.sin(angle))/2
d_bend = 0.5*(pitch-int_dist)

Dx = 0.08; Dy = 0.0; Dz = 0.0
increment = [Dx, Dy, Dz]
shutter = 1

#%% GUIDES

class Waveguide:
    def __init__(self, xi=-2, yi=0, zi=0, vpos=5):
        self.xi=xi
        self.yi=yi
        self.zi=zi
        self.vpos=vpos
        
        self._M = {}
        self.set_init_point()
      
    @property
    def M(self):
        return pd.DataFrame.from_dict(self._M)
      
    def set_init_point(self):
        self._M['x'] = [self.xi]
        self._M['y'] = [self.yi]
        self._M['z'] = [self.zi]
        self._M['f'] = [self.vpos]
        self._M['s'] = [0]
    
    def linear(self, increment, speed, shutter):
        self._M['x'].append(self._M['x'][-1] + increment[0])
        self._M['y'].append(self._M['y'][-1] + increment[1])
        self._M['z'].append(self._M['z'][-1] + increment[2])
        self._M['f'].append(speed)
        self._M['s'].append(shutter)
        
    def sin_bend(self, D, R, speed, shutter, N=100):
        dy = np.abs(D/2)
        a = np.arccos(1-(dy/R))
        dx = 2*R*np.sin(a)
        
        x = np.arange(self._M['x'][-1], self._M['x'][-1]+dx, dx/(N-1))
        omega = np.pi/dx
        self._M['x'].extend(x)
        self._M['y'].extend(self._M['y'][-1] + 0.5*D*(1 - np.cos(omega*(x - self._M['x'][-1]))))
        self._M['z'].extend(self._M['z'][-1]*np.ones(x.shape))
        self._M['f'].extend(speed*np.ones(x.shape))
        if shutter:
            s = np.ones_like(x)
        else:
            s = np.ones_like(x); s[-1] = 0
        self._M['s'].extend(s)

    
speed = 20
sht = 1

circ20 = 20*[Waveguide()]

wg1 = Waveguide()
wg1.linear(increment, speed, sht)
wg1.sin_bend(-d_bend, radius, speed, sht)
wg1.sin_bend(d_bend, radius, speed, sht)
wg1.linear(increment, speed, sht)
print(wg1.M)  

wg2 = Waveguide()
wg2.linear(increment, speed, sht)
wg2.sin_bend(-d_bend, radius, speed, sht)
wg2.sin_bend(d_bend, radius, speed, sht)
wg2.linear(increment, speed, sht)
print(wg2.M)

plt.figure()
plt.plot(wg1.M['x'], wg1.M['y'])
plt.show()

# WG1
# init_1 = [xi, -yi, zi, speed, 0]
# M.extend([init_1])
# M.extend(linear(M[-1], increment, shutter, speed))
# M.extend(sin_bend(M[-1], 
# M.extend(sin_bend(M[-1], d_bend, radius, speed, shutter))
# M.extend(linear(M[-1], increment, shutter, speed))
# # chiudi shutter

# # WG2
# init_2 = [xi, yi, zi, speed, 0]
# M.extend([init_2])
# M.extend(linear(M[-1], increment, shutter, speed))
# M.extend(sin_bend(M[-1], d_bend, radius, speed, shutter))
# M.extend(sin_bend(M[-1], -d_bend, radius, speed, shutter))
# M.extend(linear(M[-1], increment, shutter, speed))
# print(M.head())
