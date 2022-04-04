import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class Waveguide:
    def __init__(self, xi=None, yi=None, zi=None, xv=None, yv=None, vpos=5):
        self._xi=xi
        self._yi=yi
        self._zi=zi
        
        self._xv=xv
        self._yv=yv
        self._vpos=vpos

        self._M = {}
    
    @property
    def xi(self):
        return self._xi
    
    @xi.setter
    def xi(self, x):
        self._xi=x
        self.set_init_point()
    
    @property
    def yi(self):
        return self._yi
    
    @yi.setter
    def yi(self, y):
        self._yi=y
        self.set_init_point()
    
    @property
    def zi(self):
        return self._zi
    
    @zi.setter
    def zi(self, z):
        self._zi=z
        self.set_init_point()
    
    @property
    def vpos(self):
        return self._vpos
        self.set_init_point()
    
    @vpos.setter
    def vpos(self, v):
        self._vpos=v
        
    @property
    def M(self):
        return pd.DataFrame.from_dict(self._M)
    
    def start(self, init_pos):
        assert np.size(init_pos) == 3, f'Given initial position is not valid. 3 values are required, {np.size(init_pos)} were given.'
        
        self._xi = init_pos[0]; self._M['x'] = [self._xi]
        self._yi = init_pos[1]; self._M['y'] = [self._yi]
        self._zi = init_pos[2]; self._M['z'] = [self._zi]
        self._M['f'] = [self._vpos]
        self._M['s'] = [0]
        
    def end(self):
        self._M['x'].append(self._M['x'][-1])
        self._M['y'].append(self._M['y'][-1])
        self._M['z'].append(self._M['z'][-1])
        self._M['f'].append(self._M['f'][-1])
        self._M['s'].append(0)
    
    def linear(self, increment, speed, shutter):
        self._M['x'].append(self._M['x'][-1] + increment[0])
        self._M['y'].append(self._M['y'][-1] + increment[1])
        self._M['z'].append(self._M['z'][-1] + increment[2])
        self._M['f'].append(speed)
        self._M['s'].append(shutter)
        
    def sin_bend(self, D, radius, speed, shutter, N=100):
        a = np.arccos(1-(np.abs(D/2)/radius))
        dx = 2 * radius * np.sin(a)
        
        new_x = np.arange(self._M['x'][-1], self._M['x'][-1] + dx, dx/(N - 1))
        new_y = self._M['y'][-1] + 0.5 * D * (1 - np.cos(np.pi / dx * (new_x - self._M['x'][-1])))
        new_z = self._M['z'][-1]*np.ones(new_x.shape)
        
        # update coordinates
        self._M['x'].extend(new_x)
        self._M['y'].extend(new_y)
        self._M['z'].extend(new_z)
        
        # update speed array
        if np.size(speed) == 1:
            self._M['f'].extend(speed*np.ones(new_x.shape))
        elif np.size(speed) == np.size(new_x):
            self._M['f'].extend(speed)
        else:
            raise ValueError("Speed array is neither a single value nor array of appropriate size.")    

        # update shutter array
        if np.size(shutter) == 1:
            self._M['s'].extend(shutter*np.ones(new_x.shape))
        elif np.size(speed) == np.size(new_x):
            self._M['s'].extend(shutter)
        else:
            raise ValueError("Shutter array is neither a single value nor array of appropriate size.")    

# %% GEOMETRICAL DATA
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

speed = 20
sht = 1
p = 0.08
Dx = 4; Dy = 0.0; Dz = 0.0
increment = [Dx, Dy, Dz]

#%% CALCULATIONS
angle = np.arccos(1 - (pitch - int_dist)/(4 * radius))
l_straight = (tot_length - int_dist - 4*radius*np.sin(angle))/2
d_bend = 0.5*(pitch-int_dist)

# %% COUPLER
coup = [Waveguide() for _ in range(2)]
for index, wg in enumerate(coup):
    [xi, yi, zi] = [-2, -p/2 + index*p, 0.035]
    
    wg.start([xi, yi, zi])
    wg.linear(increment, speed, sht)
    wg.sin_bend((-1)**index*d_bend, radius, speed, sht)
    wg.sin_bend((-1)**(index-1)*d_bend, radius, speed, sht)
    wg.linear(increment, speed, sht)
    wg.end()

print(wg.M)

plt.figure()
for wg in coup:
    plt.plot(wg.M['x'], wg.M['y'], )
plt.show()