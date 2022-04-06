import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class Waveguide:
    def __init__(self, 
                 pitch=None, 
                 radius=None, 
                 num_scan=None, 
                 c_max=1200, 
                 l_sample=None, 
                 n_env=1.5/1.33, # TODO: importa come una costante
                 vpos=5):
        
        self._pitch=pitch
        self._radius=radius
        self._num_scan=num_scan
        self._c_max=c_max        
        self._l_sample=l_sample
        self._vpos=vpos
        self._n_env=n_env

        self._M = {}
    
    # Getters/Setters
    @property
    def num_scan(self):
        return self._num_scan
    
    @num_scan.setter
    def num_scan(self, num_scan):
        self._num_scan=num_scan
    
    @property
    def c_max(self):
        return self._c_max
    
    @c_max.setter
    def c_max(self, c_max):
        self._c_max=c_max
    
    @property
    def vpos(self):
        return self._vpos
    
    @vpos.setter
    def vpos(self, v):
        self._vpos=v
        
    @property
    def n_env(self):
        return self._n_env
    
    @n_env.setter
    def n_env(self, n_env):
        self._n_env=n_env
        
    @property
    def M(self):
        return pd.DataFrame.from_dict(self._M)
    
    # Methods
    def start(self, init_pos):
        assert np.size(init_pos) == 3, f'Given initial position is not valid. 3 values are required, {np.size(init_pos)} were given.'
        
        self._M['x'] = [init_pos[0]]
        self._M['y'] = [init_pos[1]]
        self._M['z'] = [init_pos[2]]
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

    def __get_sbend_parameter(self, D):
        assert self._radius is not None, "Try to compute S-bend parameter with R = None."
        dy = np.abs(D/2)
        a = np.arccos(1 - (dy/self._radius))
        dx = 2 * self._radius * np.sin(a)    
        return (a, dx)              
    
    def curvature(self):
        dx_dt = np.gradient(self._M['x'])
        dy_dt = np.gradient(self._M['y'])
        dz_dt = np.gradient(self._M['z'])
        
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        d2z_dt2 = np.gradient(dz_dt)
        
        num = np.sqrt((d2z_dt2*dy_dt - d2y_dt2*dz_dt)**2 + (d2x_dt2*dz_dt - d2z_dt2*dx_dt)**2 + (d2y_dt2*dx_dt - d2x_dt2*dy_dt)**2)
        den = (dx_dt*dx_dt + dy_dt*dy_dt + dz_dt*dz_dt)**1.5
        default_zero = np.ones(np.size(num))*np.inf
        curvature = np.divide(num, den, out=default_zero, where=den!=0) # only divide nonzeros else Inf
        return curvature
    
    def __cmd_per_second(self):
        pass
    
# %% GEOMETRICAL DATA
filename = "test.pgm"
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
print(wg.curvature())


# plt.figure()
# for wg in coup:
#     plt.plot(wg.M['x'], wg.M['y'], )
# plt.show()


# import plotly.graph_objects as go

# import plotly.io as pio
# pio.renderers.default = 'svg'
# # pio.renderers.default = 'browser'

# x = ['Product A', 'Product B', 'Product C']
# y = [20, 14, 23]

# fig = go.Figure(data=[go.Bar(
#             x=x, y=y,
#             text=y,
#             textposition='auto',
#         )])
# fig.show()