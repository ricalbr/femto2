import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

class Waveguide:
    def __init__(self, 
                 num_scan=None,
                 c_max=1200):
        
        self._num_scan=num_scan
        self._c_max=c_max

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
    def M(self):
        return pd.DataFrame.from_dict(self._M)
    
    # Methods
    def start(self, init_pos):
        assert np.size(init_pos) == 3, f'Given initial position is not valid. 3 values are required, {np.size(init_pos)} were given.'
        # assert che M sia vuota
        
        self._M['x'] = [init_pos[0]]
        self._M['y'] = [init_pos[1]]
        self._M['z'] = [init_pos[2]]
        self._M['f'] = [5]
        self._M['s'] = [0]
        
    def end(self):
        self._M['x'].append(self._M['x'][-1])
        self._M['y'].append(self._M['y'][-1])
        self._M['z'].append(self._M['z'][-1])
        self._M['f'].append(self._M['f'][-1])
        self._M['s'].append(0)
    
    def linear(self, increment, speed=0.0, shutter=1):
        self._M['x'].append(self._M['x'][-1] + increment[0])
        self._M['y'].append(self._M['y'][-1] + increment[1])
        self._M['z'].append(self._M['z'][-1] + increment[2])
        self._M['f'].append(speed)
        self._M['s'].append(shutter)
        
    def sin_bend(self, D, radius, speed=0.0, shutter=1, N=100):
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

        self._M['s'].extend(shutter*np.ones(new_x.shape))
            
    def acc_sin(self, D, radius, speed=0.0, shutter=1):
        self.sin_bend(D, radius, speed, shutter)
        self.sin_bend(-D, radius, speed, shutter)
    
    def mzi_sin(self, D, radius, L=0, speed=0.0, shutter=1):
        self.acc_sin(D, radius, speed, shutter)
        self.linear([L,0,0], speed, shutter)
        self.acc_sin(D, radius, speed, shutter)
    
    def arc_bend(self, D, radius, speed=0.0, shutter=1, N=100):
        pass
    
    def acc_arc():
        pass
    
    def mzi_arc():
        pass
    
    def __get_sbend_parameter(self, D, radius):
        dy = np.abs(D/2)
        a = np.arccos(1 - (dy/radius))
        dx = 2 * radius * np.sin(a)    
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
    
    def __compute_number_points(self):
        pass
    
if __name__ == '__main__':
    # %% GEOMETRICAL DATA
    filename = "test.pgm"
    speed = 20
    
    radius = 15
    pitch = 0.080
    depth = 0.035
    int_dist = 0.007
    int_length = 0.0
    tilt_angle = 0.1
    tot_length = 25
    length_arm = 1.5
    
    #%% CALCULATIONS
    angle = np.arccos(1 - (pitch - int_dist)/(4 * radius))
    l_straight = (tot_length - int_dist - 4*radius*np.sin(angle))/2
    d_bend = 0.5*(pitch-int_dist)
    Dx = 4; Dy = 0.0; Dz = 0.0
    increment = [Dx, Dy, Dz]
    
    coup = [Waveguide() for _ in range(2)]
    for index, wg in enumerate(coup):
        [xi, yi, zi] = [-2, -pitch/2 + index*pitch, depth]
        
        wg.start([xi, yi, zi])
        wg.linear(increment, speed)
        wg.mzi_sin((-1)**index*d_bend, radius, length_arm, speed)
        wg.linear(increment, speed)
        wg.end()
    
    c = wg.M
    print(wg.M)
    
    fig, ax = plt.subplots()
    for wg in coup:
        ax.plot(wg.M['x'], wg.M['y'], color='k', linewidth=2.5)
        
    plt.show()
