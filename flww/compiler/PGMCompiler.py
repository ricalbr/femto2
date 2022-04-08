import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class PGMCompiler:
    def __init__(self, filename, line='CAPABLE', angle=0.0, n_rif=None, long_pause=0.5, short_pause=0.15):
        
        self._filename = filename
        self._line=line
        self._long_pause=long_pause
        self._short_pause=short_pause
        self._angle=angle
        self._n_rif=n_rif

        self._RM = np.array([[np.cos(self._angle), -np.sin(self._angle), 0],
                             [np.sin(self._angle), np.cos(self._angle), 0],
                             [0, 0, 1]])
        self._SM = np.array([[1,0,0],
                              [0,1,0],
                              [0,0,1/self._n_rif]])
        self._instructions = ''
        
    def header(self):
        if self._line.lower() == 'capable':
            self._instructions += 'ENABLE X Y Z\n'
            self._instructions += 'METRIC\n'
            self._instructions += 'SECONDS\n'
            self._instructions += 'G359\n'
            self._instructions += 'VELOCITY ON\n'
            self._instructions += 'PSOCONTROL X RESET\n'
            self._instructions += 'PSOOUTPUT X CONTROL 3 0\n'
            self._instructions += 'PSOCONTROL X OFF\n'
            self._instructions += 'ABSOLUTE\n'
            self._instructions += 'G17\n'
            self._instructions += 'DWELL 1\n'
            self._instructions += '\n'
            self._instructions += '\n'
        elif self._line.lower() == 'fire':
            self._instructions += 'ENABLE X Y Z\n'
            self._instructions += 'METRIC\n'
            self._instructions += 'SECONDS\n'
            self._instructions += 'WAIT MODE NOWAIT\n'
            self._instructions += 'VELOCITY ON\n'
            self._instructions += 'PSOCONTROL X RESET\n'
            self._instructions += 'PSOCONTROL X OFF\n'
            self._instructions += 'ABSOLUTE\n'
            self._instructions += 'G17\n'
            self._instructions += 'DWELL 1\n'
            self._instructions += '\n'
            self._instructions += '\n'
    
    def comment(self, comstring):
        self._instructions += f'; {comstring}\n'
        
    def rpt(self, num):
        self._instructions += f'REPEAT {num}\n'
        
    def endrpt(self):
        self._instructions += 'ENDREPEAT\n\n'
    
    def point_to_instruction(self, M):
        x = M['x']
        y = M['y']
        z = M['z']
        f = M['f']
        s = M['s']
        
        TM = np.dot(self._SM, self._RM)
        coord = np.column_stack((x,y,z))
        t_coord = np.dot(TM, coord.T).T
        x, y, z = t_coord[:, 0], t_coord[:, 1], t_coord[:, 2]
        
        shutter_on = False
        
        for i in range(len(x)):
            if s[i] == 0 and shutter_on is False:
                # self._instructions += 'PSOCONTROL X OFF\n'
                self._instructions += f'LINEAR X{x[i]:.6f} Y{y[i]:.6f} Z{z[i]:.6f} F{f[i]:.6f}\n'
                self._instructions += f'DWELL {self._long_pause:.6f}\n\n'
            elif s[i] == 0 and shutter_on is True:
                self._instructions += 'PSOCONTROL X OFF\n'
                self._instructions += f'DWELL {self._short_pause:.6f}\n'
                shutter_on = False
            elif s[i] == 1 and shutter_on is False:
                self._instructions += 'PSOCONTROL X ON\n'
                self._instructions += f'LINEAR X{x[i]:.6f} Y{y[i]:.6f} Z{z[i]:.6f} F{f[i]:.6f}\n'
                shutter_on = True
            else:
                self._instructions += f'LINEAR X{x[i]:.6f} Y{y[i]:.6f} Z{z[i]:.6f} F{f[i]:.6f}\n'
        return (x,y,z,f,s)
    
    def compile_pgm(self):
        f = open(self._filename, "w")
        f.write(self._instructions)
        f.close()
        
if __name__ == '__main__':
    from flww.objects.Waveguide import Waveguide
    
    # Data
    speed = 20
    
    radius = 15
    pitch = 0.080
    depth = 0.035
    int_dist = 0.007
    int_length = 0.0
    angle = np.radians(45)
    tot_length = 25
    length_arm = 1.5
    
    d_bend = 0.5*(pitch-int_dist)
    Dx = 4; Dy = 0.0; Dz = 0.0
    increment = [Dx, Dy, Dz]
    
    # Calculations
    coup = [Waveguide(num_scan=6) for _ in range(20)]
    for index, wg in enumerate(coup):
        [xi, yi, zi] = [-2, -pitch/2 + index*pitch, depth]
        
        wg.start([xi, yi, zi])
        wg.linear(increment, speed)
        wg.mzi_sin((-1)**index*d_bend, radius, length_arm, speed,)
        wg.linear(increment, speed)
        wg.end()
    
    # Compilation 
    gc = PGMCompiler('MZImultiscan.pgm', n_rif=1.5/1.33, angle=angle)
    gc.header()
    gc.rpt(wg.num_scan)
    for i, wg in enumerate(coup):    
        gc.comment(f'Modo: {i}')
        gc.point_to_instruction(wg.M)
    gc.endrpt()
    gc.compile_pgm()