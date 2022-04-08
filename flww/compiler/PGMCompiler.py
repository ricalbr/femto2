from flww.objects.Waveguide import Waveguide
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class PGMCompiler:
    def __init__(self, filename, ind_rif, fab_line='CAPABLE', angle=0.0, long_pause=0.5, short_pause=0.15):
        
        self._filename = filename
        self._ind_rif=ind_rif
        self._fab_line=fab_line
        self._angle=angle
        self._long_pause=long_pause
        self._short_pause=short_pause
        
        self._TM = None
        self.compute_t_matrix()
        
        self._instructions = []
        
    # Getters/Setters
    @property
    def filename(self):
        return self._filename
    
    @filename.setter
    def filename(self, filename):
        self._filename=filename
        
    @property
    def ind_rif(self):
        return self._ind_rif
    
    @ind_rif.setter
    def ind_rif(self, ind_rif):
        self._ind_rif=ind_rif
        self.compute_t_matrix()
        
    @property
    def fab_line(self):
        return self._fab_line
    
    @fab_line.setter
    def fab_line(self, fab_line):
        self._fab_line=fab_line
        
    @property
    def angle(self):
        return self._angle
    
    @angle.setter
    def angle(self, angle):
        self._angle=angle
        self.compute_t_matrix()
        
    @property
    def long_pause(self):
        return self._long_pause
    
    @long_pause.setter
    def long_pause(self, long_pause):
        self._long_pause=long_pause
        
    @property
    def short_pause(self):
        return self._short_pause
    
    @short_pause.setter
    def short_pause(self, short_pause):
        self._short_pause=short_pause
    
    # Methods
    def compute_t_matrix(self):
        RM = np.array([[np.cos(self._angle), -np.sin(self._angle), 0],
                       [np.sin(self._angle), np.cos(self._angle), 0],
                       [0, 0, 1]])
        SM = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1/self._ind_rif]])
        self._TM = np.dot(SM,RM)
        
    def header(self):
        if self._fab_line.lower() == 'capable':
            self._instructions.append('ENABLE X Y Z\n')
            self._instructions.append('METRIC\n')
            self._instructions.append('SECONDS\n')
            self._instructions.append('G359\n')
            self._instructions.append('VELOCITY ON\n')
            self._instructions.append('PSOCONTROL X RESET\n')
            self._instructions.append('PSOOUTPUT X CONTROL 3 0\n')
            self._instructions.append('PSOCONTROL X OFF\n')
            self._instructions.append('ABSOLUTE\n')
            self._instructions.append('G17\n')
            self._instructions.append('DWELL 1\n')
            self._instructions.append('\n')
            self._instructions.append('\n')
        elif self._fab_line.lower() == 'fire':
            self._instructions.append('ENABLE X Y Z\n')
            self._instructions.append('METRIC\n')
            self._instructions.append('SECONDS\n')
            self._instructions.append('WAIT MODE NOWAIT\n')
            self._instructions.append('VELOCITY ON\n')
            self._instructions.append('PSOCONTROL X RESET\n')
            self._instructions.append('PSOCONTROL X OFF\n')
            self._instructions.append('ABSOLUTE\n')
            self._instructions.append('G17\n')
            self._instructions.append('DWELL 1\n')
            self._instructions.append('\n')
            self._instructions.append('\n')
    
    def comment(self, comstring):
        self._instructions.append(f'; {comstring}\n')
        
    def rpt(self, num):
        self._instructions.append(f'REPEAT {num}\n')
        
    def endrpt(self):
        self._instructions.append('ENDREPEAT\n\n')
    
    def point_to_instruction(self, M):
        x = M['x']; y = M['y']; z = M['z']; f = M['f']; s = M['s']
        
        coord = np.column_stack((x,y,z))
        t_coord = np.dot(self._TM, coord.T).T
        x, y, z = t_coord[:, 0], t_coord[:, 1], t_coord[:, 2]
        
        shutter_on = False
        
        for i in range(len(x)):
            if s[i] == 0 and shutter_on is False:
                # self._instructions += 'PSOCONTROL X OFF\n'
                self._instructions.append(f'LINEAR X{x[i]:.6f} Y{y[i]:.6f} Z{z[i]:.6f} F{f[i]:.6f}\n')
                self._instructions.append(f'DWELL {self._long_pause:.6f}\n\n')
            elif s[i] == 0 and shutter_on is True:
                self._instructions.append('PSOCONTROL X OFF\n')
                self._instructions.append(f'DWELL {self._short_pause:.6f}\n')
                shutter_on = False
            elif s[i] == 1 and shutter_on is False:
                self._instructions.append('PSOCONTROL X ON\n')
                self._instructions.append(f'LINEAR X{x[i]:.6f} Y{y[i]:.6f} Z{z[i]:.6f} F{f[i]:.6f}\n')
                shutter_on = True
            else:
                self._instructions.append(f'LINEAR X{x[i]:.6f} Y{y[i]:.6f} Z{z[i]:.6f} F{f[i]:.6f}\n')
        return (x,y,z,f,s)
    
    def compile_pgm(self):
        f = open(self._filename, "w")
        f.write(''.join(self._instructions))
        f.close()
        
if __name__ == '__main__':
    
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
    gc = PGMCompiler('MZImultiscan.pgm', ind_rif=1.5/1.33, angle=angle)
    gc.header()
    gc.rpt(wg.num_scan)
    for i, wg in enumerate(coup):    
        gc.comment(f'Modo: {i}')
        gc.point_to_instruction(wg.M)
    gc.endrpt()
    gc.compile_pgm()
