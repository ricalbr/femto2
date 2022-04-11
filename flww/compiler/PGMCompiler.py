from flww.objects.Waveguide import Waveguide
import numpy as np 
import matplotlib.pyplot as plt
import os

CWD = os.path.dirname(os.path.abspath(__file__))

class PGMCompiler:
    def __init__(self, filename, ind_rif, fab_line='CAPABLE', angle=0.0, long_pause=0.5, short_pause=0.15, output_digits=6):
    
        self.filename = filename
        self.fab_line=fab_line
        self.long_pause=long_pause
        self.short_pause=short_pause
        
        self.output_digits=output_digits
        self.total_dwell_time=0.0
        
        self.TM = None
        self._ind_rif=ind_rif
        self._angle=angle
        self._compute_t_matrix()
        
        self._shutter_on = False
        self._instructions = []
        
    # Getters/Setters
    @property
    def ind_rif(self):
        return self._ind_rif
    
    @ind_rif.setter
    def ind_rif(self, ind_rif):
        self._ind_rif=ind_rif
        self._compute_t_matrix()
        
    @property
    def angle(self):
        return self._angle
    
    @angle.setter
    def angle(self, angle):
        self._angle=angle
        self._compute_t_matrix()
    
    # Methods
    def _compute_t_matrix(self):
        RM = np.array([[np.cos(self.angle), -np.sin(self.angle), 0],
                       [np.sin(self.angle), np.cos(self.angle), 0],
                       [0, 0, 1]])
        SM = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1/self.ind_rif]])
        self.TM = np.dot(SM,RM)
        
    def header(self):
        if self.fab_line.upper() == 'CAPABLE':
            with open(os.path.join(CWD, 'header_capable.txt')) as fd:
                self._instructions.extend(fd.readlines())
        elif self.fab_line.upper() == 'FIRE':
            with open(os.path.join(CWD, 'header_fire.txt')) as fd:
                self._instructions.extend(fd.readlines())
    
    def comment(self, comstring):
        self._instructions.append(f'; {comstring}\n')
        
    def shutter(self, state):
        if state.upper() == 'ON' and self._shutter_on is False:
            self._shutter_on = True
            self._instructions.append('PSOCONTROL X ON\n')
        elif state.upper() == 'OFF' and self._shutter_on is True:
            self._shutter_on = False
            self._instructions.append('PSOCONTROL X OFF\n')
        else:
            pass
    
    def rpt(self, num):
        self._instructions.append(f'REPEAT {num}\n')
        
    def endrpt(self):
        self._instructions.append('ENDREPEAT\n\n')
    
    def dwell(self, time):
        self._instructions.append(f'DWELL {time}\n\n')
        self.total_dwell_time += float(time)
        
    def set_home(self, home_pos):
        assert self._shutter_on is False, 'Try to move with OPEN shutter.'
        assert np.size(home_pos) == 3, f'Given final position is not valid. 3 values are required, {np.size(home_pos)} were given.'
        
        x,y,z = home_pos
        args = self._format_args(x, y, z)
        space = ' ' if len(args) > 0 else ''
        self._instructions.append('G92' + space + args + '\n')
        
    def homing(self):
        self.comment('HOMING\n')
        self.move_to([0,0,0])
        
    def _format_args(self, x=None, y=None, z=None, f=None):
        args = []
        if x is not None:
            args.append('{0}{1:.{digits}f}'.format('X', x, digits=self.output_digits))
        if y is not None:
            args.append('{0}{1:.{digits}f}'.format('Y', y, digits=self.output_digits))
        if z is not None:
            args.append('{0}{1:.{digits}f}'.format('Z', z, digits=self.output_digits))
        if f is not None:
            args.append('{0}{1:.{digits}f}'.format('F', f, digits=self.output_digits))
        args = ' '.join(args)
        return args
    
    def move_to(self, position, speed_pos=50):
        assert self._shutter_on is False, 'Try to move with OPEN shutter.'
        assert np.size(position) == 3, f'Given final position is not valid. 3 values are required, {np.size(position)} were given.'
        
        x, y, z = position
        args = self._format_args(x, y, z, speed_pos)
        
        self._instructions.append(f'LINEAR {args}\n')
        self.dwell(self.long_pause)
    
    def point_to_instruction(self, M):
        c = np.column_stack((M['x'],M['y'],M['z']))
        c_rot = np.dot(self.TM, c.T).T
        x,y,z = c_rot[:, 0],c_rot[:, 1],c_rot[:, 2]; f = M['f']; s = M['s']
        
        for i in range(len(x)): 
            args = self._format_args(x[i], y[i], z[i], f[i])
            if s[i] == 0 and self._shutter_on is False:
                self._instructions.append(f'LINEAR {args}\n')
                self.dwell(self.long_pause)
            elif s[i] == 0 and self._shutter_on is True:
                self.shutter('OFF')
                self.dwell(self.short_pause)
            elif s[i] == 1 and self._shutter_on is False:
                self.shutter('ON')
                self._instructions.append(f'LINEAR {args}\n')
            else:
                self._instructions.append(f'LINEAR {args}\n')
        return (x,y,z,f,s)
    
    def compile_pgm(self):
        
        if self.filename is None:
            return
        
        if not self.filename.endswith('.pgm'):
            self.filename += '.pgm'
        
        f = open(self.filename, "w")
        f.write(''.join(self._instructions))
        f.close()
        print('G-code compilation completed.')
        
if __name__ == '__main__':
    
    # Data
    pitch = 0.080
    int_dist = 0.007
    angle = np.radians(1)
    ind_rif = 1.5/1.33
    
    d_bend = 0.5*(pitch-int_dist)
    increment = [4,0,0]
    
    # Calculations
    coup = [Waveguide(num_scan=6) for _ in range(2)]
    for index, wg in enumerate(coup):
        wg.start([-2, -pitch/2 + index*pitch, 0.035])
        wg.linear(increment, speed=20)
        wg.sin_mzi((-1)**index*d_bend, radius=15, arm_length=1.0, speed=20, N=50)
        wg.linear(increment, speed=20)
        wg.end()
    
    # Compilation 
    gc = PGMCompiler('testPGMcompiler', ind_rif=ind_rif, angle=angle)
    gc.header()
    gc.rpt(wg.num_scan)
    for i, wg in enumerate(coup):    
        gc.comment(f'Modo: {i}')
        gc.point_to_instruction(wg.M)
    gc.endrpt()
    gc.move_to([None,0,0.1])
    gc.set_home([0,0,0])
    gc.homing()
    gc.compile_pgm()