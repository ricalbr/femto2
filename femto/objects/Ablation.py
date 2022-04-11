from Waveguide import Waveguide
from femto.compiler.PGMCompiler import PGMCompiler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Marker(Waveguide):
    def __init__(self, lx, ly, num_scan=1):
        super(Marker, self).__init__(num_scan)

        self.lx = lx
        self.ly = ly
        self._M = {}
        
    def cross(self, position, speed=1, speed_pos=5):
        self.start(position)
        self.linear([-self.lx/2,0,0], speed=speed_pos, shutter=0)
        self.linear([self.lx,0,0], speed=speed)
        self.linear([-self.lx/2,0,0], speed=speed_pos, shutter=0)
        self.linear([0,-self.ly/2,0], speed=speed_pos, shutter=0)
        self.linear([0,self.ly,0], speed=speed)
        self.linear([0,-self.ly/2,0], speed=speed_pos, shutter=0)
        self.end(speed_pos)
        

if __name__ == '__main__':
    c = Marker(1, 0.60)
    c.cross([5,5,0.001])    