from femto.objects.Waveguide import Waveguide
# from femto.compiler.PGMCompiler import PGMCompiler
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from typing import List
vector: List[float] = list()


class Marker(Waveguide):
    def __init__(self, lx: float, ly: float, num_scan: int = 1):
        super(Marker, self).__init__(num_scan)

        self.lx = lx
        self.ly = ly
        self._M = {}

    def cross(self,
              position: vector,
              speed: float = 1,
              speed_pos: float = 5):
        """
        Cross marker

        The function computes the point of a cross marker of given widht along
        x- and y-direction.

        Parameters
        ----------
        position : vector
        Ordered coordinate list that specifies the cross position [mm].
            position[0] -> X
            position[1] -> Y
            position[2] -> Z
        speed : float, optional
            Shutter open transition speed [mm/s]. The default is 1.
        speed_pos : float, optional
            Shutter closed transition speed [mm/s]. The default is 5.

        Returns
        -------
        None.

        """
        self.start(position)
        self.linear([-self.lx/2, 0, 0], speed=speed_pos, shutter=0)
        self.linear([self.lx, 0, 0], speed=speed)
        self.linear([-self.lx/2, 0, 0], speed=speed_pos, shutter=0)
        self.linear([0, -self.ly/2, 0], speed=speed_pos, shutter=0)
        self.linear([0, self.ly, 0], speed=speed)
        self.linear([0, -self.ly/2, 0], speed=speed_pos, shutter=0)
        self.end(speed_pos)

    def ruler(self, y_ticks, speed=1, speed_pos=5):
        pass


if __name__ == '__main__':
    c = Marker(1, 0.60)
    c.cross([5, 5, 0.001])
