import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Waveguide:
    def __init__(self, num_scan=None, c_max=1200):

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
        return self._unique_points()

    def _unique_points(self):
        """
        Remove all consecutive duplicates.

        At least one coordinate (X,Y,Z,F,S) have to change between two
        consecutive lines.

        Duplicates can be selected by crating a boolean index mask as follow:
            - make a row-wise diff operation (data.diff)
            - make a column-wise sum (.sum(axis=1))
        In this way consecutive duplicates correspond to a 0.0 value in the
        latter array.
        Converting this array to boolean (all non-zero values are True) the
        index mask can be retrieved.
        The first element is set to True by default since it is lost by the
        diff operation.

        Also indexes have to be resetted in order

        Returns
        -------
        pandas DataFrame
            Coordinate dataframe (x, y, z, f, s).

        """

        data = pd.DataFrame.from_dict(self._M)
        mask = (data.diff()             # row-wise diff
                    .sum(axis=1)        # col-wise sum
                    .astype('bool'))    # cast to bool
        mask[0] = True
        return data[mask].reset_index()

    # Methods
    def start(self, init_pos):
        assert np.size(init_pos) == 3, f'Given initial position is not valid. 3 values are required, {np.size(init_pos)} were given.'
        assert bool(self._M) == False, 'Coordinate matrix is not empty. Cannot start a new waveguide in this point.'

        self._M['x'] = [init_pos[0]]
        self._M['y'] = [init_pos[1]]
        self._M['z'] = [init_pos[2]]
        self._M['f'] = [5]
        self._M['s'] = [0]

    def end(self, speed=75):
        self._M['x'].append(self._M['x'][-1]);  self._M['x'].append(self._M['x'][0]);
        self._M['y'].append(self._M['y'][-1]);  self._M['y'].append(self._M['y'][0]);
        self._M['z'].append(self._M['z'][-1]);  self._M['z'].append(self._M['z'][0]);
        self._M['f'].append(self._M['f'][-1]);  self._M['f'].append(speed);
        self._M['s'].append(0);                 self._M['s'].append(0);

    def linear(self, increment, speed=0.0, shutter=1):
        self._M['x'].append(self._M['x'][-1] + increment[0])
        self._M['y'].append(self._M['y'][-1] + increment[1])
        self._M['z'].append(self._M['z'][-1] + increment[2])
        self._M['f'].append(speed)
        self._M['s'].append(shutter)

    def sin_bend(self, D, radius, speed=0.0, shutter=1, N=25):
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

    def sin_acc(self, D, radius, arm_length=0.0, speed=0.0, shutter=1, N=50):
        self.sin_bend(D, radius, speed, shutter, N/2)
        self.linear([arm_length,0,0], speed, shutter)
        self.sin_bend(-D, radius, speed, shutter, N/2)

    def sin_mzi(self, D, radius, int_length=0.0, arm_length=0.0, speed=0.0, shutter=1, N=100):
        self.sin_acc(D, radius, int_length, speed, shutter, N/2)
        self.linear([arm_length,0,0], speed, shutter)
        self.sin_acc(D, radius, int_length, speed, shutter, N/2)

    def circ(self, initial_angle, final_angle, radius, speed=0.0, shutter=1, N=25):

        t = np.linspace(initial_angle, final_angle, N);
        new_x = self._M['x'][-1] - radius*np.cos(initial_angle) + radius*np.cos(t);
        new_y = self._M['y'][-1] - radius*np.sin(initial_angle) + radius*np.sin(t);
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

    def arc_bend(self, D, radius, speed=0.0, shutter=1, N=25):
        (a, _) = self.__get_sbend_parameter(D, radius)

        if D>0:
            self.circ(np.pi*(3/2), np.pi*(3/2)+a, radius, speed, shutter, np.round(N/2))
            self.circ(np.pi*(1/2)+a, np.pi*(1/2), radius, speed, shutter, np.round(N/2))
        else:
            self.circ(np.pi*(1/2), np.pi*(1/2)-a, radius, speed, shutter, np.round(N/2))
            self.circ(np.pi*(3/2)-a, np.pi*(3/2), radius, speed, shutter, np.round(N/2))

    def arc_acc(self, D, radius, arm_length=0.0, speed=0.0, shutter=1, N=50):
        self.arc_bend(D, radius, speed, shutter, N/2)
        self.linear([arm_length,0,0], speed, shutter)
        self.arc_bend(-D, radius, speed, shutter, N/2)

    def arc_mzi(self, D, radius, int_length=0.0, arm_length=0.0, speed=0.0, shutter=1, N=100):
        self.arc_acc(D, radius, arm_length, speed, shutter, N/2)
        self.linear([int_length,0,0], speed, shutter)
        self.arc_acc(D, radius, arm_length, speed, shutter, N/2)

    def __get_sbend_parameter(self, D, radius):
        dy = np.abs(D/2)
        a = np.arccos(1 - (dy/radius))
        dx = 2 * radius * np.sin(a)
        return (a, dx)

    def curvature(self):
        data = self._unique_points()

        x = np.array(data['x'])
        y = np.array(data['y'])
        z = np.array(data['z'])

        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        dz_dt = np.gradient(z)

        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        d2z_dt2 = np.gradient(dz_dt)

        num = (dx_dt**2+ dy_dt**2 + dz_dt**2)**1.5
        den = np.sqrt((d2z_dt2*dy_dt - d2y_dt2*dz_dt)**2 + (d2x_dt2*dz_dt - d2z_dt2*dx_dt)**2 + (d2y_dt2*dx_dt - d2x_dt2*dy_dt)**2)
        default_zero = np.ones(np.size(num))*np.inf
        curvature = np.divide(num, den, out=default_zero, where=den!=0) # only divide nonzeros else Inf
        return curvature

    def cmd_rate(self):
        data = self._unique_points()

        x = np.array(data['x'][:-1]) # exclude last point, it's there just to close the shutter
        y = np.array(data['y'][:-1])
        z = np.array(data['z'][:-1])
        v = np.array(data['f'][:-1])

        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        dz_dt = np.gradient(z)
        dt = np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)

        default_zero = np.ones(np.size(dt))*np.inf
        cmd_rate = np.divide(v, dt, out=default_zero, where=v!=0) # only divide nonzeros else Inf
        return cmd_rate

    def __compute_number_points(self):
        pass

if __name__ == '__main__':

    # Data
    pitch = 0.080; int_dist = 0.007
    d_bend = 0.5*(pitch-int_dist)
    increment = [4, 0, 0]

    # Calculations
    mzi = [Waveguide() for _ in range(2)]
    for index, wg in enumerate(mzi):
        [xi, yi, zi] = [-2, -pitch/2 + index*pitch, 0.035]

        wg.start([xi, yi, zi])
        wg.linear(increment, speed=20)
        wg.sin_mzi((-1)**index*d_bend, radius=15, speed=20)
        wg.linear(increment, speed=20)
        wg.end()

    print(wg.M)


    # # Plot
    # fig, ax = plt.subplots()
    # for wg in mzi:
    #     ax.plot(wg.M['x'][:-1], wg.M['y'][:-1], '-k', linewidth=2.5)
    #     ax.plot(wg.M['x'][-2:], wg.M['y'][-2:], ':b', linewidth=1.0)
    # plt.show()
